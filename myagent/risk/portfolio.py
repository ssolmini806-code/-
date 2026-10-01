"""
포지션 · 계좌 상태 관리 및 서킷 브레이커.

여기서 강제되는 한도는 LLM 판단과 무관하게 항상 작동한다.
TradeTrap 연구에서 계좌 상태 교란 시 에이전트가 단일 종목 100% 집중,
최대 낙폭 92%까지 간 사례가 보고됐다. 그 실패는 이 계층이 없을 때 일어난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import ExitReason, PlannedTrade, RiskConfig, Side, should_time_stop, update_trailing_stop


@dataclass
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_bar: int
    stop_price: float
    target_price: float
    stop_distance: float
    favorable_extreme: float
    initial_stop_price: float = 0.0
    bars_held: int = 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        diff = (price - self.entry_price) if self.side is Side.LONG else (self.entry_price - price)
        return diff * self.quantity

    def r_multiple(self, price: float) -> float:
        """현재 손익을 R 배수로. 1R = 최초 손절폭."""
        if self.stop_distance <= 0:
            return 0.0
        return self.unrealized_pnl(price) / (self.stop_distance * self.quantity)


@dataclass
class ClosedTrade:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    pnl: float
    r_multiple: float
    reason: ExitReason


@dataclass
class Portfolio:
    """
    계좌 상태의 단일 소유자(single source of truth).

    equity는 반드시 mark_to_market()을 통해서만 갱신한다.
    외부에서 직접 대입하면 서킷 브레이커가 무력화된다.
    """

    initial_equity: float
    cfg: RiskConfig
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    halted: bool = False
    halt_started_bar: int | None = None
    bar_index: int = 0
    equity_curve: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity는 양수여야 합니다")
        self.cash = self.initial_equity
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

    # ------------------------------------------------------------------ 상태

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return 1.0 - (self.equity / self.peak_equity)

    def gross_exposure(self, prices: dict[str, float] | None = None) -> float:
        """총 익스포저 비율. prices를 주면 시가평가, 없으면 진입가 기준."""
        if self.equity <= 0:
            return 0.0
        prices = prices or {}
        total = sum(
            abs(p.quantity * prices.get(s, p.entry_price))
            for s, p in self.positions.items()
        )
        return total / self.equity

    def _gross_notional(self, prices: dict[str, float]) -> float:
        return sum(
            abs(p.market_value(prices.get(s, p.entry_price)))
            for s, p in self.positions.items()
        )

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """봉 종료 시 1회 호출. 여기서만 equity가 갱신된다."""
        holdings = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            if pos.side is Side.LONG:
                holdings += pos.market_value(price)
            else:
                # 숏: 진입 명목 + 미실현손익
                holdings += pos.quantity * pos.entry_price + pos.unrealized_pnl(price)

        self.equity = self.cash + holdings
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        self._check_circuit_breaker()

    # ------------------------------------------------- 서킷 브레이커 (하드 리밋)

    def _check_circuit_breaker(self) -> None:
        if not self.halted and self.drawdown >= self.cfg.max_drawdown_halt:
            self.halted = True
            self.halt_started_bar = self.bar_index
            return

        if self.halted and self.halt_started_bar is not None:
            cooled = (self.bar_index - self.halt_started_bar) >= self.cfg.halt_cooldown_bars
            recovered = self.drawdown < self.cfg.max_drawdown_halt * 0.5
            if cooled and recovered:
                self.halted = False
                self.halt_started_bar = None

    # ------------------------------------------------------------------ 진입

    def can_open(self, plan: PlannedTrade, prices: dict[str, float]) -> tuple[bool, str]:
        """진입 가능 여부. 거부 사유를 함께 반환해 로그에 남긴다."""
        if self.halted:
            return False, f"서킷 브레이커 작동 중 (낙폭 {self.drawdown:.1%})"
        if not plan.is_valid:
            return False, plan.rejected_reason or "유효하지 않은 계획"
        if plan.symbol in self.positions:
            return False, "이미 보유 중인 종목"
        if len(self.positions) >= self.cfg.max_open_positions:
            return False, f"최대 보유 종목 수 초과 ({self.cfg.max_open_positions})"
        if plan.weight > self.cfg.max_position_weight + 1e-9:
            return False, f"종목 비중 한도 초과 ({plan.weight:.1%})"

        cost = plan.notional * (1 + self.cfg.slippage_rate + self.cfg.commission_rate)
        if cost > self.cash:
            return False, "현금 부족"

        projected = (self._gross_notional(prices) + plan.notional) / max(self.equity, 1e-9)
        if projected > self.cfg.max_gross_exposure + 1e-9:
            return False, f"총 익스포저 한도 초과 ({projected:.1%})"

        return True, ""

    def open_position(self, plan: PlannedTrade, prices: dict[str, float]) -> bool:
        ok, _ = self.can_open(plan, prices)
        if not ok:
            return False

        fill_price = plan.entry_price * (
            1 + self.cfg.slippage_rate if plan.side is Side.LONG else 1 - self.cfg.slippage_rate
        )
        cost = plan.quantity * fill_price * (1 + self.cfg.commission_rate)
        self.cash -= cost

        self.positions[plan.symbol] = Position(
            symbol=plan.symbol,
            side=plan.side,
            quantity=plan.quantity,
            entry_price=fill_price,
            entry_bar=self.bar_index,
            stop_price=plan.stop_price,
            target_price=plan.target_price,
            stop_distance=plan.stop_distance,
            favorable_extreme=fill_price,
            initial_stop_price=plan.stop_price,
        )
        return True

    # ------------------------------------------------------------------ 청산

    def close_position(self, symbol: str, price: float, reason: ExitReason) -> ClosedTrade | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        fill_price = price * (
            1 - self.cfg.slippage_rate if pos.side is Side.LONG else 1 + self.cfg.slippage_rate
        )
        pnl = pos.unrealized_pnl(fill_price)
        proceeds = pos.quantity * pos.entry_price + pnl
        self.cash += proceeds * (1 - self.cfg.commission_rate)

        trade = ClosedTrade(
            symbol=pos.symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            entry_bar=pos.entry_bar,
            exit_bar=self.bar_index,
            pnl=pnl,
            r_multiple=pos.r_multiple(fill_price),
            reason=reason,
        )
        self.closed_trades.append(trade)
        return trade

    # ------------------------------------------------- 봉마다 실행되는 청산 점검

    def process_bar(
        self,
        bars: dict[str, dict[str, float]],
        atr_values: dict[str, float] | None = None,
    ) -> list[ClosedTrade]:
        """
        청산 우선순위: 손절 > 익절 > 시간 손절.

        같은 봉에서 손절가와 익절가를 모두 건드린 경우 손절을 먼저 적용한다.
        일봉 데이터로는 어느 쪽이 먼저였는지 알 수 없으므로 보수적으로 처리한다.
        (이 가정을 뒤집으면 백테스트 성과가 실제보다 좋게 나온다)
        """
        atr_values = atr_values or {}
        exits: list[ClosedTrade] = []

        for symbol in list(self.positions.keys()):
            bar = bars.get(symbol)
            if bar is None:
                continue

            pos = self.positions[symbol]
            pos.bars_held += 1
            high, low, close = bar["high"], bar["low"], bar["close"]

            if pos.side is Side.LONG:
                pos.favorable_extreme = max(pos.favorable_extreme, high)
                hit_stop = low <= pos.stop_price
                hit_target = high >= pos.target_price
            else:
                pos.favorable_extreme = min(pos.favorable_extreme, low)
                hit_stop = high >= pos.stop_price
                hit_target = low <= pos.target_price

            if hit_stop:
                # 추적 손절로 이동한 뒤 맞은 것과, 최초 손절가에 맞은 것을 구분한다.
                # 구분하지 않으면 "손절 247건"에 이익 청산이 섞여 진단이 왜곡된다.
                moved = pos.stop_price != pos.initial_stop_price
                reason = ExitReason.TRAILING_STOP if moved else ExitReason.STOP_LOSS
                trade = self.close_position(symbol, pos.stop_price, reason)
                if trade:
                    exits.append(trade)
                continue

            if hit_target:
                trade = self.close_position(symbol, pos.target_price, ExitReason.TAKE_PROFIT)
                if trade:
                    exits.append(trade)
                continue

            if should_time_stop(
                pos.side, pos.entry_price, close, pos.stop_distance, pos.bars_held, self.cfg
            ):
                trade = self.close_position(symbol, close, ExitReason.TIME_STOP)
                if trade:
                    exits.append(trade)
                continue

            atr_now = atr_values.get(symbol)
            if atr_now and atr_now > 0:
                pos.stop_price = update_trailing_stop(
                    side=pos.side,
                    current_stop=pos.stop_price,
                    entry_price=pos.entry_price,
                    stop_distance=pos.stop_distance,
                    favorable_extreme=pos.favorable_extreme,
                    atr_value=atr_now,
                    cfg=self.cfg,
                )

        return exits

    # ------------------------------------------------------------------ 통계

    def stats(self) -> dict[str, float | int]:
        import math

        trades = self.closed_trades
        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        peak, max_dd = self.initial_equity, 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            max_dd = max(max_dd, 1.0 - eq / peak if peak > 0 else 0.0)

        rets = [
            self.equity_curve[i] / self.equity_curve[i - 1] - 1
            for i in range(1, len(self.equity_curve))
            if self.equity_curve[i - 1] > 0
        ]
        if len(rets) > 1:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            std = math.sqrt(var)
            sharpe = (mean / std) * math.sqrt(252) if std > 0 else 0.0
        else:
            sharpe = 0.0

        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        return {
            "trades": n,
            "win_rate": len(wins) / n if n else 0.0,
            "avg_r": sum(t.r_multiple for t in trades) / n if n else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
            "total_return": self.equity / self.initial_equity - 1.0,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "final_equity": self.equity,
            "halted_now": self.halted,
        }
