"""
단일 실행 경로.

AI Hedge Fund의 핵심 설계를 차용: BACKTEST / PAPER / LIVE가
전부 이 함수 하나를 탄다. 모드는 주입되는 어댑터(data, broker)로만 갈린다.

백테스트용 별도 코드를 짜는 순간, 검증한 것과 실제로 돌아가는 것이 달라진다.
개인 개발자가 가장 흔히 죽는 지점이 여기다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.contract import PricePanel
from risk.portfolio import Portfolio
from risk.rules import RiskConfig, Side, plan_trade
from signals.models import AlphaModel


@dataclass
class PendingOrder:
    """
    t봉에 결정되고 t+1봉 시가에 체결될 주문.

    계획을 그대로 들고 가지 않고 신호와 ATR만 들고 간다.
    체결가(t+1 시가)를 알게 된 시점에 손절가와 수량을 다시 계산해야
    갭이 발생해도 1회 위험이 의도한 금액으로 유지된다.

    이렇게 하지 않으면 시가 +5% 갭에서 실제 위험이 의도의 2.25배가 되어
    risk_per_trade 규칙 전체가 무력화된다.
    """

    symbol: str
    side: "Side"
    conviction: float
    atr_at_decision: float
    decision_close: float


@dataclass
class CycleLog:
    """봉마다 무슨 일이 있었는지. 사후 진단이 안 되면 개선도 안 된다."""

    bar: int
    signals: int = 0
    opened: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    exits: int = 0
    equity: float = 0.0
    halted: bool = False


def run_backtest(
    panel: PricePanel,
    model: AlphaModel,
    cfg: RiskConfig,
    initial_equity: float,
    *,
    start_bar: int | None = None,
    end_bar: int | None = None,
    collect_logs: bool = False,
    max_entry_gap: float = 0.03,
) -> tuple[Portfolio, list[CycleLog]]:
    """
    실행 순서 (이 순서가 룩어헤드를 막는다):

      t봉 종가에 판단  →  t+1봉 시가에 체결

    t봉 종가에 즉시 체결한다고 가정하면 실제로는 불가능한 체결을 얻게 되고,
    백테스트 성과가 조용히 부풀려진다.
    """
    warmup = max(model.warmup_bars, cfg.atr_period + 5)
    start = max(warmup, start_bar or 0)
    end = min(panel.n_bars - 1, end_bar if end_bar is not None else panel.n_bars - 1)

    pf = Portfolio(initial_equity=initial_equity, cfg=cfg)
    logs: list[CycleLog] = []
    pending: list[PendingOrder] = []  # t봉에 결정되어 t+1봉에 체결될 주문

    from risk.indicators import atr as atr_fn

    atr_cache = {
        s: atr_fn(panel.frames[s]["high"], panel.frames[s]["low"], panel.frames[s]["close"], cfg.atr_period)
        for s in panel.symbols
    }

    for t in range(start, end + 1):
        pf.bar_index = t
        log = CycleLog(bar=t)

        bars_now = {s: panel.bar(s, t, as_of=t) for s in panel.symbols}
        atr_now = {
            s: float(atr_cache[s].iloc[t])
            for s in panel.symbols
            if not _isnan(atr_cache[s].iloc[t])
        }

        # 1) 전 봉에 결정된 주문을 이번 봉 시가에 체결
        #    체결가가 확정된 지금 손절가·수량을 재계산한다(갭 대응)
        price_map = {s: b["open"] for s, b in bars_now.items()}
        for order in pending:
            fill_ref = bars_now[order.symbol]["open"]

            # 갭 가드: 결정 시점 종가에서 크게 벌어지면 진입을 포기한다.
            # 이미 움직여버린 뒤에 쫓아 들어가는 것을 막는다.
            gap = abs(fill_ref / order.decision_close - 1.0)
            if gap > max_entry_gap:
                log.rejected[f"진입 갭 과대({gap:.1%})"] = (
                    log.rejected.get(f"진입 갭 과대({gap:.1%})", 0) + 1
                )
                continue

            plan = plan_trade(
                symbol=order.symbol,
                side=order.side,
                equity=pf.equity,
                entry_price=fill_ref,
                atr_value=order.atr_at_decision,
                cfg=cfg,
                available_cash=pf.cash,
                conviction=order.conviction,
            )
            if not plan.is_valid:
                log.rejected["재계산 후 수량 0"] = log.rejected.get("재계산 후 수량 0", 0) + 1
                continue

            if pf.open_position(plan, price_map):
                log.opened += 1
            else:
                _, why = pf.can_open(plan, price_map)
                log.rejected[why] = log.rejected.get(why, 0) + 1
        pending = []

        # 2) 보유 포지션 청산 점검
        exits = pf.process_bar(bars_now, atr_now)
        log.exits = len(exits)

        # 3) 이번 봉 종가로 다음 봉 주문 결정
        pf.mark_to_market({s: b["close"] for s, b in bars_now.items()})
        log.equity = pf.equity
        log.halted = pf.halted

        if not pf.halted and t < end:
            signals = model.generate(panel, as_of=t)
            log.signals = len(signals)
            for sig in signals:
                if sig.symbol in pf.positions:
                    continue
                a = atr_now.get(sig.symbol)
                if not a or a <= 0:
                    continue
                # 계획을 확정하지 않고 신호만 넘긴다. 사이징은 체결가 확정 후.
                pending.append(
                    PendingOrder(
                        symbol=sig.symbol,
                        side=sig.side,
                        conviction=sig.conviction,
                        atr_at_decision=a,
                        decision_close=bars_now[sig.symbol]["close"],
                    )
                )

        if collect_logs:
            logs.append(log)

    # 잔여 포지션 청산
    final_bar = {s: panel.bar(s, end, as_of=end) for s in panel.symbols}
    from risk.rules import ExitReason

    for symbol in list(pf.positions.keys()):
        pf.close_position(symbol, final_bar[symbol]["close"], ExitReason.MANUAL)
    pf.mark_to_market({s: b["close"] for s, b in final_bar.items()})

    return pf, logs


def _isnan(x) -> bool:
    return x != x


def buy_and_hold_benchmark(panel: PricePanel, start_bar: int, end_bar: int) -> float:
    """동일가중 매수 후 보유 수익률. 전략이 이걸 못 이기면 존재 이유가 없다."""
    rets = []
    for s in panel.symbols:
        df = panel.frames[s]
        p0, p1 = float(df["close"].iloc[start_bar]), float(df["close"].iloc[end_bar])
        if p0 > 0:
            rets.append(p1 / p0 - 1.0)
    return sum(rets) / len(rets) if rets else 0.0
