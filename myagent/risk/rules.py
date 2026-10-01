"""
손절가 산정 · 포지션 사이징 · 시간 손절.

설계 원칙:
  1. LLM은 이 모듈의 어떤 숫자도 결정하지 않는다.
     LLM이 주는 것은 방향(long/short)과 확신도(conviction)뿐이다.
  2. 포지션 크기는 "얼마를 살까"가 아니라
     "틀렸을 때 얼마를 잃을까"에서 역산한다.
  3. 모든 한도는 하드 리밋이다. 어떤 확신도도 이를 뚫지 못한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"          # 논리가 틀렸다
    TAKE_PROFIT = "take_profit"      # 목표 도달
    TIME_STOP = "time_stop"          # 기간 내 움직임 없음 → 자본 회수
    TRAILING_STOP = "trailing_stop"  # 이익 반납 방지
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL = "manual"


@dataclass(frozen=True)
class RiskConfig:
    """
    0단계에서 문장으로 쓴 규칙이 여기 숫자로 들어온다.
    이 값들을 바꾸는 것이 곧 전략을 바꾸는 것이므로 버전 관리 대상이다.
    """

    # --- 1회 거래 위험 ---
    risk_per_trade: float = 0.005       # 계좌의 0.5%. 1회 손절 시 최대 손실
    atr_period: int = 14
    atr_stop_multiple: float = 2.0      # 손절 = 진입가 -/+ 2 x ATR
    reward_risk_ratio: float = 2.5      # 익절폭 = 손절폭 x 2.5 (손익비)

    # --- 시간 손절 ---
    time_stop_bars: int = 10            # 10거래일 내 판가름 안 나면 청산
    time_stop_min_progress: float = 0.5 # 손절폭의 50%만큼도 못 갔으면 "진전 없음"

    # --- 추적 손절 ---
    use_trailing: bool = True
    trailing_activate_r: float = 1.0    # +1R 도달 후 추적 시작
    trailing_atr_multiple: float = 2.0

    # --- 포트폴리오 한도 (하드 리밋) ---
    max_position_weight: float = 0.20   # 한 종목 최대 20%
    max_open_positions: int = 8
    max_gross_exposure: float = 1.0     # 레버리지 금지

    # --- 서킷 브레이커 ---
    max_drawdown_halt: float = 0.10     # 계좌 낙폭 10% → 신규 진입 전면 중단
    halt_cooldown_bars: int = 20        # 해제까지 최소 20거래일

    # --- 소액 계좌 대응 ---
    allow_fractional: bool = False      # 소수점 주식 허용 여부
    min_notional: float = 0.0           # 이 금액 미만 포지션은 진입하지 않음(더스트 방지)

    # --- 체결 비용 ---
    commission_rate: float = 0.0
    slippage_rate: float = 0.0005       # 0.05%. 낙관적으로 잡지 말 것

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.02:
            raise ValueError("risk_per_trade는 0~2% 범위를 권장합니다")
        if self.atr_stop_multiple <= 0:
            raise ValueError("atr_stop_multiple은 양수여야 합니다")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight는 0~1 사이여야 합니다")


@dataclass(frozen=True)
class PlannedTrade:
    """진입 전에 확정되는 거래 계획. 진입 후 손절가를 바꾸지 않는다(추적 손절 제외)."""

    symbol: str
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    risk_amount: float      # 손절 시 예상 손실(절대금액)
    stop_distance: float    # 1R
    notional: float
    weight: float
    rejected_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.rejected_reason is None and self.quantity > 0


def compute_stop_price(
    entry_price: float,
    atr_value: float,
    side: Side,
    cfg: RiskConfig,
) -> float:
    """
    변동성 기반 손절가.

    고정 % 손절(-7% 등)을 쓰지 않는 이유:
    변동성이 큰 종목은 정상 노이즈만으로도 -7%를 찍기 때문에
    논리가 틀리지 않았는데도 털리게 된다(whipsaw).
    """
    if atr_value <= 0:
        raise ValueError("ATR이 0 이하입니다. 데이터 기간이 부족한지 확인하세요")

    distance = cfg.atr_stop_multiple * atr_value
    if side is Side.LONG:
        stop = entry_price - distance
        if stop <= 0:
            raise ValueError("손절가가 0 이하입니다. ATR 배수를 줄이세요")
        return stop
    return entry_price + distance


def compute_target_price(
    entry_price: float,
    stop_price: float,
    side: Side,
    cfg: RiskConfig,
) -> float:
    """익절가 = 손절폭 x 손익비. 승률이 낮아도 손익비로 기댓값을 만든다."""
    stop_distance = abs(entry_price - stop_price)
    reward = stop_distance * cfg.reward_risk_ratio
    return entry_price + reward if side is Side.LONG else entry_price - reward


def size_position(
    equity: float,
    entry_price: float,
    stop_price: float,
    cfg: RiskConfig,
    *,
    available_cash: float | None = None,
    conviction: float = 1.0,
    allow_fractional: bool = False,
) -> PlannedTrade | None:
    """
    핵심 공식:
        위험금액 = 계좌 x risk_per_trade
        수량     = 위험금액 / 손절폭

    손절폭이 넓으면(= 변동성이 크면) 수량이 자동으로 줄어든다.
    이 한 줄이 계좌를 지킨다.

    conviction(0~1)은 위험금액을 '줄이는' 방향으로만 작동한다.
    확신이 높다고 한도를 넘겨 키우지 않는다 — 확신은 자주 틀리기 때문이다.
    """
    if equity <= 0:
        return None
    if not 0.0 <= conviction <= 1.0:
        raise ValueError("conviction은 0~1 사이여야 합니다")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return None

    risk_amount = equity * cfg.risk_per_trade * conviction
    raw_qty = risk_amount / stop_distance
    if not allow_fractional:
        raw_qty = float(int(raw_qty))
    if raw_qty <= 0:
        return None

    notional = raw_qty * entry_price
    cap_notional = equity * cfg.max_position_weight

    # 한도 1: 종목 비중 상한
    if notional > cap_notional:
        raw_qty = cap_notional / entry_price
        if not allow_fractional:
            raw_qty = float(int(raw_qty))
        notional = raw_qty * entry_price

    # 한도 2: 가용 현금
    if available_cash is not None and notional > available_cash:
        raw_qty = available_cash / entry_price
        if not allow_fractional:
            raw_qty = float(int(raw_qty))
        notional = raw_qty * entry_price

    if raw_qty <= 0:
        return None

    # 더스트 포지션 방지: 너무 작으면 수수료·슬리피지가 손익을 지배한다
    if cfg.min_notional > 0 and notional < cfg.min_notional:
        return None

    return PlannedTrade(
        symbol="",
        side=Side.LONG,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=entry_price,
        quantity=raw_qty,
        risk_amount=raw_qty * stop_distance,
        stop_distance=stop_distance,
        notional=notional,
        weight=notional / equity if equity else 0.0,
    )


def plan_trade(
    symbol: str,
    side: Side,
    equity: float,
    entry_price: float,
    atr_value: float,
    cfg: RiskConfig,
    *,
    available_cash: float | None = None,
    conviction: float = 1.0,
    allow_fractional: bool | None = None,
) -> PlannedTrade:
    """진입 계획 전체를 한 번에 산출. 거부 사유가 있으면 rejected_reason에 담는다."""
    if allow_fractional is None:
        allow_fractional = cfg.allow_fractional

    stop_price = compute_stop_price(entry_price, atr_value, side, cfg)
    target_price = compute_target_price(entry_price, stop_price, side, cfg)

    sized = size_position(
        equity=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        cfg=cfg,
        available_cash=available_cash,
        conviction=conviction,
        allow_fractional=allow_fractional,
    )

    if sized is None:
        return PlannedTrade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=0.0,
            risk_amount=0.0,
            stop_distance=abs(entry_price - stop_price),
            notional=0.0,
            weight=0.0,
            rejected_reason="사이징 결과 수량 0 (자본 부족 또는 손절폭 과대)",
        )

    return PlannedTrade(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=sized.quantity,
        risk_amount=sized.risk_amount,
        stop_distance=sized.stop_distance,
        notional=sized.notional,
        weight=sized.weight,
    )


def update_trailing_stop(
    side: Side,
    current_stop: float,
    entry_price: float,
    stop_distance: float,
    favorable_extreme: float,
    atr_value: float,
    cfg: RiskConfig,
) -> float:
    """
    추적 손절. 손절가는 유리한 방향으로만 이동한다(절대 되돌리지 않는다).

    favorable_extreme: 진입 후 도달한 최고가(롱) / 최저가(숏)
    """
    if not cfg.use_trailing or stop_distance <= 0:
        return current_stop

    if side is Side.LONG:
        r_multiple = (favorable_extreme - entry_price) / stop_distance
        if r_multiple < cfg.trailing_activate_r:
            return current_stop
        candidate = favorable_extreme - cfg.trailing_atr_multiple * atr_value
        return max(current_stop, candidate)

    r_multiple = (entry_price - favorable_extreme) / stop_distance
    if r_multiple < cfg.trailing_activate_r:
        return current_stop
    candidate = favorable_extreme + cfg.trailing_atr_multiple * atr_value
    return min(current_stop, candidate)


def should_time_stop(
    side: Side,
    entry_price: float,
    current_price: float,
    stop_distance: float,
    bars_held: int,
    cfg: RiskConfig,
) -> bool:
    """
    시간 손절: N봉이 지나도 손절폭의 일정 비율만큼도 진전이 없으면 청산.

    죽은 포지션이 자본을 묶는 것을 막는다.
    대부분의 개인 전략이 빠뜨리지만 성과 기여가 큰 규칙이다.
    """
    if bars_held < cfg.time_stop_bars or stop_distance <= 0:
        return False

    progress = (current_price - entry_price) if side is Side.LONG else (entry_price - current_price)
    return (progress / stop_distance) < cfg.time_stop_min_progress
