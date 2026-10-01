"""
확정 파라미터 프로파일.

이 파일의 값을 바꾸면 이전 백테스트 결과는 전부 무효다.
변경 시 반드시 커밋 메시지에 이유를 남기고, 백테스트를 처음부터 다시 돌린다.

--------------------------------------------------------------------------
결정 근거 (2026-09, 계좌 100만원 기준)
--------------------------------------------------------------------------
risk_per_trade = 0.005
    1회 손절 시 5,000원 손실. 0.5% 규칙이 자동으로 종목당 비중 약 12.5%를
    만들고, 8종목이면 100%가 되어 max_gross_exposure와 정확히 맞물린다.
    1%로 올리면 종목당 25%가 되어 max_position_weight(20%)에 계속 걸리므로
    실효 위험은 0.8%로 잘린다. 즉 지금 구조에서 0.5%가 자연스러운 값이다.

allow_fractional = True  ← 100만원 계좌에서는 선택이 아니라 필수
    종목당 배분액이 약 $87이라 GOOGL($255), NVDA($180) 모두 1주도 못 산다.
    정수 주식만 가능한 환경에서는 이 시스템이 아예 돌지 않는다.
    소수점 주식을 지원하는 브로커가 전제 조건이다.

min_notional = 20,000원 (약 $14)
    이보다 작은 포지션은 슬리피지가 손익을 지배해 통계가 오염된다.

time_stop_bars = 10  ← 잠정값. 2단계에서 실측으로 교체할 것
    보유 기간을 아직 정하지 않았으므로 단타 기준 기본값으로 둔다.
    2단계에서 MAE/MFE 분포(진입 후 최대 역행폭 / 최대 순행폭이
    몇 봉째에 나오는지)를 측정해 확정한다. 추측으로 확정하지 않는다.

max_drawdown_halt = 0.10
    8종목 동시 손절 = 4% 손실이므로 10%는 "전 종목 몰살 2.5회"를 견디는 값.
    본인이 손으로 끌 지점과 일치시켰다. 지킬 수 없는 규칙은 규칙이 아니다.

max_open_positions = 8
    8 x 12.5% = 100%. 레버리지 없이 완전 투자되는 지점.

slippage_rate = 0.001  ← 기본값(0.0005)의 2배로 상향
    100만원 계좌는 주문 금액이 작아 체결 품질이 나쁘고, 소수점 주문은
    브로커 내부 처리라 스프레드가 더 불리할 수 있다. 낙관적으로 잡지 않는다.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from risk.rules import RiskConfig

KRW_PER_USD = 1430.0  # 백테스트 재현성을 위해 고정. 실운용 시 실시간 환율 사용

# 실제로 쓸 설정 --------------------------------------------------------------
LIVE_SMALL = RiskConfig(
    risk_per_trade=0.005,
    atr_period=14,
    atr_stop_multiple=2.0,
    reward_risk_ratio=2.5,
    time_stop_bars=10,           # 잠정 — 2단계에서 실측 교체
    time_stop_min_progress=0.5,
    use_trailing=True,
    trailing_activate_r=1.0,
    trailing_atr_multiple=2.0,
    max_position_weight=0.20,
    max_open_positions=8,
    max_gross_exposure=1.0,
    max_drawdown_halt=0.10,
    halt_cooldown_bars=20,
    allow_fractional=True,       # 필수
    min_notional=20_000 / KRW_PER_USD,
    commission_rate=0.0,
    slippage_rate=0.001,
)

# 시간 손절 후보값 — 2단계 MAE/MFE 실측으로 확정한다
TIME_STOP_CANDIDATES = (5, 10, 15, 20, 30)

# 런타임 LLM (3단계에서 사용)
#   구독(Claude Pro / ChatGPT Plus / Gemini Advanced)은 API 접근을 포함하지 않는다.
#   자동매매는 프로그래밍 방식 호출이므로 구독으로는 돌릴 수 없다.
#   구독은 개발·분석용으로만 쓴다.
RUNTIME_LLM_PRIMARY = "gemini-flash-lite"   # AI Studio 무료 티어 (~1,500 req/day)
RUNTIME_LLM_FALLBACK = "claude-haiku"       # 유료. 월 2천원대
#   주의: Gemini 무료 티어 입출력은 모델 개선에 사용될 수 있고,
#         Google이 예고 없이 한도를 조정한 전례가 있다.

# --------------------------------------------------------------------------
# 계좌 증액 조건 — 네 개를 모두 충족하기 전에는 늘리지 않는다.
#
# 자본을 늘려도 기대수익률은 오르지 않는다. 절대금액만 커진다.
# 전략이 음의 기대값이면 증액은 손실 속도를 높일 뿐이다.
# 성과가 좋아 보일 때 충동적으로 늘리는 것을 막기 위해 코드에 남긴다.
# --------------------------------------------------------------------------
CAPITAL_INCREASE_CRITERIA = (
    "페이퍼 트레이딩 3개월 이상 + 거래 표본 50건 이상",
    "무작위 진입 기준선(-0.076R)을 명확히 상회",
    "실계좌 1개월 운영 중 체결 오류 0건 (미체결·손절 미발동·수량 오류)",
    "실제 낙폭이 시뮬레이션 예측 범위 이내",
)
CAPITAL_LADDER_KRW = (1_000_000, 3_000_000, 5_000_000, 10_000_000)
CURRENT_CAPITAL_KRW = 1_000_000


def account_krw_to_usd(krw: float) -> float:
    return krw / KRW_PER_USD
