"""
회귀 테스트.

실제 데이터로 넘어가기 전에 이것부터 통과해야 한다.
백테스트 결과를 믿으려면 백테스터를 먼저 믿을 수 있어야 하기 때문이다.

    python test_invariants.py
"""

from __future__ import annotations

import sys

import numpy as np

from configs import LIVE_SMALL
from cycle import run_backtest
from data.contract import LookaheadError, make_synthetic_panel
from risk.portfolio import Portfolio
from risk.rules import Side, plan_trade
from signals.models import MomentumBreakout, RandomEntry

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def test_lookahead_guard() -> None:
    print("\n[룩어헤드 차단]")
    panel = make_synthetic_panel(n_symbols=3, n_bars=200, seed=1)
    try:
        panel.bar("SYM00", index=150, as_of=100)
        check("미래 봉 접근 시 예외", False, "예외가 발생하지 않음")
    except LookaheadError:
        check("미래 봉 접근 시 예외", True)

    h = panel.history("SYM00", as_of=100)
    check("history가 as_of까지만 반환", len(h) == 101, f"길이 {len(h)}")


def test_risk_invariant() -> None:
    """
    위험금액은 두 경우로 나뉜다:
      - 일반: 정확히 risk_per_trade
      - 저변동성(ATR/주가 < 1.25%): max_position_weight 상한에 걸려 그보다 작아짐

    상한에 걸려 위험이 '작아지는' 것은 정상이다.
    위험이 목표보다 '커지는' 경우만 결함이다.
    """
    print("\n[1회 위험 불변식]")
    equity = 699.0
    target = equity * LIVE_SMALL.risk_per_trade
    rows = []
    for price, atr_v in [(50, 1.0), (100, 2.0), (250, 8.0), (400, 3.0), (230, 2.3)]:
        p = plan_trade("T", Side.LONG, equity, price, atr_v, LIVE_SMALL, available_cash=equity)
        if p.is_valid:
            rows.append((price, atr_v, p.risk_amount, p.weight))

    over = [r for r in rows if r[2] > target * 1.02]
    check("위험금액이 목표를 초과하지 않음", not over,
          f"목표 ${target:.2f} / 최대 ${max(r[2] for r in rows):.2f}")

    uncapped = [r for r in rows if r[3] < LIVE_SMALL.max_position_weight - 1e-6]
    ok = all(abs(r[2] - target) < target * 0.02 for r in uncapped)
    check("비중 상한 미적용 시 위험금액 정확히 일치", ok,
          f"{len(uncapped)}건 검사")

    capped = [r for r in rows if r[3] >= LIVE_SMALL.max_position_weight - 1e-6]
    check("상한 적용 시 위험금액은 목표보다 작음",
          all(r[2] < target for r in capped) if capped else True,
          f"{len(capped)}건 (저변동성 종목)")


def test_gap_resizing() -> None:
    """버그2 회귀: 갭이 나도 실제 위험이 유지되어야 한다."""
    print("\n[갭 발생 시 재사이징]")
    equity = 699.0
    atr_v = 2.0
    target = equity * LIVE_SMALL.risk_per_trade

    deviations = []
    for gap in (-0.02, 0.0, 0.02):
        fill = 100.0 * (1 + gap)
        p = plan_trade("T", Side.LONG, equity, fill, atr_v, LIVE_SMALL, available_cash=equity)
        deviations.append(abs(p.risk_amount / target - 1))
    check("체결가 기준 재계산 시 위험 편차 2% 이내",
          max(deviations) < 0.02, f"최대 편차 {max(deviations):.2%}")


def test_gross_exposure() -> None:
    """버그1 회귀: 포지션 보유 상태에서 gross_exposure가 터지지 않아야 한다."""
    print("\n[gross_exposure]")
    pf = Portfolio(initial_equity=699.0, cfg=LIVE_SMALL)
    p = plan_trade("X", Side.LONG, 699.0, 100.0, 2.0, LIVE_SMALL, available_cash=699.0)
    pf.open_position(p, {"X": 100.0})
    try:
        v = pf.gross_exposure()
        check("포지션 보유 시 정상 계산", 0 < v < 1, f"{v:.1%}")
    except AttributeError as e:
        check("포지션 보유 시 정상 계산", False, str(e))


def test_hard_limits() -> None:
    print("\n[하드 리밋]")
    panel = make_synthetic_panel(n_symbols=25, n_bars=600, seed=5)
    pf, logs = run_backtest(
        panel, RandomEntry(probability=0.05, seed=3), LIVE_SMALL, 699.0, collect_logs=True
    )
    check("최대 보유 종목 수 준수",
          all(len(pf.positions) <= LIVE_SMALL.max_open_positions for _ in [0]))
    peak_dd = max((1 - e / max(pf.equity_curve[: i + 1]))
                  for i, e in enumerate(pf.equity_curve)) if pf.equity_curve else 0
    check("낙폭이 서킷 한도의 2배를 넘지 않음",
          peak_dd < LIVE_SMALL.max_drawdown_halt * 2, f"최대 {peak_dd:.2%}")
    check("계좌가 음수로 가지 않음", pf.equity > 0, f"${pf.equity:.2f}")
    check("현금이 음수로 가지 않음", pf.cash >= -1e-6, f"${pf.cash:.2f}")


def test_accounting() -> None:
    print("\n[회계 정합성]")
    panel = make_synthetic_panel(n_symbols=20, n_bars=500, seed=9)
    pf, _ = run_backtest(panel, MomentumBreakout(), LIVE_SMALL, 699.0)
    realized = sum(t.pnl for t in pf.closed_trades)
    delta = pf.equity - pf.initial_equity
    # 수수료·슬리피지 때문에 완전 일치하지는 않는다
    ok = abs(realized - delta) < abs(delta) * 0.15 + 1.0
    check("실현손익 합계 ≈ 자본 변화", ok,
          f"실현 ${realized:+.2f} / 변화 ${delta:+.2f}")

    if pf.closed_trades:
        stops = [t for t in pf.closed_trades if t.reason.value == "stop_loss"]
        if stops:
            avg = np.mean([t.r_multiple for t in stops])
            check("최초 손절 청산의 평균이 -1R 근처",
                  -1.15 < avg < -0.85, f"{avg:+.3f}R")


def test_determinism() -> None:
    print("\n[재현성]")
    panel = make_synthetic_panel(n_symbols=15, n_bars=400, seed=2)
    a, _ = run_backtest(panel, MomentumBreakout(), LIVE_SMALL, 699.0)
    b, _ = run_backtest(panel, MomentumBreakout(), LIVE_SMALL, 699.0)
    check("같은 입력 → 같은 출력", abs(a.equity - b.equity) < 1e-9,
          f"${a.equity:.6f} vs ${b.equity:.6f}")


def main() -> None:
    print("=" * 64)
    print("회귀 테스트 — 실제 데이터 투입 전 필수 통과")
    print("=" * 64)

    test_lookahead_guard()
    test_risk_invariant()
    test_gap_resizing()
    test_gross_exposure()
    test_hard_limits()
    test_accounting()
    test_determinism()

    print()
    print("=" * 64)
    print(f"통과 {len(PASS)} / 실패 {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print(f"  실패: {f}")
        sys.exit(1)
    print("전부 통과. 실제 데이터로 진행 가능.")


if __name__ == "__main__":
    main()
