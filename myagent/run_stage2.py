"""
2단계 실행: 시간 손절 확정 + 결정론적 신호 walk-forward 검증

실행 순서:
  A. MAE/MFE 실측 → time_stop_bars 확정
  B. 확정값으로 walk-forward → 무작위 기준선 대비 평가
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from configs import LIVE_SMALL, TIME_STOP_CANDIDATES, account_krw_to_usd
from cycle import buy_and_hold_benchmark, run_backtest
from data.contract import make_synthetic_panel
from eval.harness import make_walk_forward_splits, measure_excursions, summarize_excursions
from signals.models import MeanReversion, MomentumBreakout, RandomEntry

EQUITY = account_krw_to_usd(1_000_000)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def stage_a_time_stop(panel) -> dict[str, int]:
    section("A. MAE/MFE 실측 — time_stop_bars 확정")
    print("진입 후 궤적을 R 단위로 추적한다. 손절·익절 미적용(궤적을 끝까지 보기 위함).")

    chosen: dict[str, int] = {}
    for model in (MomentumBreakout(), MeanReversion()):
        exs = measure_excursions(panel, model, LIVE_SMALL, horizon=40)
        s = summarize_excursions(exs, TIME_STOP_CANDIDATES)
        if not s:
            print(f"\n[{model.name}] 표본 없음")
            continue

        print(f"\n[{model.name}] 표본 {s['n']}건")
        print(f"  MFE 중앙값 {s['mfe_median']:+.2f}R · 상위25% {s['mfe_p75']:+.2f}R")
        print(f"  MAE 중앙값 {s['mae_median']:+.2f}R · 하위25% {s['mae_p25']:+.2f}R")
        print(f"  +1R 도달 비율 {s['pct_reach_1r']:.1%} · 도달까지 중앙 {s['bars_to_1r_median']:.0f}봉")
        print(f"  MFE 도달 중앙 {s['bars_to_mfe_median']:.0f}봉")
        print("  후보값별:")
        for c, v in s["by_candidate"].items():
            print(f"    {c:>3}봉 → 그 시점 평균 {v['avg_r_at_cut']:+.3f}R · "
                  f"+1R 조기절단 {v['pct_1r_cut_early']:.1%}")

        # 선택 규칙: +1R 조기 절단이 20% 미만이 되는 가장 짧은 값
        pick = None
        for c in sorted(s["by_candidate"]):
            if s["by_candidate"][c]["pct_1r_cut_early"] < 0.20:
                pick = c
                break
        pick = pick or max(s["by_candidate"])
        chosen[model.name] = pick
        print(f"  → 선택: {pick}봉 (기준: +1R 조기절단 20% 미만인 최단값)")

    return chosen


def stage_b_walkforward(panel, time_stops: dict[str, int]) -> None:
    section("B. Walk-forward — 무작위 기준선 대비 평가")
    splits = make_walk_forward_splits(panel.n_bars, train_bars=600, test_bars=300)
    print(f"구간 {len(splits)}개 (train 600봉 / test 300봉). test 구간 성과만 인정한다.")

    models = {
        "random": (RandomEntry(probability=0.015, seed=1), LIVE_SMALL.time_stop_bars),
        "momentum": (MomentumBreakout(), time_stops.get("momentum", 10)),
        "meanrev": (MeanReversion(), time_stops.get("meanrev", 10)),
    }

    table: dict[str, list[dict]] = {k: [] for k in models}
    bh: list[float] = []

    for sp in splits:
        bh.append(buy_and_hold_benchmark(panel, sp.test_start, sp.test_end))
        for name, (model, ts) in models.items():
            cfg = replace(LIVE_SMALL, time_stop_bars=ts)
            pf, _ = run_backtest(
                panel, model, cfg, EQUITY,
                start_bar=sp.test_start, end_bar=sp.test_end,
            )
            table[name].append(pf.stats())

    print()
    hdr = f"{'모델':<10}{'거래':>7}{'승률':>9}{'평균R':>9}{'수익':>10}{'MDD':>9}{'Sharpe':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, rows in table.items():
        n = sum(r["trades"] for r in rows)
        wr = np.mean([r["win_rate"] for r in rows if r["trades"]]) if n else 0
        ar = np.mean([r["avg_r"] for r in rows if r["trades"]]) if n else 0
        ret = np.mean([r["total_return"] for r in rows])
        dd = np.mean([r["max_drawdown"] for r in rows])
        sh = np.mean([r["sharpe"] for r in rows])
        print(f"{name:<10}{n:>7}{wr:>8.1%}{ar:>+9.3f}{ret:>+10.2%}{dd:>9.2%}{sh:>9.2f}")
    print("-" * len(hdr))
    print(f"{'매수보유':<10}{'':>7}{'':>9}{'':>9}{np.mean(bh):>+10.2%}")

    section("판정")
    base = np.mean([r["avg_r"] for r in table["random"] if r["trades"]])
    print(f"무작위 기준선: 평균 {base:+.3f}R")
    for name in ("momentum", "meanrev"):
        rows = [r for r in table[name] if r["trades"]]
        if not rows:
            print(f"  {name}: 거래 없음 — 신호 조건이 너무 빡빡함")
            continue
        ar = np.mean([r["avg_r"] for r in rows])
        verdict = "통과" if ar > base + 0.05 else "미달"
        print(f"  {name}: 평균 {ar:+.3f}R (기준선 대비 {ar - base:+.3f}R) → {verdict}")


def main() -> None:
    panel = make_synthetic_panel(n_symbols=40, n_bars=1500, seed=11, drift=True)
    print(f"패널: 종목 {len(panel.symbols)} · 봉 {panel.n_bars} · 계좌 ${EQUITY:,.0f}")
    print("주의: 합성 데이터다. 여기 나온 수익률은 전략의 증거가 아니라 배관 검증용이다.")

    time_stops = stage_a_time_stop(panel)
    stage_b_walkforward(panel, time_stops)


if __name__ == "__main__":
    main()
