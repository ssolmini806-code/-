"""
실험 3 — 횡단면 모멘텀(xs_mom) 강건성 검증

    python exp_03_robust_xsmom.py --all

목적은 개선이 아니라 파괴다.
통과시키려고 돌리는 게 아니라, 무너지는지 보려고 돌린다.

검증 대상
----------
exp_02에서 xs_mom만 두 시장에서 통과했다.
    크립토  +92.3% vs BTC +64.0%   (Sharpe 0.45 vs 0.36)
    미국주식 +5612% vs 균등 +2640%  (Sharpe 1.41 vs 1.15)

파라미터: lookback 90 / top 5 / skip 5 / 리밸런싱 5일

네 가지 검사
-------------
1. 무작위 5개 대조군  ← 가장 중요
   xs_mom은 20개 중 5개만 보유한다. 집중 자체가 수익을 만들 수 있다.
   같은 주기로 무작위 5개를 교체하는 것과 비교해야 신호의 순수 기여가 보인다.
   여러 시드로 분포를 만들고, xs_mom이 그 분포의 어디에 있는지 본다.

2. 기간 분할 (3등분)
   한 구간에서만 나오면 특정 국면의 산물이다.

3. ETF 유니버스
   생존 편향이 거의 없는 곳에서도 재현되는가.

4. 파라미터 민감도
   (90,5,5)만 되고 주변 값이 안 되면 그건 우연이다.
   개선할 조합을 찾는 게 아니라, 넓은 영역에서 고르게 되는지를 본다.

기각 조건 (사전 확정)
----------------------
  - 무작위 5개 분포의 상위 10% 밖에 있으면 기각
  - 3개 구간 중 2개 이상에서 대조군에 지면 기각
  - ETF에서 대조군에 지면 기각
  - 파라미터 조합의 과반에서 대조군에 지면 기각

하나라도 걸리면 기각한다. 전부 통과해야 살아남는다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

from allocation import AllocResult, buy_and_hold, equal_weight_hold, run_allocation, w_xs_momentum
from data.contract import PricePanel

BASE = {"lookback": 90, "top_n": 5, "skip": 5, "rebal": 5}


def make_xs_mom(lookback: int, top_n: int, skip: int):
    def fn(hist: pd.DataFrame) -> np.ndarray:
        return w_xs_momentum(hist, lookback=lookback, top_n=top_n, skip=skip)
    return fn


def make_random_n(top_n: int, seed: int):
    rng = np.random.default_rng(seed)

    def fn(hist: pd.DataFrame) -> np.ndarray:
        n = hist.shape[1]
        idx = rng.choice(n, size=min(top_n, n), replace=False)
        w = np.zeros(n)
        w[idx] = 1.0 / len(idx)
        return w
    return fn


def slice_panel(panel: PricePanel, lo: int, hi: int) -> PricePanel:
    return PricePanel(
        {s: df.iloc[lo:hi].reset_index(drop=True) for s, df in panel.frames.items()}
    )


# ------------------------------------------------------------------ 검사 1


def test_random_control(panel, label: str, cost: float, n_seeds: int = 40) -> bool:
    print("\n" + "=" * 80)
    print(f"검사 1 — 무작위 {BASE['top_n']}개 대조군  [{label}]")
    print("=" * 80)
    print("20개 중 5개만 보유하는 '집중' 효과와 '신호' 효과를 분리한다.")
    print(f"무작위 5개를 같은 주기로 교체하는 시나리오 {n_seeds}개와 비교.\n")

    sig = run_allocation(panel, make_xs_mom(**{k: BASE[k] for k in ("lookback", "top_n", "skip")}),
                         "xs_mom", rebalance_every=BASE["rebal"], cost_rate=cost)
    s = sig.stats()

    rets, sharpes = [], []
    for seed in range(n_seeds):
        r = run_allocation(panel, make_random_n(BASE["top_n"], seed), f"rnd{seed}",
                           rebalance_every=BASE["rebal"], cost_rate=cost)
        st = r.stats()
        rets.append(st["total_return"])
        sharpes.append(st["sharpe"])

    rets, sharpes = np.array(rets), np.array(sharpes)
    pct_ret = float((rets < s["total_return"]).mean())
    pct_sh = float((sharpes < s["sharpe"]).mean())

    print(f"  xs_mom        총수익 {s['total_return']:>+10.1%}   Sharpe {s['sharpe']:>6.2f}")
    print(f"  무작위 5개 중앙 {np.median(rets):>+10.1%}   Sharpe {np.median(sharpes):>6.2f}")
    print(f"  무작위 5개 90% {np.percentile(rets, 90):>+10.1%}   Sharpe {np.percentile(sharpes, 90):>6.2f}")
    print(f"  무작위 5개 최고 {rets.max():>+10.1%}   Sharpe {sharpes.max():>6.2f}")
    print()
    print(f"  xs_mom의 위치: 수익 상위 {(1 - pct_ret):.0%} · Sharpe 상위 {(1 - pct_sh):.0%}")

    ok = pct_ret >= 0.90
    print(f"  → {'통과' if ok else '기각'} (상위 10% 안에 들어야 통과)")
    if not ok:
        print("     집중 효과로 설명된다. 신호의 기여 증거 없음.")
    return ok


# ------------------------------------------------------------------ 검사 2


def test_periods(panel, label: str, cost: float, bench_is_equal: bool) -> bool:
    print("\n" + "=" * 80)
    print(f"검사 2 — 기간 3등분  [{label}]")
    print("=" * 80)

    n = panel.n_bars
    thirds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    wins = 0
    print(f"  {'구간':<8}{'xs_mom':>12}{'무작위5 중앙':>14}{'대조군':>12}{'판정':>8}")
    for i, (lo, hi) in enumerate(thirds, 1):
        sub = slice_panel(panel, lo, hi)
        if sub.n_bars < 250:
            print(f"  P{i:<7} 봉 부족")
            continue
        sig = run_allocation(sub, make_xs_mom(BASE["lookback"], BASE["top_n"], BASE["skip"]),
                             "x", rebalance_every=BASE["rebal"], cost_rate=cost).stats()
        rnd = np.median([
            run_allocation(sub, make_random_n(BASE["top_n"], sd), "r",
                           rebalance_every=BASE["rebal"], cost_rate=cost).stats()["total_return"]
            for sd in range(12)
        ])
        bench = (equal_weight_hold(sub, start_bar=120) if bench_is_equal
                 else buy_and_hold(sub, None, start_bar=120)).stats()["total_return"]
        ok = sig["total_return"] > rnd and sig["total_return"] > bench
        wins += ok
        print(f"  P{i:<7}{sig['total_return']:>+12.1%}{rnd:>+14.1%}{bench:>+12.1%}"
              f"{'  통과' if ok else '  미달':>8}")

    passed = wins >= 2
    print(f"\n  → {wins}/3 구간 통과. {'유지' if passed else '기각'} (2개 이상 필요)")
    return passed


# ------------------------------------------------------------------ 검사 3


def test_etf(cost: float) -> bool:
    print("\n" + "=" * 80)
    print("검사 3 — ETF 유니버스 (생존 편향 거의 없음)")
    print("=" * 80)

    from patch_04_robustness import ETF_UNIVERSE
    from run_real import load_panel

    panel = load_panel(ETF_UNIVERSE, "2015-01-01", "2026-09-01")
    sig = run_allocation(panel, make_xs_mom(BASE["lookback"], BASE["top_n"], BASE["skip"]),
                         "x", rebalance_every=BASE["rebal"], cost_rate=cost).stats()
    rnd = [run_allocation(panel, make_random_n(BASE["top_n"], sd), "r",
                          rebalance_every=BASE["rebal"], cost_rate=cost).stats()
           for sd in range(30)]
    rnd_ret = np.array([r["total_return"] for r in rnd])
    eq = equal_weight_hold(panel, start_bar=120).stats()

    pct = float((rnd_ret < sig["total_return"]).mean())
    print(f"  xs_mom       {sig['total_return']:>+10.1%}  Sharpe {sig['sharpe']:>5.2f}  MDD {sig['max_drawdown']:>6.1%}")
    print(f"  무작위5 중앙  {np.median(rnd_ret):>+10.1%}")
    print(f"  균등 방치     {eq['total_return']:>+10.1%}  Sharpe {eq['sharpe']:>5.2f}  MDD {eq['max_drawdown']:>6.1%}")
    print(f"\n  무작위 분포 내 위치: 상위 {(1 - pct):.0%}")

    ok = pct >= 0.80 and sig["total_return"] > eq["total_return"]
    print(f"  → {'통과' if ok else '기각'}")
    return ok


# ------------------------------------------------------------------ 검사 4


def test_param_sensitivity(panel, label: str, cost: float) -> bool:
    print("\n" + "=" * 80)
    print(f"검사 4 — 파라미터 민감도  [{label}]")
    print("=" * 80)
    print("최적값을 찾는 게 아니다. 넓은 영역에서 고르게 되는지를 본다.")
    print("특정 조합만 되면 그건 우연이다.\n")

    rnd_med = np.median([
        run_allocation(panel, make_random_n(BASE["top_n"], sd), "r",
                       rebalance_every=BASE["rebal"], cost_rate=cost).stats()["total_return"]
        for sd in range(12)
    ])
    print(f"  무작위 5개 중앙값: {rnd_med:+.1%}\n")
    print(f"  {'lookback':>9}{'top_n':>7}{'skip':>6}{'총수익':>12}{'Sharpe':>9}{'판정':>8}")

    combos = [(lb, tn, sk)
              for lb in (60, 90, 120)
              for tn in (3, 5, 8)
              for sk in (0, 5)]
    wins = 0
    for lb, tn, sk in combos:
        st = run_allocation(panel, make_xs_mom(lb, tn, sk), "x",
                            rebalance_every=BASE["rebal"], cost_rate=cost).stats()
        ok = st["total_return"] > rnd_med
        wins += ok
        print(f"  {lb:>9}{tn:>7}{sk:>6}{st['total_return']:>+12.1%}{st['sharpe']:>9.2f}"
              f"{'  O' if ok else '  X':>8}")

    rate = wins / len(combos)
    passed = rate > 0.5
    print(f"\n  → {wins}/{len(combos)} ({rate:.0%}) 조합에서 대조군 초과. "
          f"{'유지' if passed else '기각'} (과반 필요)")
    return passed


def load_crypto():
    path = "data_cache/crypto_1d.pkl"
    if not os.path.exists(path):
        raise SystemExit("먼저 python exp_01_daily.py --fetch 를 실행하세요.")
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    from run_real import DEFAULT_TICKERS, load_panel

    crypto = load_crypto()
    stocks = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")

    results = {}
    results["crypto_random"] = test_random_control(crypto, "크립토", 0.0007)
    results["stocks_random"] = test_random_control(stocks, "미국주식", 0.0005)
    results["crypto_period"] = test_periods(crypto, "크립토", 0.0007, bench_is_equal=False)
    results["stocks_period"] = test_periods(stocks, "미국주식", 0.0005, bench_is_equal=True)
    results["etf"] = test_etf(0.0005)
    results["crypto_param"] = test_param_sensitivity(crypto, "크립토", 0.0007)
    results["stocks_param"] = test_param_sensitivity(stocks, "미국주식", 0.0005)

    print("\n" + "=" * 80)
    print("종합 판정")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k:<18} {'통과' if v else '기각'}")

    survived = all(results.values())
    print()
    if survived:
        print("  전부 통과. xs_mom은 무너뜨리려는 시도를 견뎠다.")
        print("  다만 이것은 '증거가 있다'이지 '수익이 보장된다'가 아니다.")
        print("  다음 단계는 소액 페이퍼 트레이딩이며, 실거래는 그 후다.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  기각. 실패 항목: {', '.join(failed)}")
        print("  사전 약속대로 능동 탐색을 종료하고 C로 넘어간다.")


if __name__ == "__main__":
    main()
