"""
패치 4 — 생존편향 강건성 검사 + meanrev walk-forward

적용:
    python patch_04_robustness.py --etf        # 생존편향 없는 유니버스에서 재현되는가
    python patch_04_robustness.py --period     # 기간을 쪼개도 유지되는가
    python patch_04_robustness.py --walkforward
    python patch_04_robustness.py --all

배경
-----
클러스터 부트스트랩에서 meanrev 10봉이 초과 +0.165R (t=3.47)로 통과했다.
momentum은 전 구간 음수(5봉 t=-4.79)로 폐기한다.

그런데 아직 가장 큰 의심이 남아 있다.

  유니버스 = 2026년 현재 살아남은 대형주 30개
  신호     = 200일선 위 RSI(3)<15  (= 눌림목 매수)

즉 "결국 회복해서 대형주가 된 종목들의 눌림목"만 테스트한 것이다.
회복 못 한 종목은 애초에 표본에 없다. 눌림목 매수는 생존 편향에
가장 취약한 전략 유형이므로, +0.165R 중 얼마가 진짜인지 알 수 없다.

세 가지 강건성 검사
--------------------
1. ETF 유니버스 (--etf)
   섹터/자산 ETF는 개별 종목처럼 사라지지 않는다. 생존 편향이 거의 없다.
   여기서도 재현되면 신호가 진짜일 가능성이 크게 올라간다.
   재현되지 않으면 개별주 특유의 효과이거나 편향이다.

2. 기간 분할 (--period)
   전반기/후반기로 나눠 양쪽 다 유지되는지 본다.
   한쪽에서만 나오면 특정 국면(예: 2020 폭락 후 반등)의 산물이다.

3. Walk-forward (--walkforward)
   실제 거래 규칙(손절·익절·시간손절·비용)을 적용한 out-of-sample 성과.
   여기까지 통과해야 3단계로 간다.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

# 생존 편향이 거의 없는 유니버스.
# ETF는 개별 종목처럼 상장폐지·인수합병으로 사라지는 일이 드물다.
# (완전히 없지는 않다 — 청산된 ETF도 있다 — 하지만 개별주보다 훨씬 낫다)
ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",                      # 지수
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",        # 섹터
    "XLP", "XLU", "XLB", "XLRE",
    "EFA", "EEM", "EWJ", "EWG", "EWU",               # 해외
    "TLT", "IEF", "LQD", "HYG",                      # 채권
    "GLD", "SLV", "USO", "DBC",                      # 원자재
    "VNQ", "KRE", "SMH",                             # 섹터 보조
]


def crit_value(n_tests: int) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - (0.05 / n_tests) / 2)


def _excess_table(panel, model, cfg, candidates, label: str, n_tests: int) -> dict:
    from eval.harness import measure_excursions
    from patch_03_cluster import cluster_bootstrap_excess
    from signals.models import RandomEntry

    crit = crit_value(n_tests)
    base = measure_excursions(
        panel, RandomEntry(probability=0.04, seed=99), cfg, horizon=80, max_samples=8000
    )
    sig = measure_excursions(panel, model, cfg, horizon=80, max_samples=8000)
    if not sig or not base:
        print(f"  {label}: 표본 부족 (신호 {len(sig)} / 기준선 {len(base)})")
        return {}

    print(f"  {label}: 신호 {len(sig)}건 vs 무작위 {len(base)}건")
    print(f"    {'봉':>5}{'초과R':>10}{'95% CI':>22}{'유효t':>9}  판정")
    out = {}
    for c in candidates:
        r = cluster_bootstrap_excess(sig, base, c)
        if "excess" not in r:
            continue
        ok = r["t_equivalent"] > crit
        out[c] = r
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"    {c:>5}{r['excess']:>+10.3f}{ci:>22}{r['t_equivalent']:>9.2f}"
              f"  {'통과' if ok else '미달'}")
    return out


def cmd_etf() -> None:
    from configs import LIVE_SMALL
    from run_real import load_panel
    from signals.models import MeanReversion

    print("\n" + "=" * 78)
    print("검사 1 — ETF 유니버스 (생존 편향 거의 없음)")
    print("=" * 78)
    print("개별주에서 나온 +0.165R이 편향이 아니라면 여기서도 재현되어야 한다.\n")

    panel = load_panel(ETF_UNIVERSE, "2015-01-01", "2026-09-01")
    _excess_table(panel, MeanReversion(), LIVE_SMALL, (5, 10, 20, 30), "ETF", n_tests=4)

    print("\n  해석:")
    print("    통과 → 신호가 진짜일 가능성이 크게 올라간다")
    print("    미달 → 개별주 특유 효과이거나 생존 편향. 개별주 결과를 크게 할인해야 한다")


def cmd_period() -> None:
    from configs import LIVE_SMALL
    from data.contract import PricePanel
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion

    print("\n" + "=" * 78)
    print("검사 2 — 기간 분할")
    print("=" * 78)
    print("한쪽 기간에서만 나오면 특정 국면의 산물이다.\n")

    full = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")
    mid = full.n_bars // 2

    for label, lo, hi in [("전반기", 0, mid), ("후반기", mid, full.n_bars)]:
        sub = PricePanel({s: df.iloc[lo:hi].reset_index(drop=True) for s, df in full.frames.items()})
        _excess_table(sub, MeanReversion(), LIVE_SMALL, (10, 20), label, n_tests=4)
        print()


def cmd_walkforward() -> None:
    from configs import LIVE_SMALL, account_krw_to_usd
    from cycle import buy_and_hold_benchmark, run_backtest
    from eval.harness import make_walk_forward_splits
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, RandomEntry

    print("\n" + "=" * 78)
    print("검사 3 — Walk-forward (실제 거래 규칙 + 비용 적용)")
    print("=" * 78)
    print("초과R 분석은 비용을 뺀 순수 궤적이었다. 여기서는 손절·익절·")
    print("시간손절·슬리피지를 모두 적용한 out-of-sample 성과를 본다.\n")

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")
    equity = account_krw_to_usd(1_000_000)
    splits = make_walk_forward_splits(panel.n_bars, train_bars=750, test_bars=250)

    cfg_mr = replace(LIVE_SMALL, time_stop_bars=10)   # 클러스터 부트스트랩 결과 반영
    models = {
        "random": (RandomEntry(probability=0.015, seed=1), LIVE_SMALL),
        "meanrev": (MeanReversion(), cfg_mr),
    }

    rows = {k: [] for k in models}
    bh = []
    for sp in splits:
        bh.append(buy_and_hold_benchmark(panel, sp.test_start, sp.test_end))
        for name, (model, cfg) in models.items():
            pf, _ = run_backtest(panel, model, cfg, equity,
                                 start_bar=sp.test_start, end_bar=sp.test_end)
            rows[name].append(pf.stats())

    hdr = f"  {'모델':<10}{'거래':>7}{'승률':>9}{'평균R':>9}{'수익':>10}{'MDD':>9}{'Sharpe':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    agg = {}
    for name, rs in rows.items():
        live = [r for r in rs if r["trades"]]
        agg[name] = {
            "avg_r": float(np.mean([r["avg_r"] for r in live])) if live else 0.0,
            "per_split": [r["avg_r"] for r in rs],
            "sharpe": float(np.mean([r["sharpe"] for r in rs])),
            "ret": float(np.mean([r["total_return"] for r in rs])),
        }
        n = sum(r["trades"] for r in rs)
        wr = np.mean([r["win_rate"] for r in live]) if live else 0
        dd = np.mean([r["max_drawdown"] for r in rs])
        print(f"  {name:<10}{n:>7}{wr:>8.1%}{agg[name]['avg_r']:>+9.3f}"
              f"{agg[name]['ret']:>+10.2%}{dd:>9.2%}{agg[name]['sharpe']:>9.2f}")
    print(f"  {'매수보유':<10}{'':>7}{'':>9}{'':>9}{np.mean(bh):>+10.2%}")

    print("\n  판정:")
    base = agg["random"]
    m = agg["meanrev"]
    c1 = m["avg_r"] > base["avg_r"] + 0.05
    wins = sum(1 for a, b in zip(m["per_split"], base["per_split"]) if a > b)
    c2 = wins > len(splits) / 2
    c3 = m["sharpe"] >= 0
    print(f"    1) 기준선 +0.05R 초과   {m['avg_r']:+.3f} vs {base['avg_r'] + 0.05:+.3f}  {'O' if c1 else 'X'}")
    print(f"    2) 과반 구간 우세       {wins}/{len(splits)}  {'O' if c2 else 'X'}")
    print(f"    3) Sharpe 음수 아님     {m['sharpe']:+.2f}  {'O' if c3 else 'X'}")
    print(f"    → {'통과' if (c1 and c2 and c3) else '미달'}")


def main() -> None:
    args = sys.argv[1:]
    if not args or "--all" in args:
        cmd_etf()
        cmd_period()
        cmd_walkforward()
        return
    if "--etf" in args:
        cmd_etf()
    if "--period" in args:
        cmd_period()
    if "--walkforward" in args:
        cmd_walkforward()


if __name__ == "__main__":
    main()
