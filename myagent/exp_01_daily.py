"""
실험 1 — 크립토 meanrev의 비용 민감도

    python exp_01_daily.py --fetch     # 일봉 데이터 (2~5분)
    python exp_01_daily.py --run

배경
-----
4시간봉 결과에서 meanrev의 초과R은 4개 구간 전부 통과했다.
    5봉 +0.154R (t=4.09) / 10봉 +0.203R (t=3.65)
    20봉 +0.268R (t=3.40) / 30봉 +0.279R (t=2.80)

그런데 walk-forward에서는 -0.182R로 무작위(-0.111R)보다 나빴다.
원인은 거래비용이다. 중앙값 0.076R, BTC는 0.140R.

질문
-----
비용을 줄이면 살아나는가?

    4h + taker  →  1d + maker
    비용 0.076R  →  0.016R 예상 (약 1/5)

동시에 두 가지가 일어난다:
    비용 감소   ← 유리
    초과R 희석  ← 불리 (평균회귀는 단기 효과라 시간축을 늘리면 약해진다)

어느 쪽이 큰지가 이 실험의 전부다.

사전 약속
----------
1. 이 조합 하나만 돌린다. "알트만 남기기", "1h로 바꾸기" 같은 변형은 없다.
2. 미달하면 즉시 C(실행 품질)로 넘어간다.
3. 통과해도 바로 실거래가 아니다. 별도 검증이 필요하다.

maker 수수료 가정에 대하여
---------------------------
maker(지정가)는 체결이 보장되지 않는다. 지정가를 걸어두고 가격이
안 오면 진입 자체가 안 된다. 이 백테스트는 "걸면 체결된다"고 가정하므로
실제보다 낙관적이다. 통과하더라도 이 가정을 별도로 검증해야 한다.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import replace
from statistics import NormalDist

import numpy as np

from configs import LIVE_SMALL, account_krw_to_usd

CACHE = "data_cache/crypto_1d.pkl"

# 일봉 + maker 설정.
#   maker 수수료: Binance 현물 maker 0.02% (VIP0 기준 실제로는 0.1%지만
#                 BNB 할인 + 지정가 조합으로 낮출 수 있다. 낙관적 가정)
#   슬리피지: 지정가라 슬리피지는 작지만 미체결 위험이 대신 생긴다
DAILY_CFG = replace(
    LIVE_SMALL,
    commission_rate=0.0002,
    slippage_rate=0.0005,
    allow_fractional=True,
    min_notional=10.0,
    time_stop_bars=10,
)

# 비교용: 일봉 + taker (수수료 효과만 분리해서 보기 위함)
DAILY_TAKER_CFG = replace(DAILY_CFG, commission_rate=0.001, slippage_rate=0.0008)


def load(force: bool = False):
    from data.crypto import DEFAULT_CRYPTO, load_crypto_panel

    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            panel = pickle.load(f)
        print(f"캐시 사용: 종목 {len(panel.symbols)} · 봉 {panel.n_bars}")
        return panel

    os.makedirs("data_cache", exist_ok=True)
    print("일봉 다운로드 중 (2~5분)...")
    # 일봉이므로 기간을 늘려 표본을 확보한다
    panel = load_crypto_panel(DEFAULT_CRYPTO, timeframe="1d", since_days=2200)
    with open(CACHE, "wb") as f:
        pickle.dump(panel, f)
    return panel


def show_cost(panel) -> float:
    from data.crypto import cost_in_r_units
    from risk.indicators import atr as atr_fn

    print("\n" + "=" * 74)
    print("1. 비용 확인 — 실제로 줄었는가")
    print("=" * 74)
    print(f"  {'자산':<10}{'ATR/가':>9}{'4h taker':>11}{'1d maker':>11}")
    costs = []
    for s in panel.symbols[:8]:
        df = panel.frames[s]
        a = atr_fn(df["high"], df["low"], df["close"], DAILY_CFG.atr_period)
        atr_pct = float((a / df["close"]).dropna().median())
        c_new = cost_in_r_units(atr_pct, DAILY_CFG.atr_stop_multiple,
                                DAILY_CFG.commission_rate, DAILY_CFG.slippage_rate)
        c_old = cost_in_r_units(atr_pct / 2.4, DAILY_CFG.atr_stop_multiple, 0.001, 0.0008)
        costs.append(c_new)
        print(f"  {s:<10}{atr_pct:>8.2%}{c_old:>11.3f}{c_new:>11.3f}")
    med = float(np.median(costs))
    print(f"\n  일봉+maker 비용 중앙값 {med:.3f}R (4시간봉 taker는 0.076R이었다)")
    return med


def show_excess(panel) -> dict:
    from eval.harness import measure_excursions
    from patch_03_cluster import cluster_bootstrap_excess
    from signals.models import MeanReversion, RandomEntry

    candidates = (5, 10, 20, 30)
    crit = NormalDist().inv_cdf(1 - (0.05 / len(candidates)) / 2)

    print("\n" + "=" * 74)
    print("2. 초과R — 일봉에서도 남아 있는가")
    print("=" * 74)
    print(f"다중검정 보정: {len(candidates)}회 → 임계값 |t| > {crit:.2f}")
    print("평균회귀는 단기 효과라 일봉에서는 희석될 수 있다.\n")

    base = measure_excursions(panel, RandomEntry(probability=0.03, seed=99),
                              DAILY_CFG, horizon=40, max_samples=8000)
    sig = measure_excursions(panel, MeanReversion(), DAILY_CFG,
                             horizon=40, max_samples=8000)
    print(f"신호 {len(sig)}건 vs 무작위 {len(base)}건")
    if len(sig) < 150:
        print("  표본 부족 — 일봉 + RSI(3)<15 조건이 너무 빡빡하다")
        return {}

    print(f"  {'봉':>5}{'초과R':>10}{'95% CI':>22}{'유효t':>9}  판정   (4h 대비)")
    ref = {5: 0.154, 10: 0.203, 20: 0.268, 30: 0.279}
    out = {}
    for c in candidates:
        r = cluster_bootstrap_excess(sig, base, c)
        if "excess" not in r:
            continue
        out[c] = r
        ok = r["t_equivalent"] > crit
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        delta = r["excess"] - ref[c]
        print(f"  {c:>5}{r['excess']:>+10.3f}{ci:>22}{r['t_equivalent']:>9.2f}"
              f"  {'통과' if ok else '미달'}   {delta:+.3f}")
    return out


def show_walkforward(panel) -> bool:
    from cycle import run_backtest
    from eval.harness import make_walk_forward_splits
    from signals.models import MeanReversion, RandomEntry

    print("\n" + "=" * 74)
    print("3. Walk-forward — 최종 판정")
    print("=" * 74)

    equity = account_krw_to_usd(1_000_000)
    splits = make_walk_forward_splits(panel.n_bars, train_bars=750, test_bars=250)
    if not splits:
        splits = make_walk_forward_splits(panel.n_bars, train_bars=500, test_bars=200)
    print(f"구간 {len(splits)}개\n")

    btc = next((s for s in panel.symbols if s.startswith("BTC")), panel.symbols[0])
    setups = {
        "random":       (RandomEntry(probability=0.015, seed=1), DAILY_CFG),
        "meanrev(maker)": (MeanReversion(), DAILY_CFG),
        "meanrev(taker)": (MeanReversion(), DAILY_TAKER_CFG),
    }
    rows = {k: [] for k in setups}
    hold = []

    for sp in splits:
        df = panel.frames[btc]
        hold.append(float(df["close"].iloc[sp.test_end]) / float(df["close"].iloc[sp.test_start]) - 1)
        for name, (model, cfg) in setups.items():
            pf, _ = run_backtest(panel, model, cfg, equity,
                                 start_bar=sp.test_start, end_bar=sp.test_end)
            rows[name].append(pf.stats())

    hdr = f"  {'설정':<16}{'거래':>7}{'승률':>9}{'평균R':>9}{'수익':>10}{'MDD':>9}{'Sharpe':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    agg = {}
    for name, rs in rows.items():
        live = [r for r in rs if r["trades"]]
        agg[name] = {
            "avg_r": float(np.mean([r["avg_r"] for r in live])) if live else 0.0,
            "ret": float(np.mean([r["total_return"] for r in rs])),
            "sharpe": float(np.mean([r["sharpe"] for r in rs])),
            "per_split": [r["avg_r"] for r in rs],
        }
        n = sum(r["trades"] for r in rs)
        wr = np.mean([r["win_rate"] for r in live]) if live else 0
        dd = np.mean([r["max_drawdown"] for r in rs])
        print(f"  {name:<16}{n:>7}{wr:>8.1%}{agg[name]['avg_r']:>+9.3f}"
              f"{agg[name]['ret']:>+10.2%}{dd:>9.2%}{agg[name]['sharpe']:>9.2f}")
    print(f"  {'BTC 보유':<16}{'':>7}{'':>9}{'':>9}{np.mean(hold):>+10.2%}")

    m, b = agg["meanrev(maker)"], agg["random"]
    c1 = m["avg_r"] > b["avg_r"] + 0.05
    wins = sum(1 for x, y in zip(m["per_split"], b["per_split"]) if x > y)
    c2 = wins > len(splits) / 2
    c3 = m["ret"] > np.mean(hold)

    print("\n  판정 (meanrev maker 기준)")
    print(f"    1) 무작위 +0.05R 초과   {m['avg_r']:+.3f} vs {b['avg_r'] + 0.05:+.3f}  {'O' if c1 else 'X'}")
    print(f"    2) 과반 구간 우세       {wins}/{len(splits)}  {'O' if c2 else 'X'}")
    print(f"    3) BTC 보유 초과        {m['ret']:+.2%} vs {np.mean(hold):+.2%}  {'O' if c3 else 'X'}")
    ok = c1 and c2 and c3
    print(f"    → {'통과' if ok else '미달'}")

    gap = m["avg_r"] - agg["meanrev(taker)"]["avg_r"]
    print(f"\n  참고: maker vs taker 차이 {gap:+.3f}R — 비용 절감이 실제로 기여한 몫")
    return ok


def main() -> None:
    args = sys.argv[1:]
    panel = load(force="--fetch" in args)
    if "--fetch" in args and len(args) == 1:
        return

    show_cost(panel)
    show_excess(panel)
    ok = show_walkforward(panel)

    print("\n" + "=" * 74)
    if ok:
        print("통과. 다만 세 가지를 먼저 확인해야 실거래를 논할 수 있다:")
        print("  1) maker 지정가 미체결 위험 — 이 백테스트는 '걸면 체결'을 가정했다")
        print("  2) 생존 편향 — 유니버스가 살아남은 20개 코인이다")
        print("  3) 국내 규제 — 거래소 선택과 세금 처리")
        print("\n다음은 2번(무엇을 얼마나 보유할까)으로 깊게 들어간다.")
    else:
        print("미달. 사전 약속대로 여기서 능동 매매 탐색을 종료한다.")
        print("비용을 1/5로 줄여도 안 된다면, 문제는 비용이 아니라 엣지의 크기다.")
        print("\n다음은 C(실행 품질)다.")


if __name__ == "__main__":
    main()
