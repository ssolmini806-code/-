"""
크립토 검증 파이프라인.

미국 주식과 정확히 같은 순서로 검증한다.
결론이 갈리면 그건 시장의 차이지 방법론의 차이가 아니어야 한다.

    python run_crypto.py --fetch        # 데이터 받기 (최초 1회, 5~15분)
    python run_crypto.py --cost         # 비용 진단 (먼저 볼 것)
    python run_crypto.py --excess       # 초과R + 클러스터 부트스트랩
    python run_crypto.py --walkforward  # 최종 판정
    python run_crypto.py --all

사전 약속
----------
1. 프롬프트/파라미터 조정 없음. 미국 주식에서 쓴 신호를 그대로 쓴다.
   여기서 신호를 새로 튜닝하면 크립토 데이터에 과적합하는 것이다.
2. 벤치마크는 BTC 매수보유. 이걸 못 이기면 미달이다.
3. 미달하면 매수 후 보유로 간다. 추가 시도 없음.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import replace

import numpy as np

from configs import LIVE_SMALL, account_krw_to_usd

CACHE = "data_cache/crypto_4h.pkl"

# 크립토 비용 설정.
#
# 미국 주식은 수수료 0, 슬리피지 0.05%였다. 크립토는 다르다.
#   Binance 현물 taker 0.1% (BNB 할인 시 0.075%)
#   Upbit 0.05%
#   슬리피지: 소액이라 크지 않지만 알트는 호가 얇음
#
# 낙관적으로 잡으면 백테스트가 통째로 거짓말이 된다.
CRYPTO_CFG = replace(
    LIVE_SMALL,
    commission_rate=0.001,     # 0.1% 편도
    slippage_rate=0.0008,      # 0.08% 편도
    allow_fractional=True,     # 크립토는 원래 소수점
    min_notional=10.0,         # 거래소 최소 주문금액 여유분
    time_stop_bars=10,
)


def load(force: bool = False):
    from data.crypto import DEFAULT_CRYPTO, load_crypto_panel

    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            panel = pickle.load(f)
        print(f"캐시 사용: 종목 {len(panel.symbols)} · 봉 {panel.n_bars}")
        return panel

    os.makedirs("data_cache", exist_ok=True)
    print("ccxt 다운로드 중 (5~15분 소요)...")
    panel = load_crypto_panel(DEFAULT_CRYPTO, timeframe="4h", since_days=1095)
    with open(CACHE, "wb") as f:
        pickle.dump(panel, f)
    return panel


def cmd_cost(panel) -> None:
    from data.crypto import cost_in_r_units
    from risk.indicators import atr as atr_fn

    print("\n" + "=" * 74)
    print("비용 진단 — 여기서 이미 결판날 수 있다")
    print("=" * 74)
    print("거래 1회당 비용을 R 단위로 환산한다.")
    print("미국 주식 일봉은 0.025R이었다. 이보다 훨씬 크면 엣지가 남지 않는다.\n")

    print(f"  {'자산':<10}{'ATR/가':>9}{'손절폭':>9}{'비용(R)':>10}")
    costs = []
    for s in panel.symbols[:10]:
        df = panel.frames[s]
        a = atr_fn(df["high"], df["low"], df["close"], CRYPTO_CFG.atr_period)
        atr_pct = float((a / df["close"]).dropna().median())
        c = cost_in_r_units(atr_pct, CRYPTO_CFG.atr_stop_multiple,
                            CRYPTO_CFG.commission_rate, CRYPTO_CFG.slippage_rate)
        costs.append(c)
        print(f"  {s:<10}{atr_pct:>8.2%}{CRYPTO_CFG.atr_stop_multiple * atr_pct:>8.2%}{c:>10.3f}")

    med = float(np.median(costs))
    print(f"\n  중앙값 {med:.3f}R  (미국 주식 0.025R 대비 {med / 0.025:.1f}배)")
    print()
    if med > 0.10:
        print("  ⚠ 비용이 매우 크다. 미국 주식 meanrev의 초과가 +0.165R이었으므로,")
        print("    그 수준의 엣지라도 절반 이상이 비용으로 사라진다.")
        print("    통과하려면 초과R이 최소 " f"{med * 2:.2f}R 이상이어야 실질 의미가 있다.")


def cmd_excess(panel) -> None:
    from eval.harness import measure_excursions
    from patch_03_cluster import cluster_bootstrap_excess
    from signals.models import MeanReversion, MomentumBreakout, RandomEntry
    from statistics import NormalDist

    candidates = (5, 10, 20, 30)
    n_tests = len(candidates) * 2
    crit = NormalDist().inv_cdf(1 - (0.05 / n_tests) / 2)

    print("\n" + "=" * 74)
    print("초과R + 시간 클러스터 부트스트랩")
    print("=" * 74)
    print(f"다중검정 보정: {n_tests}회 → 임계값 |t| > {crit:.2f}")
    print("주의: 4시간봉이므로 '월 클러스터'는 약 180봉이다.\n")

    base = measure_excursions(panel, RandomEntry(probability=0.03, seed=99),
                              CRYPTO_CFG, horizon=40, max_samples=8000)
    print(f"무작위 기준선 {len(base)}건\n")

    for model in (MomentumBreakout(), MeanReversion()):
        sig = measure_excursions(panel, model, CRYPTO_CFG, horizon=40, max_samples=8000)
        if len(sig) < 200:
            print(f"[{model.name}] 표본 {len(sig)}건 — 부족\n")
            continue
        print(f"[{model.name}] 신호 {len(sig)}건")
        print(f"  {'봉':>5}{'초과R':>10}{'95% CI':>22}{'유효t':>9}  판정")
        for c in candidates:
            # 4시간봉: 월 클러스터를 180봉으로 환산
            r = cluster_bootstrap_excess(sig, base, c)
            if "excess" not in r:
                continue
            ok = r["t_equivalent"] > crit
            ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            print(f"  {c:>5}{r['excess']:>+10.3f}{ci:>22}{r['t_equivalent']:>9.2f}"
                  f"  {'통과' if ok else '미달'}")
        print()


def cmd_walkforward(panel) -> None:
    from cycle import run_backtest
    from eval.harness import make_walk_forward_splits
    from signals.models import MeanReversion, MomentumBreakout, RandomEntry

    print("\n" + "=" * 74)
    print("Walk-forward — 비용 전면 반영")
    print("=" * 74)

    equity = account_krw_to_usd(1_000_000)
    # 4시간봉: 하루 6봉. train 1500봉(약 250일) / test 500봉(약 83일)
    splits = make_walk_forward_splits(panel.n_bars, train_bars=1500, test_bars=500)
    if not splits:
        splits = make_walk_forward_splits(panel.n_bars, train_bars=900, test_bars=300)
    print(f"구간 {len(splits)}개 (train {splits[0].train_end - splits[0].train_start + 1}봉 / "
          f"test {splits[0].test_end - splits[0].test_start + 1}봉)\n")

    models = {
        "random": RandomEntry(probability=0.01, seed=1),
        "momentum": MomentumBreakout(),
        "meanrev": MeanReversion(),
    }
    rows = {k: [] for k in models}
    btc_hold = []

    btc = next((s for s in panel.symbols if s.startswith("BTC")), panel.symbols[0])
    for sp in splits:
        df = panel.frames[btc]
        p0 = float(df["close"].iloc[sp.test_start])
        p1 = float(df["close"].iloc[sp.test_end])
        btc_hold.append(p1 / p0 - 1.0)
        for name, model in models.items():
            pf, _ = run_backtest(panel, model, CRYPTO_CFG, equity,
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
            "ret": float(np.mean([r["total_return"] for r in rs])),
            "sharpe": float(np.mean([r["sharpe"] for r in rs])),
            "per_split": [r["avg_r"] for r in rs],
        }
        n = sum(r["trades"] for r in rs)
        wr = np.mean([r["win_rate"] for r in live]) if live else 0
        dd = np.mean([r["max_drawdown"] for r in rs])
        print(f"  {name:<10}{n:>7}{wr:>8.1%}{agg[name]['avg_r']:>+9.3f}"
              f"{agg[name]['ret']:>+10.2%}{dd:>9.2%}{agg[name]['sharpe']:>9.2f}")
    print(f"  {'BTC 보유':<10}{'':>7}{'':>9}{'':>9}{np.mean(btc_hold):>+10.2%}")

    print("\n" + "=" * 74)
    print("최종 판정")
    print("=" * 74)
    base = agg["random"]
    passed = []
    for name in ("momentum", "meanrev"):
        a = agg[name]
        c1 = a["avg_r"] > base["avg_r"] + 0.05
        wins = sum(1 for x, y in zip(a["per_split"], base["per_split"]) if x > y)
        c2 = wins > len(splits) / 2
        c3 = a["ret"] > np.mean(btc_hold)
        ok = c1 and c2 and c3
        passed.append(ok)
        print(f"[{name}]")
        print(f"  1) 무작위 +0.05R 초과   {a['avg_r']:+.3f} vs {base['avg_r'] + 0.05:+.3f}  {'O' if c1 else 'X'}")
        print(f"  2) 과반 구간 우세       {wins}/{len(splits)}  {'O' if c2 else 'X'}")
        print(f"  3) BTC 보유 초과        {a['ret']:+.2%} vs {np.mean(btc_hold):+.2%}  {'O' if c3 else 'X'}")
        print(f"  → {'통과' if ok else '미달'}\n")

    if not any(passed):
        print("사전 약속대로 여기서 능동 매매 탐색을 종료한다.")
        print("결론: 미국 주식과 크립토 양쪽에서 타이밍 기반 초과수익의 증거가 없다.")
        print("다음은 매수 후 보유다.")


def main() -> None:
    args = sys.argv[1:]
    panel = load(force="--fetch" in args)
    if "--fetch" in args and len(args) == 1:
        return
    if not args or "--all" in args:
        cmd_cost(panel)
        cmd_excess(panel)
        cmd_walkforward(panel)
        return
    if "--cost" in args:
        cmd_cost(panel)
    if "--excess" in args:
        cmd_excess(panel)
    if "--walkforward" in args:
        cmd_walkforward(panel)


if __name__ == "__main__":
    main()
