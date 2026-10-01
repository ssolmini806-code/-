"""
실제 데이터 실행 CLI.

사용법:
    python run_real.py --check          # 시간축 유효성만 검사 (빠름)
    python run_real.py --calibrate      # MAE/MFE로 time_stop 확정
    python run_real.py --walkforward    # walk-forward 판정
    python run_real.py --all            # 전부

데이터는 data_cache/에 저장되어 재실행 시 다시 받지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import replace

import numpy as np

from configs import LIVE_SMALL, TIME_STOP_CANDIDATES, account_krw_to_usd
from cycle import buy_and_hold_benchmark, run_backtest
from data.contract import PricePanel
from eval.harness import (
    check_timescale_validity,
    make_walk_forward_splits,
    measure_excursions,
    summarize_excursions,
)
from signals.models import MeanReversion, MomentumBreakout, RandomEntry

CACHE_DIR = "data_cache"

# 기본 유니버스.
#
# ⚠ 생존 편향 주의: 아래는 '오늘 시점의 대형주'다.
#    2015년에 이 종목들을 고를 수 있었을 리 없다. 과거로 갈수록
#    백테스트 성과가 실제보다 좋게 나온다(망한 회사가 빠져 있으므로).
#    엄밀히 하려면 시점별 지수 구성종목이 필요하고, 그건 유료 데이터다.
#    지금 단계에서는 이 한계를 알고 결과를 할인해서 읽는 것으로 충분하다.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "TSLA",
    "JPM", "V", "MA", "UNH", "JNJ", "XOM", "CVX", "WMT",
    "PG", "HD", "KO", "PEP", "ABBV", "MRK", "COST", "ADBE",
    "CRM", "NFLX", "AMD", "INTC", "QCOM", "TXN",
]


def load_panel(tickers: list[str], start: str, end: str, refresh: bool = False) -> PricePanel:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{len(tickers)}_{start}_{end}_{hash(tuple(sorted(tickers))) & 0xFFFFFF:06x}"
    path = os.path.join(CACHE_DIR, f"panel_{key}.pkl")

    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            panel = pickle.load(f)
        print(f"캐시 사용: {path} (종목 {len(panel.symbols)} · 봉 {panel.n_bars})")
        return panel

    print(f"yfinance 다운로드 중... 종목 {len(tickers)} · {start} ~ {end}")
    from data.contract import load_yfinance_panel

    panel = load_yfinance_panel(tickers, start, end)
    if not panel.symbols:
        raise SystemExit("데이터를 받지 못했습니다. 네트워크와 티커를 확인하세요.")
    with open(path, "wb") as f:
        pickle.dump(panel, f)
    print(f"저장: {path} (종목 {len(panel.symbols)} · 봉 {panel.n_bars})")
    return panel


def cmd_check(panel: PricePanel) -> dict:
    print("\n" + "=" * 70)
    print("1. 시간축 유효성 검사")
    print("=" * 70)
    print("MFE 도달 봉수가 관측 범위에 비례하면 고유 시간축이 없다는 뜻이고,")
    print("그 경우 time_stop_bars는 아무 값이나 넣는 것과 같다.\n")

    out = {}
    for model in (MomentumBreakout(), MeanReversion()):
        r = check_timescale_validity(panel, model, LIVE_SMALL, horizons=(30, 60, 90))
        out[model.name] = r
        print(f"[{model.name}]")
        if "bars_to_mfe" not in r:
            print(f"  {r.get('reason')}")
            continue
        print(f"  관측범위 {r['horizons']} → MFE도달 {r['bars_to_mfe']}")
        print(f"  비율 {r['ratios']} (편차 {r['ratio_spread']})")
        print(f"  → {'유효' if r['valid'] else '무효'}: {r['reason']}\n")
    return out


def cmd_calibrate(panel: PricePanel) -> dict[str, int]:
    print("\n" + "=" * 70)
    print("2. MAE/MFE 캘리브레이션 — time_stop_bars 확정")
    print("=" * 70)

    chosen: dict[str, int] = {}
    for model in (MomentumBreakout(), MeanReversion()):
        exs = measure_excursions(panel, model, LIVE_SMALL, horizon=60, max_samples=5000)
        s = summarize_excursions(exs, TIME_STOP_CANDIDATES)
        if not s:
            print(f"\n[{model.name}] 표본 없음 — 신호 조건이 너무 빡빡함")
            continue

        print(f"\n[{model.name}] 표본 {s['n']}건")
        print(f"  MFE 중앙 {s['mfe_median']:+.2f}R · MAE 중앙 {s['mae_median']:+.2f}R")
        print(f"  +1R 도달 {s['pct_reach_1r']:.1%} · 도달 중앙 {s['bars_to_1r_median']:.0f}봉")
        for c, v in s["by_candidate"].items():
            print(f"    {c:>3}봉 → 평균 {v['avg_r_at_cut']:+.3f}R · +1R 조기절단 {v['pct_1r_cut_early']:.1%}")

        pick = next(
            (c for c in sorted(s["by_candidate"]) if s["by_candidate"][c]["pct_1r_cut_early"] < 0.20),
            None,
        )
        if pick is None:
            pick = max(s["by_candidate"])
            print(f"  → {pick}봉 (⚠ 20% 기준 미달성, 최댓값 폴백 — 실제 최적값은 더 클 수 있음)")
        else:
            print(f"  → {pick}봉 확정")
        chosen[model.name] = pick

    if chosen:
        print(f"\nconfigs.py의 time_stop_bars에 반영하세요: {chosen}")
    return chosen


def cmd_walkforward(panel: PricePanel, time_stops: dict[str, int]) -> None:
    print("\n" + "=" * 70)
    print("3. Walk-forward 판정")
    print("=" * 70)

    equity = account_krw_to_usd(1_000_000)
    splits = make_walk_forward_splits(panel.n_bars, train_bars=750, test_bars=250)
    if not splits:
        raise SystemExit("데이터가 짧아 walk-forward 구간을 만들 수 없습니다. 기간을 늘리세요.")
    print(f"구간 {len(splits)}개 (train 750 / test 250). test 성과만 인정.\n")

    models = {
        "random": (RandomEntry(probability=0.015, seed=1), LIVE_SMALL.time_stop_bars),
        "momentum": (MomentumBreakout(), time_stops.get("momentum", 20)),
        "meanrev": (MeanReversion(), time_stops.get("meanrev", 20)),
    }

    per_split: dict[str, list] = {k: [] for k in models}
    bh = []
    for sp in splits:
        bh.append(buy_and_hold_benchmark(panel, sp.test_start, sp.test_end))
        for name, (model, ts) in models.items():
            cfg = replace(LIVE_SMALL, time_stop_bars=ts)
            pf, _ = run_backtest(panel, model, cfg, equity,
                                 start_bar=sp.test_start, end_bar=sp.test_end)
            per_split[name].append(pf.stats())

    hdr = f"{'모델':<10}{'거래':>7}{'승률':>9}{'평균R':>9}{'수익':>10}{'MDD':>9}{'Sharpe':>9}"
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for name, rows in per_split.items():
        live = [r for r in rows if r["trades"]]
        n = sum(r["trades"] for r in rows)
        agg[name] = {
            "avg_r": float(np.mean([r["avg_r"] for r in live])) if live else 0.0,
            "ret": float(np.mean([r["total_return"] for r in rows])),
            "sharpe": float(np.mean([r["sharpe"] for r in rows])),
            "per_split_r": [r["avg_r"] for r in rows],
        }
        wr = np.mean([r["win_rate"] for r in live]) if live else 0
        dd = np.mean([r["max_drawdown"] for r in rows])
        print(f"{name:<10}{n:>7}{wr:>8.1%}{agg[name]['avg_r']:>+9.3f}"
              f"{agg[name]['ret']:>+10.2%}{dd:>9.2%}{agg[name]['sharpe']:>9.2f}")
    print("-" * len(hdr))
    print(f"{'매수보유':<10}{'':>7}{'':>9}{'':>9}{np.mean(bh):>+10.2%}")

    # 통과 판정
    print("\n" + "=" * 70)
    print("판정 — 셋 다 충족해야 3단계 진행")
    print("=" * 70)
    base = agg["random"]["avg_r"]
    base_splits = agg["random"]["per_split_r"]
    print(f"무작위 기준선: 평균 {base:+.3f}R\n")

    for name in ("momentum", "meanrev"):
        a = agg[name]
        c1 = a["avg_r"] > base + 0.05
        wins = sum(1 for x, y in zip(a["per_split_r"], base_splits) if x > y)
        c2 = wins > len(splits) / 2
        c3 = a["sharpe"] >= 0
        print(f"[{name}]")
        print(f"  1) 기준선 +0.05R 초과      {a['avg_r']:+.3f} vs {base + 0.05:+.3f}  {'O' if c1 else 'X'}")
        print(f"  2) 과반 구간에서 우세       {wins}/{len(splits)}  {'O' if c2 else 'X'}")
        print(f"  3) Sharpe 음수 아님        {a['sharpe']:+.2f}  {'O' if c3 else 'X'}")
        print(f"  → {'통과' if (c1 and c2 and c3) else '미달'}\n")

    print("미달이어도 파라미터를 바꿔가며 통과할 때까지 돌리지 말 것.")
    print("그건 과적합을 만드는 가장 확실한 방법이다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 재다운로드")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not any([args.check, args.calibrate, args.walkforward, args.all]):
        args.all = True

    panel = load_panel(args.tickers, args.start, args.end, args.refresh)
    print("\n⚠ 유니버스는 '오늘의 대형주'라 생존 편향이 있다. 결과를 할인해서 읽을 것.")

    results = {}
    if args.check or args.all:
        results["timescale"] = cmd_check(panel)

    time_stops: dict[str, int] = {}
    if args.calibrate or args.all:
        time_stops = cmd_calibrate(panel)
        results["time_stops"] = time_stops

    if args.walkforward or args.all:
        cmd_walkforward(panel, time_stops)

    os.makedirs("results", exist_ok=True)
    with open("results/last_run.json", "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print("\n결과 저장: results/last_run.json")


if __name__ == "__main__":
    main()
