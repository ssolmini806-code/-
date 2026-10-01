"""
패치 5 — 출구 불일치 진단 및 train 기반 출구 도출

적용:
    python patch_05_exit_fit.py --diagnose     # 왜 walk-forward에서 죽었는가
    python patch_05_exit_fit.py --walkforward  # train에서만 출구를 도출해 재검정
    python patch_05_exit_fit.py --all

상황
-----
meanrev 신호는 궤적 분석에서 일관되게 초과수익을 보였다:
  개별주 10봉 +0.165R (t=3.47) / ETF 10봉 +0.189R (t=3.62)
  전반기 +0.169R (t=2.59) / 후반기 +0.184R (t=2.86)
생존 편향 의심은 ETF 재현으로 상당히 해소됐다.

그런데 실제 거래 규칙을 적용한 walk-forward에서는 미달했다
(기준선 +0.126R vs meanrev +0.096R).

가설
-----
출구 규칙이 추세추종용으로 설계되어 평균회귀 신호와 맞지 않는다.

  현재: 손절 2xATR(=-1R), 익절 2.5R(=5xATR), 추적 +1R 발동
  평균회귀: 급락 직후 진입이라 ATR이 이미 높고, 며칠 더 흘러내리기 쉽고,
            반등 폭은 작다.

궤적 분석은 "10봉 뒤 종가"만 봤으므로, 중간에 -1R을 찍고 털렸을 경로가
그대로 포함돼 있다. 즉 +0.165R은 "끝까지 버텼다면"의 값이다.

과적합을 피하는 방법
---------------------
출구 파라미터를 백테스트 성과가 좋아질 때까지 바꾸면 과적합이다.
대신 각 walk-forward 구간의 **train 데이터에서만** MAE/MFE 분포를 측정해
출구를 도출하고, test 구간에는 그 값을 그대로 적용한다.

  train MAE 90퍼센타일 → 손절 배수
  train MFE 중앙값     → 익절 배수

test 성과는 파라미터 선택에 전혀 관여하지 않는다.
이렇게 해도 미달하면, 그 신호는 거래 가능한 형태가 아니다.

사전 약속 (중요)
-----------------
이번 한 번만 시도한다. 여기서 미달하면 파라미터를 더 만지지 않고
B(출구만 개선) 또는 C(3단계 LLM 필터)로 넘어간다.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np


def diagnose(panel, model, cfg, window: int = 10) -> dict:
    """
    궤적과 실제 거래의 격차를 분해한다.

    핵심 질문: 10봉 뒤 종가가 좋았던 경로들 중 몇 %가
              그 전에 손절선(-1R)을 건드렸는가?
    """
    from eval.harness import measure_excursions

    exs = measure_excursions(panel, model, cfg, horizon=window + 5, max_samples=8000)
    if not exs:
        return {}

    mae = np.array([e.mae_r for e in exs])
    mfe = np.array([e.mfe_r for e in exs])
    final = np.array([e.final_r_by_bar.get(window, 0.0) for e in exs])

    stopped = mae <= -1.0                      # 손절선 도달
    hit_target = mfe >= cfg.reward_risk_ratio  # 익절선 도달

    # 손절에 걸리지 않았다면 얼마였을까
    survived = final[~stopped]

    return {
        "n": len(exs),
        "stop_hit_rate": float(stopped.mean()),
        "target_hit_rate": float(hit_target.mean()),
        "final_all": float(final.mean()),
        "final_survivors": float(survived.mean()) if len(survived) else 0.0,
        "mae_p10": float(np.percentile(mae, 10)),
        "mae_p25": float(np.percentile(mae, 25)),
        "mfe_median": float(np.median(mfe)),
        "mfe_p75": float(np.percentile(mfe, 75)),
        # 손절을 안 맞았을 때의 성과 x 생존율 = 실제 기대치의 근사
        "implied": float(
            (1 - stopped.mean()) * (survived.mean() if len(survived) else 0.0)
            + stopped.mean() * -1.0
        ),
    }


def derive_exit_from_train(panel, model, base_cfg, train_start, train_end, window=10) -> dict:
    """
    train 구간에서만 출구 파라미터를 도출한다.

    손절 배수: MAE 하위 15퍼센타일을 견디도록.
               너무 타이트하면 정상 노이즈에 털리고,
               너무 넓으면 1회 손실이 커져 사이징이 줄어든다.
    익절 배수: MFE 중앙값 근처. 평균회귀 반등은 작으므로 멀리 두면 도달 못 한다.
    """
    from data.contract import PricePanel
    from eval.harness import measure_excursions

    sub = PricePanel(
        {s: df.iloc[train_start : train_end + 1].reset_index(drop=True)
         for s, df in panel.frames.items()}
    )
    exs = measure_excursions(sub, model, base_cfg, horizon=window + 5, max_samples=6000)
    if len(exs) < 100:
        return {"atr_stop_multiple": base_cfg.atr_stop_multiple,
                "reward_risk_ratio": base_cfg.reward_risk_ratio,
                "source": "표본 부족 — 기본값 유지"}

    # MAE는 현재 손절폭(2xATR = 1R) 기준이므로 배수로 환산
    mae_p15 = float(np.percentile([e.mae_r for e in exs], 15))
    mfe_med = float(np.median([e.mfe_r for e in exs]))

    stop_mult = float(np.clip(base_cfg.atr_stop_multiple * abs(mae_p15), 1.0, 5.0))
    # 새 손절폭 기준으로 익절을 다시 R 단위로 환산
    scale = base_cfg.atr_stop_multiple / stop_mult
    rr = float(np.clip(mfe_med * scale, 0.5, 3.0))

    return {
        "atr_stop_multiple": round(stop_mult, 2),
        "reward_risk_ratio": round(rr, 2),
        "mae_p15": round(mae_p15, 3),
        "mfe_median": round(mfe_med, 3),
        "source": "train 구간 MAE/MFE",
    }


def cmd_diagnose() -> None:
    from configs import LIVE_SMALL
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, RandomEntry

    print("\n" + "=" * 78)
    print("진단 — 궤적 분석과 실제 거래의 격차")
    print("=" * 78)

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")
    for label, model in [("meanrev", MeanReversion()),
                         ("random", RandomEntry(probability=0.04, seed=99))]:
        d = diagnose(panel, model, LIVE_SMALL)
        if not d:
            continue
        print(f"\n[{label}] 표본 {d['n']}건 (10봉 기준)")
        print(f"  손절선(-1R) 도달률        {d['stop_hit_rate']:.1%}")
        print(f"  익절선(+{LIVE_SMALL.reward_risk_ratio}R) 도달률       {d['target_hit_rate']:.1%}")
        print(f"  10봉 뒤 평균 (전체)        {d['final_all']:+.3f}R")
        print(f"  10봉 뒤 평균 (손절 회피분) {d['final_survivors']:+.3f}R")
        print(f"  손절 반영한 실질 기대치     {d['implied']:+.3f}R   ← 실제로 얻는 값")
        print(f"  MAE 하위10% {d['mae_p10']:+.2f}R · MFE 중앙 {d['mfe_median']:+.2f}R")

    print("\n  읽는 법:")
    print("    '10봉 뒤 평균'과 '실질 기대치'의 차이가 크면,")
    print("    엣지가 손절선에서 잘려나가고 있다는 뜻이다.")
    print("    익절 도달률이 매우 낮으면 익절선이 너무 멀다는 뜻이다.")


def cmd_walkforward() -> None:
    from configs import LIVE_SMALL, account_krw_to_usd
    from cycle import buy_and_hold_benchmark, run_backtest
    from eval.harness import make_walk_forward_splits
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, RandomEntry

    print("\n" + "=" * 78)
    print("Walk-forward — 출구를 train 구간에서만 도출")
    print("=" * 78)
    print("test 성과는 파라미터 선택에 관여하지 않는다.\n")

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")
    equity = account_krw_to_usd(1_000_000)
    splits = make_walk_forward_splits(panel.n_bars, train_bars=750, test_bars=250)

    mr_rows, rnd_rows, bh = [], [], []
    print(f"  {'구간':<6}{'손절x':>7}{'손익비':>7}{'meanrev R':>12}{'random R':>11}{'거래':>7}")
    print("  " + "-" * 52)

    for sp in splits:
        params = derive_exit_from_train(panel, MeanReversion(), LIVE_SMALL,
                                        sp.train_start, sp.train_end)
        cfg_mr = replace(
            LIVE_SMALL,
            atr_stop_multiple=params["atr_stop_multiple"],
            reward_risk_ratio=params["reward_risk_ratio"],
            time_stop_bars=10,
        )
        pf_mr, _ = run_backtest(panel, MeanReversion(), cfg_mr, equity,
                                start_bar=sp.test_start, end_bar=sp.test_end)
        pf_rn, _ = run_backtest(panel, RandomEntry(probability=0.015, seed=1), LIVE_SMALL,
                                equity, start_bar=sp.test_start, end_bar=sp.test_end)
        a, b = pf_mr.stats(), pf_rn.stats()
        mr_rows.append(a)
        rnd_rows.append(b)
        bh.append(buy_and_hold_benchmark(panel, sp.test_start, sp.test_end))
        print(f"  {sp.name:<6}{params['atr_stop_multiple']:>7.2f}{params['reward_risk_ratio']:>7.2f}"
              f"{a['avg_r']:>+12.3f}{b['avg_r']:>+11.3f}{a['trades']:>7}")

    mr_live = [r for r in mr_rows if r["trades"]]
    rn_live = [r for r in rnd_rows if r["trades"]]
    mr_r = float(np.mean([r["avg_r"] for r in mr_live])) if mr_live else 0.0
    rn_r = float(np.mean([r["avg_r"] for r in rn_live])) if rn_live else 0.0
    wins = sum(1 for a, b in zip(mr_rows, rnd_rows) if a["avg_r"] > b["avg_r"])

    print("  " + "-" * 52)
    print(f"  {'평균':<6}{'':>7}{'':>7}{mr_r:>+12.3f}{rn_r:>+11.3f}")
    print(f"\n  meanrev 수익 {np.mean([r['total_return'] for r in mr_rows]):+.2%} · "
          f"Sharpe {np.mean([r['sharpe'] for r in mr_rows]):+.2f}")
    print(f"  random  수익 {np.mean([r['total_return'] for r in rnd_rows]):+.2%} · "
          f"Sharpe {np.mean([r['sharpe'] for r in rnd_rows]):+.2f}")
    print(f"  매수보유 수익 {np.mean(bh):+.2%}")

    c1 = mr_r > rn_r + 0.05
    c2 = wins > len(splits) / 2
    c3 = np.mean([r["sharpe"] for r in mr_rows]) >= 0
    print(f"\n  1) 기준선 +0.05R 초과   {mr_r:+.3f} vs {rn_r + 0.05:+.3f}  {'O' if c1 else 'X'}")
    print(f"  2) 과반 구간 우세       {wins}/{len(splits)}  {'O' if c2 else 'X'}")
    print(f"  3) Sharpe 음수 아님     {np.mean([r['sharpe'] for r in mr_rows]):+.2f}  {'O' if c3 else 'X'}")
    print(f"  → {'통과' if (c1 and c2 and c3) else '미달'}")

    if not (c1 and c2 and c3):
        print()
        print("  사전 약속대로 여기서 파라미터 조정을 멈춘다.")
        print("  다음은 B(출구만 개선) 또는 C(3단계 LLM 필터)다.")


def main() -> None:
    args = sys.argv[1:]
    if not args or "--all" in args:
        cmd_diagnose()
        cmd_walkforward()
        return
    if "--diagnose" in args:
        cmd_diagnose()
    if "--walkforward" in args:
        cmd_walkforward()


if __name__ == "__main__":
    main()
