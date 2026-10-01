"""
패치 3 — 시간 클러스터 부트스트랩 (최종 검정)

적용:
    python patch_03_cluster.py --run

왜 필요한가
------------
패치2의 t검정은 표본 독립을 가정한다. 그런데 실제로는:

  - 같은 날 여러 종목이 동시에 신호를 낸다 (시장 전체가 함께 움직이므로)
  - 60봉 관측 구간이 서로 크게 중첩된다
  - 따라서 신호 5000건의 실효 표본수는 5000보다 훨씬 작다

독립을 가정하면 표준오차가 과소평가되고, t값이 부풀려진다.
패치2에서 나온 meanrev 60봉 t=2.29도 이 문제를 안고 있다.

무엇을 하는가
--------------
1. 진입 시점을 '월' 단위로 묶어 월별 평균 초과R을 구한다.
   같은 달 안의 종목 간 상관은 평균 안에서 흡수된다.

2. 월 블록을 복원추출해 부트스트랩 분포를 만든다.
   연속된 달끼리의 상관까지 다루려면 블록 길이를 늘린다.

3. 다중검정 보정을 적용한다.
   후보 7개 x 모델 2개 = 14회 검정이므로 임계값이 |t| > 2.91이다.

이 검정을 통과하지 못하면, 그 신호에는 증거가 없는 것이다.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

BARS_PER_MONTH = 21


def cluster_bootstrap_excess(
    signal_exs: list,
    baseline_exs: list,
    bars: int,
    block_months: int = 3,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """
    월 클러스터 블록 부트스트랩으로 초과R의 신뢰구간을 구한다.

    반환:
      excess     관측된 초과R
      ci_low/high  95% 신뢰구간
      p_value    초과R <= 0일 확률 (단측)
      n_months   유효 월 수 = 실효 표본수의 현실적 추정
    """
    rng = np.random.default_rng(seed)

    def by_month(exs):
        d = defaultdict(list)
        for e in exs:
            d[e.entry_bar // BARS_PER_MONTH].append(e.final_r_by_bar.get(bars, 0.0))
        return {m: float(np.mean(v)) for m, v in d.items() if v}

    sig_m = by_month(signal_exs)
    base_m = by_month(baseline_exs)
    months = sorted(set(sig_m) & set(base_m))
    if len(months) < 12:
        return {"n_months": len(months), "reason": "월 표본 부족"}

    diffs = np.array([sig_m[m] - base_m[m] for m in months])
    observed = float(diffs.mean())

    # 블록 부트스트랩 (연속 월 간 상관 처리)
    n_blocks = max(1, len(diffs) // block_months)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, len(diffs) - block_months), size=n_blocks)
        sample = np.concatenate([diffs[s : s + block_months] for s in starts])
        boots[i] = sample.mean()

    return {
        "excess": observed,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_value": float((boots <= 0).mean()),
        "n_months": len(months),
        "t_equivalent": float(observed / boots.std()) if boots.std() > 0 else 0.0,
    }


def run() -> None:
    from configs import LIVE_SMALL
    from eval.harness import measure_excursions
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, MomentumBreakout, RandomEntry

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")
    candidates = (5, 10, 20, 30, 40, 60)
    n_tests = len(candidates) * 2

    from statistics import NormalDist

    crit = NormalDist().inv_cdf(1 - (0.05 / n_tests) / 2)

    print("\n" + "=" * 78)
    print("시간 클러스터 부트스트랩 — 최종 검정")
    print("=" * 78)
    print(f"월 단위 클러스터 + 3개월 블록 부트스트랩 (5,000회)")
    print(f"다중검정 보정: {n_tests}회 검정 → 임계값 |t| > {crit:.2f}\n")

    print("기준선 표본 확보 중 (무작위 진입 밀도 상향)...")
    base_exs = measure_excursions(
        panel, RandomEntry(probability=0.04, seed=99), LIVE_SMALL,
        horizon=80, max_samples=8000,
    )
    print(f"  무작위 {len(base_exs)}건\n")

    any_pass = False
    for model in (MomentumBreakout(), MeanReversion()):
        sig_exs = measure_excursions(panel, model, LIVE_SMALL, horizon=80, max_samples=8000)
        print(f"[{model.name}] 신호 {len(sig_exs)}건")
        print(f"  {'봉':>5}{'초과R':>10}{'95% 신뢰구간':>22}{'유효t':>9}{'p':>8}  판정")

        for c in candidates:
            r = cluster_bootstrap_excess(sig_exs, base_exs, c)
            if "excess" not in r:
                print(f"  {c:>5}  {r.get('reason')}")
                continue
            ok = abs(r["t_equivalent"]) > crit and r["excess"] > 0
            any_pass = any_pass or ok
            ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            print(f"  {c:>5}{r['excess']:>+10.3f}{ci:>22}"
                  f"{r['t_equivalent']:>9.2f}{r['p_value']:>8.3f}"
                  f"  {'통과' if ok else '미달'}")

        print(f"  (월 표본 {cluster_bootstrap_excess(sig_exs, base_exs, 20).get('n_months')}개월)\n")

    print("=" * 78)
    if any_pass:
        print("결론: 일부 조합이 보정 후에도 유의하다. 다만 생존 편향이 남아 있으므로")
        print("      다른 유니버스(중소형주, 다른 기간)에서 재현되는지 확인할 것.")
    else:
        print("결론: 두 신호 모두 무작위 진입 대비 증거 없음.")
        print()
        print("이건 실패가 아니라 결과다. 55일 신고가 돌파와 RSI(3) 과매도는")
        print("널리 알려진 규칙이라 초과수익이 소멸했을 가능성이 높다.")
        print()
        print("여기서 파라미터를 바꿔가며 통과할 때까지 돌리면 과적합이 된다.")
        print("선택지는 세 가지다:")
        print("  A. 다른 종류의 신호를 시도 (변동성 수축, 섹터 상대강도, 실적 후 드리프트)")
        print("  B. 진입을 포기하고 출구 로직만 개선 (무작위 진입 + 좋은 출구)")
        print("  C. 3단계로 넘어가 LLM 필터가 기여하는지 같은 방식으로 검증")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    else:
        print("사용법: python patch_03_cluster.py --run")
