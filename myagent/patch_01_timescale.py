"""
패치 1 — 시간축 판정 수정 및 자본효율 기반 캘리브레이션

적용:
    myagent/ 폴더에 이 파일을 놓고
    python patch_01_timescale.py

무엇을 고치는가
----------------
기존 check_timescale_validity()는 '비율 편차 > 0.12'로 판정했는데,
이건 노이즈를 수렴으로 오인한다. 실제로 momentum이 잘못 '유효' 판정을 받았다.

올바른 지표는 탄력성이다:
    탄력성 = log(봉수 증가배) / log(관측범위 증가배)

    탄력성 ≈ 1  → 관측범위에 완전 비례 → 고유 시간축 없음(랜덤워크)
    탄력성 ≈ 0  → 관측범위와 무관하게 수렴 → 고유 시간축 있음

무엇을 추가하는가
----------------
시간축이 없어도 시간 손절 자체는 여전히 필요하다. 다만 목적이 다르다.

  잘못된 목적: "MFE가 나오는 시점에 맞춰 끊는다"  ← 시간축이 없으면 불가능
  올바른 목적: "죽은 포지션이 자본을 묶는 것을 막는다"  ← 자본 회전율 문제

그래서 R 총합이 아니라 '봉당 R'(R per bar held)을 최대화하는 값을 찾는다.
같은 +0.3R이라도 10봉 만에 나오는 것과 60봉 걸리는 것은 가치가 다르다.
"""

from __future__ import annotations

import os
import re
import sys

PATCH_VALIDITY = '''

def check_timescale_validity(
    panel: PricePanel,
    model: AlphaModel,
    cfg: RiskConfig,
    horizons: tuple[int, ...] = (30, 60, 90),
) -> dict:
    """
    시간 손절 캘리브레이션이 유효한지 검사한다. (수정판)

    지표: 탄력성 = log(MFE도달봉수 증가배) / log(관측범위 증가배)

      탄력성 ~ 1  → 관측범위에 비례 → 고유 시간축 없음 → MFE 기반 캘리브레이션 무효
      탄력성 ~ 0  → 수렴 → 고유 시간축 있음 → 캘리브레이션 유효

    이전 버전은 '비율 편차'로 판정했는데 노이즈를 수렴으로 오인했다.
    관측범위 30/60/90에서 비율이 0.70/0.85/0.83이면 편차는 0.15지만,
    봉수 자체는 21→51→75로 계속 늘어나므로 수렴이 아니다.
    """
    import math

    medians, used = [], []
    for h in horizons:
        exs = measure_excursions(panel, model, cfg, horizon=h, max_samples=2000)
        if not exs:
            continue
        medians.append(float(np.median([e.bars_to_mfe for e in exs])))
        used.append(h)

    if len(medians) < 2 or medians[0] <= 0:
        return {"valid": False, "reason": "표본 부족"}

    elasticity = math.log(medians[-1] / medians[0]) / math.log(used[-1] / used[0])
    valid = elasticity < 0.5

    return {
        "horizons": used,
        "bars_to_mfe": medians,
        "ratios": [round(m / h, 3) for m, h in zip(medians, used)],
        "elasticity": round(elasticity, 3),
        "valid": bool(valid),
        "reason": (
            f"탄력성 {elasticity:.2f} — 관측범위에 비례. 고유 시간축 없음(랜덤워크). "
            "MFE 기반 시간손절 캘리브레이션은 무효이므로 "
            "calibrate_time_stop_by_efficiency()를 사용할 것."
            if not valid
            else f"탄력성 {elasticity:.2f} — 수렴. 캘리브레이션 유효."
        ),
    }


def calibrate_time_stop_by_efficiency(
    panel: PricePanel,
    model: AlphaModel,
    cfg: RiskConfig,
    candidates: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60),
    horizon: int = 80,
    max_samples: int = 5000,
) -> dict:
    """
    자본효율 기반 시간손절 캘리브레이션.

    고유 시간축이 없을 때 쓴다. MFE 시점을 맞추려 하지 않고,
    '보유 1봉당 기대 R'을 최대화하는 지점을 찾는다.

    핵심: 총 R이 아니라 R/봉을 본다.
    같은 +0.3R이라도 10봉 만에 나오면 60봉 걸리는 것보다 6배 효율적이다.
    자본이 묶여 있는 동안 다른 기회를 놓치기 때문이다.

    주의: 이 값도 '최적해'가 아니라 '덜 나쁜 선택'이다.
    R/봉 곡선이 평평하면 어느 값을 골라도 큰 차이가 없다는 뜻이고,
    그때는 짧은 쪽(자본 회전이 빠른 쪽)을 고르는 게 낫다.
    """
    exs = measure_excursions(panel, model, cfg, horizon=horizon, max_samples=max_samples)
    if not exs:
        return {"n": 0, "reason": "표본 없음"}

    rows = {}
    for c in candidates:
        if c > horizon:
            continue
        r_at_c = np.array([e.final_r_by_bar.get(c, 0.0) for e in exs])
        mean_r = float(r_at_c.mean())
        rows[c] = {
            "avg_r": mean_r,
            "r_per_bar": mean_r / c,
            "hit_rate": float((r_at_c > 0).mean()),
        }

    best = max(rows, key=lambda c: rows[c]["r_per_bar"])
    peak = rows[best]["r_per_bar"]

    # 최고값의 90% 이상을 내는 가장 짧은 값 = 실질적으로 동등하면서 회전이 빠른 쪽
    practical = min(
        (c for c in rows if rows[c]["r_per_bar"] >= peak * 0.9),
        default=best,
    )
    flat = (max(v["r_per_bar"] for v in rows.values())
            - min(v["r_per_bar"] for v in rows.values())) < abs(peak) * 0.5

    return {
        "n": len(exs),
        "by_candidate": rows,
        "best_r_per_bar": best,
        "recommended": practical,
        "curve_is_flat": bool(flat),
        "note": (
            "R/봉 곡선이 평평하다. 시간 손절값의 영향이 작으므로 "
            "회전이 빠른 짧은 값을 택했다."
            if flat
            else "R/봉이 뚜렷한 최대를 가진다."
        ),
    }
'''


def main() -> None:
    path = "eval/harness.py"
    if not os.path.exists(path):
        sys.exit(f"{path}를 찾을 수 없습니다. myagent/ 폴더 안에서 실행하세요.")

    src = open(path, encoding="utf-8").read()

    if "elasticity" in src:
        print("이미 패치가 적용되어 있습니다.")
        return

    # 기존 check_timescale_validity 함수 전체를 제거
    idx = src.find("def check_timescale_validity(")
    if idx == -1:
        sys.exit("check_timescale_validity를 찾지 못했습니다. 파일이 예상과 다릅니다.")
    src = src[:idx].rstrip() + "\n" + PATCH_VALIDITY

    open(path, "w", encoding="utf-8").write(src)
    print(f"패치 완료: {path}")
    print()
    print("다음 명령으로 재검사하세요:")
    print("    python run_real.py --check")
    print("    python patch_01_timescale.py --calibrate")


def run_calibration() -> None:
    """패치 적용 후 자본효율 기반 캘리브레이션 실행."""
    from configs import LIVE_SMALL
    from eval.harness import calibrate_time_stop_by_efficiency
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, MomentumBreakout

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")

    print("\n" + "=" * 70)
    print("자본효율 기반 시간손절 캘리브레이션")
    print("=" * 70)
    print("MFE 시점을 맞추지 않는다. 보유 1봉당 기대 R을 최대화한다.\n")

    for model in (MomentumBreakout(), MeanReversion()):
        r = calibrate_time_stop_by_efficiency(panel, model, LIVE_SMALL)
        if not r.get("n"):
            print(f"[{model.name}] {r.get('reason')}")
            continue
        print(f"[{model.name}] 표본 {r['n']}건")
        print(f"  {'봉':>5}{'평균R':>10}{'R/봉':>11}{'승률':>9}")
        for c, v in sorted(r["by_candidate"].items()):
            mark = "  ←추천" if c == r["recommended"] else ""
            print(f"  {c:>5}{v['avg_r']:>+10.3f}{v['r_per_bar']:>+11.4f}{v['hit_rate']:>8.1%}{mark}")
        print(f"  → {r['recommended']}봉. {r['note']}\n")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        run_calibration()
    else:
        main()
