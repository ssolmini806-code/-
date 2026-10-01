"""
패치 2 — 기준선 대조 추가 및 KeyError 수정

적용:
    python patch_02_baseline.py            # 코드 수정
    python patch_02_baseline.py --run      # 기준선 대조 캘리브레이션 실행

무엇을 고치는가
----------------
1. run_real.py의 KeyError('ratio_spread')
   패치1에서 반환 키가 바뀌었는데 호출부를 안 고쳤다.

2. 캘리브레이션 결과 메시지 모순
   "곡선이 평평해서 짧은 값을 택했다"면서 가장 긴 60봉을 고르고 있었다.

3. 【핵심】 기준선 대조 부재
   평균 R이 보유 기간에 단조증가한다(5봉 +0.06R → 60봉 +1.12R).
   이건 신호의 성과가 아니라 시장 드리프트일 가능성이 크다.

   2015~2026 미국 대형주는 역사적 강세장이고, 유니버스는 생존 편향이 있다.
   이 구간에서는 아무 종목이나 사서 오래 들고 있으면 돈을 번다.

   따라서 '무작위 진입도 같은 기간 들고 있었으면 얼마였나'를 빼야
   신호가 실제로 기여한 몫이 보인다.

       초과 R = 신호 평균 R - 무작위 평균 R

   초과 R이 0 근처면, 그 신호는 진입 시점을 고르는 데 아무 값어치가 없다.
   그냥 아무 때나 사서 같은 기간 들고 있는 것과 같다.
"""

from __future__ import annotations

import os
import sys

FIXED_CMD_CHECK = '''def cmd_check(panel: PricePanel) -> dict:
    print("\\n" + "=" * 70)
    print("1. 시간축 유효성 검사")
    print("=" * 70)
    print("MFE 도달 봉수가 관측 범위에 비례하면 고유 시간축이 없다는 뜻이고,")
    print("그 경우 MFE 기반 time_stop 캘리브레이션은 무효다.\\n")

    out = {}
    for model in (MomentumBreakout(), MeanReversion()):
        r = check_timescale_validity(panel, model, LIVE_SMALL, horizons=(30, 60, 90))
        out[model.name] = r
        print(f"[{model.name}]")
        if "bars_to_mfe" not in r:
            print(f"  {r.get('reason')}")
            continue
        print(f"  관측범위 {r['horizons']} → MFE도달 {r['bars_to_mfe']}")
        print(f"  탄력성 {r['elasticity']} (1에 가까우면 비례 = 시간축 없음)")
        print(f"  → {'유효' if r['valid'] else '무효'}: {r['reason']}\\n")
    return out
'''


def patch_run_real() -> None:
    path = "run_real.py"
    src = open(path, encoding="utf-8").read()
    if "elasticity" in src:
        print("  run_real.py: 이미 적용됨")
        return
    start = src.find("def cmd_check(panel: PricePanel) -> dict:")
    end = src.find("def cmd_calibrate(")
    if start == -1 or end == -1:
        sys.exit("run_real.py 구조가 예상과 다릅니다.")
    open(path, "w", encoding="utf-8").write(src[:start] + FIXED_CMD_CHECK + "\n\n" + src[end:])
    print("  run_real.py: KeyError 수정 완료")


BASELINE_FN = '''

def calibrate_with_baseline(
    panel: PricePanel,
    model: AlphaModel,
    cfg: RiskConfig,
    baseline_seed: int = 99,
    candidates: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60),
    horizon: int = 80,
    max_samples: int = 5000,
) -> dict:
    """
    기준선 대조 캘리브레이션.

    같은 기간을 무작위 진입으로 보유했을 때의 평균 R을 빼서,
    신호가 실제로 기여한 몫만 남긴다.

        초과 R = 신호 평균 R - 무작위 평균 R

    강세장에서는 아무 때나 사도 오래 들고 있으면 R이 커진다.
    그 몫을 빼지 않으면 시장 드리프트를 신호의 성과로 착각한다.

    판정:
      초과 R이 보유 기간 내내 0 근처  → 신호가 진입 시점 선택에 기여하지 않음
      초과 R이 짧은 구간에서 크고 감소 → 단기 신호. 짧게 들고 나와야 함
      초과 R이 계속 증가             → 장기 신호. 시간 손절을 길게 (드물다)
    """
    from signals.models import RandomEntry

    sig_exs = measure_excursions(panel, model, cfg, horizon=horizon, max_samples=max_samples)
    base_exs = measure_excursions(
        panel, RandomEntry(probability=0.01, seed=baseline_seed), cfg,
        horizon=horizon, max_samples=max_samples,
    )
    if not sig_exs or not base_exs:
        return {"n": 0, "reason": "표본 부족"}

    rows = {}
    for c in candidates:
        if c > horizon:
            continue
        s = np.array([e.final_r_by_bar.get(c, 0.0) for e in sig_exs])
        b = np.array([e.final_r_by_bar.get(c, 0.0) for e in base_exs])
        excess = float(s.mean() - b.mean())

        # 초과분이 표본 노이즈로 설명되는지 (Welch t 근사)
        se = float(np.sqrt(s.var(ddof=1) / len(s) + b.var(ddof=1) / len(b)))
        t = excess / se if se > 0 else 0.0

        rows[c] = {
            "signal_r": float(s.mean()),
            "baseline_r": float(b.mean()),
            "excess_r": excess,
            "excess_per_bar": excess / c,
            "t_stat": t,
            "significant": abs(t) > 2.0,
        }

    sig_rows = {c: v for c, v in rows.items() if v["significant"] and v["excess_r"] > 0}
    if sig_rows:
        best = max(sig_rows, key=lambda c: sig_rows[c]["excess_per_bar"])
        verdict = "신호가 기준선을 유의하게 초과"
    else:
        best = None
        verdict = "초과분이 노이즈 범위 — 신호가 진입 시점 선택에 기여하지 않음"

    return {
        "n_signal": len(sig_exs),
        "n_baseline": len(base_exs),
        "by_candidate": rows,
        "recommended": best,
        "verdict": verdict,
    }
'''


def patch_harness() -> None:
    path = "eval/harness.py"
    src = open(path, encoding="utf-8").read()
    if "calibrate_with_baseline" in src:
        print("  eval/harness.py: 이미 적용됨")
        return

    # 메시지 모순 수정
    src = src.replace(
        '''        "note": (
            "R/봉 곡선이 평평하다. 시간 손절값의 영향이 작으므로 "
            "회전이 빠른 짧은 값을 택했다."
            if flat
            else "R/봉이 뚜렷한 최대를 가진다."
        ),''',
        '''        "note": (
            (f"R/봉 최대는 {best}봉, 그 90% 이상을 내는 최단값은 {practical}봉."
             + (" 곡선이 평평해 값 선택의 영향이 작다." if flat else ""))
        ),''',
    )
    open(path, "w", encoding="utf-8").write(src + BASELINE_FN)
    print("  eval/harness.py: 기준선 대조 함수 추가 + 메시지 수정")


def run() -> None:
    from configs import LIVE_SMALL
    from eval.harness import calibrate_with_baseline
    from run_real import DEFAULT_TICKERS, load_panel
    from signals.models import MeanReversion, MomentumBreakout

    panel = load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")

    print("\n" + "=" * 74)
    print("기준선 대조 캘리브레이션")
    print("=" * 74)
    print("같은 기간 무작위 진입으로 보유했을 때를 빼서 신호의 순수 기여만 본다.")
    print("강세장에서는 아무 때나 사도 오래 들면 R이 커지므로 그 몫을 제거해야 한다.\n")

    for model in (MomentumBreakout(), MeanReversion()):
        r = calibrate_with_baseline(panel, model, LIVE_SMALL)
        if not r.get("n_signal"):
            print(f"[{model.name}] {r.get('reason')}")
            continue

        print(f"[{model.name}] 신호 {r['n_signal']}건 vs 무작위 {r['n_baseline']}건")
        print(f"  {'봉':>5}{'신호R':>10}{'무작위R':>10}{'초과R':>10}{'초과/봉':>10}{'t':>8}  유의")
        for c, v in sorted(r["by_candidate"].items()):
            mark = "  ←" if c == r["recommended"] else ""
            print(f"  {c:>5}{v['signal_r']:>+10.3f}{v['baseline_r']:>+10.3f}"
                  f"{v['excess_r']:>+10.3f}{v['excess_per_bar']:>+10.4f}"
                  f"{v['t_stat']:>8.2f}{'   O' if v['significant'] else '   X'}{mark}")
        print(f"  → {r['verdict']}")
        if r["recommended"]:
            print(f"     추천 time_stop_bars = {r['recommended']}")
        print()

    print("=" * 74)
    print("읽는 법")
    print("=" * 74)
    print("초과R이 0 근처이고 t가 2 미만이면, 그 신호는 진입 시점을 고르는 데")
    print("값어치가 없다는 뜻이다. 아무 때나 사서 같은 기간 들고 있는 것과 같다.")
    print()
    print("이 경우 할 일은 파라미터를 바꿔 통과시키는 게 아니라,")
    print("신호 종류를 바꾸거나 3단계 LLM 필터로 넘어가는 것이다.")


def main() -> None:
    if not os.path.exists("run_real.py") or not os.path.exists("eval/harness.py"):
        sys.exit("myagent/ 폴더 안에서 실행하세요.")
    print("패치 적용 중...")
    patch_run_real()
    patch_harness()
    print()
    print("완료. 다음을 실행하세요:")
    print("    python run_real.py --check")
    print("    python patch_02_baseline.py --run")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    else:
        main()
