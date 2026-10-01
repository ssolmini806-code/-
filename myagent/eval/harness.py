"""
평가 하네스.

두 가지를 한다:
  1. MAE/MFE 분석 → time_stop_bars를 추측이 아니라 실측으로 확정
  2. Walk-forward → 한 구간에 맞춘 과적합을 걸러낸다

MAE = Maximum Adverse Excursion, 진입 후 최대 역행폭
MFE = Maximum Favorable Excursion, 진입 후 최대 순행폭

MFE가 보통 몇 봉째에 나오는지를 보면 시간 손절을 언제 걸어야 하는지 알 수 있다.
MFE 중앙값이 12봉째에 나오는데 10봉에서 끊으면 이익을 스스로 잘라내는 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.contract import PricePanel
from risk.rules import RiskConfig, Side, compute_stop_price
from signals.models import AlphaModel


@dataclass
class Excursion:
    symbol: str
    entry_bar: int
    mae_r: float          # 최대 역행폭 (R 단위, 음수)
    mfe_r: float          # 최대 순행폭 (R 단위, 양수)
    bars_to_mfe: int      # MFE까지 걸린 봉 수  ← 시간 손절의 근거
    final_r_by_bar: dict[int, float]


def measure_excursions(
    panel: PricePanel,
    model: AlphaModel,
    cfg: RiskConfig,
    horizon: int = 40,
    max_samples: int = 4000,
) -> list[Excursion]:
    """
    진입 신호마다 이후 horizon봉의 궤적을 R 단위로 기록한다.

    주의: 이 측정은 손절·익절을 적용하지 않은 '순수 궤적'이다.
    출구 규칙을 적용하면 궤적이 잘려서 MFE를 관측할 수 없기 때문이다.
    """
    from risk.indicators import atr as atr_fn

    atr_cache = {
        s: atr_fn(panel.frames[s]["high"], panel.frames[s]["low"], panel.frames[s]["close"], cfg.atr_period)
        for s in panel.symbols
    }
    warmup = max(model.warmup_bars, cfg.atr_period + 5)
    out: list[Excursion] = []

    for t in range(warmup, panel.n_bars - horizon - 1):
        if len(out) >= max_samples:
            break
        for sig in model.generate(panel, as_of=t):
            if len(out) >= max_samples:
                break
            a = float(atr_cache[sig.symbol].iloc[t])
            if not np.isfinite(a) or a <= 0:
                continue

            entry = float(panel.frames[sig.symbol]["open"].iloc[t + 1])
            stop = compute_stop_price(entry, a, sig.side, cfg)
            r_unit = abs(entry - stop)
            if r_unit <= 0:
                continue

            df = panel.frames[sig.symbol]
            mae_r, mfe_r, bars_to_mfe = 0.0, 0.0, 0
            by_bar: dict[int, float] = {}

            for k in range(1, horizon + 1):
                idx = t + k
                if idx >= len(df):
                    break
                hi, lo, cl = float(df["high"].iloc[idx]), float(df["low"].iloc[idx]), float(df["close"].iloc[idx])
                if sig.side is Side.LONG:
                    adverse, favorable = (lo - entry) / r_unit, (hi - entry) / r_unit
                    closed = (cl - entry) / r_unit
                else:
                    adverse, favorable = (entry - hi) / r_unit, (entry - lo) / r_unit
                    closed = (entry - cl) / r_unit

                if adverse < mae_r:
                    mae_r = adverse
                if favorable > mfe_r:
                    mfe_r, bars_to_mfe = favorable, k
                by_bar[k] = closed

            out.append(Excursion(sig.symbol, t, mae_r, mfe_r, bars_to_mfe, by_bar))

    return out


def summarize_excursions(exs: list[Excursion], candidates: tuple[int, ...]) -> dict:
    """시간 손절 후보값별로 '얼마나 이익을 잘라먹는지'를 계산한다."""
    if not exs:
        return {}

    mfe = np.array([e.mfe_r for e in exs])
    mae = np.array([e.mae_r for e in exs])
    btm = np.array([e.bars_to_mfe for e in exs])

    reached_1r = mfe >= 1.0
    result = {
        "n": len(exs),
        "mfe_median": float(np.median(mfe)),
        "mfe_p75": float(np.percentile(mfe, 75)),
        "mae_median": float(np.median(mae)),
        "mae_p25": float(np.percentile(mae, 25)),
        "pct_reach_1r": float(reached_1r.mean()),
        "bars_to_mfe_median": float(np.median(btm)),
        "bars_to_1r_median": float(np.median(btm[reached_1r])) if reached_1r.any() else float("nan"),
        "by_candidate": {},
    }

    for c in candidates:
        # 이 시점에 끊었을 때 놓치는 MFE 비율
        cut_early = np.array([1.0 if e.bars_to_mfe > c and e.mfe_r >= 1.0 else 0.0 for e in exs])
        r_at_c = np.array([e.final_r_by_bar.get(c, 0.0) for e in exs])
        result["by_candidate"][c] = {
            "avg_r_at_cut": float(r_at_c.mean()),
            "pct_1r_cut_early": float(cut_early.mean()),
        }

    return result


@dataclass
class WalkForwardSplit:
    name: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def make_walk_forward_splits(
    n_bars: int,
    train_bars: int = 500,
    test_bars: int = 250,
    step: int | None = None,
) -> list[WalkForwardSplit]:
    """
    앵커 없는 롤링 walk-forward.

    train 구간에서 파라미터를 고르고 test 구간에서만 성과를 인정한다.
    train과 test를 섞어보고 싶은 유혹이 들면, 그게 과적합의 시작이다.
    """
    step = step or test_bars
    splits, i, k = [], 0, 1
    while i + train_bars + test_bars <= n_bars:
        splits.append(
            WalkForwardSplit(
                name=f"WF{k}",
                train_start=i,
                train_end=i + train_bars - 1,
                test_start=i + train_bars,
                test_end=i + train_bars + test_bars - 1,
            )
        )
        i += step
        k += 1
    return splits


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
            (f"R/봉 최대는 {best}봉, 그 90% 이상을 내는 최단값은 {practical}봉."
             + (" 곡선이 평평해 값 선택의 영향이 작다." if flat else ""))
        ),
    }


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
