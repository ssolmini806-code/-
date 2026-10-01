"""
실험 4 — 온체인 지표가 매수보유를 이기는가

    python exp_04_onchain.py --discover   # 어떤 지표가 무료로 쓸 수 있나 (먼저)
    python exp_04_onchain.py --fetch      # 데이터 받기
    python exp_04_onchain.py --run        # 가설 검증

지금까지와 무엇이 다른가
-------------------------
여섯 번의 시도는 전부 OHLC만 썼다. 이번엔 새 데이터 축이다.

그리고 이번에는 대조군을 처음부터 제대로 잡는다.
지난 실험에서 '무작위 5개'를 기준선으로 삼았다가, 정작 넘어야 할
상대가 '균등 방치'였다는 걸 ETF 검사에서야 알았다. 같은 실수를 반복하지 않는다.

    1차 기준선: 매수보유 (이걸 못 이기면 끝)
    2차 기준선: 무작위 신호 (신호의 순수 기여 확인용)

가설
-----
On1. MVRV 밸류에이션
     MVRV(시총/실현시총)가 역사적으로 높으면 노출을 줄이고 낮으면 늘린다.
     "지금 비싼가"를 묻는 것이지 "언제 오를까"를 묻는 게 아니다.

On2. 네트워크 성장
     활성 주소 증가율이 높은 자산에 비중을 준다. 횡단면.

On3. NVT (가치 대비 사용량)
     NVT가 낮으면(사용량 대비 저평가) 비중을 준다. 횡단면.

On4. 복합
     위 셋의 백분위 평균.

시점 봉인
----------
모든 백분위는 확장 윈도우로 계산한다. 전체 기간 백분위를 쓰면
임계값 자체가 미래 데이터에서 나오게 된다 — 온체인 백테스트의 전형적 오류다.
그리고 온체인 지표는 발표 지연이 있을 수 있으므로 1일 시프트를 추가로 건다.

성공 기준 (사전 확정)
----------------------
매수보유 대비 (a) 총수익 초과 또는 (b) 수익 80% 유지 + MDD 30% 이상 축소.
미달하면 C로 넘어간다. 이번이 마지막이다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

CACHE = "data_cache/onchain.pkl"
COST = 0.0007  # 회전율 1당 비용


# ------------------------------------------------------------------ 유틸


def stats_from_curve(eq: np.ndarray, periods: int = 365) -> dict:
    if len(eq) < 3:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    years = len(eq) / periods
    peak = np.maximum.accumulate(eq)
    r = np.diff(eq) / eq[:-1]
    return {
        "total_return": float(eq[-1] / eq[0] - 1),
        "cagr": float((eq[-1] / eq[0]) ** (1 / years) - 1) if years > 0 else 0.0,
        "max_drawdown": float(np.max(1 - eq / peak)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0,
    }


def run_weights(prices: pd.DataFrame, weights: pd.DataFrame, cost: float = COST) -> np.ndarray:
    """
    비중 시계열을 수익 곡선으로 변환한다.

    weights.iloc[t]는 t 시점 종가 기준 목표 비중이며,
    t→t+1 수익률에 적용된다. weights는 이미 시프트되어 들어와야 한다.
    """
    rets = prices.pct_change().shift(-1)  # t행 = t→t+1 수익률
    common = weights.index.intersection(rets.index)
    w = weights.loc[common].fillna(0.0)
    r = rets.loc[common].fillna(0.0)

    eq = [1.0]
    prev = np.zeros(w.shape[1])
    for i in range(len(common) - 1):
        cur = w.iloc[i].values
        turnover = float(np.abs(cur - prev).sum())
        val = eq[-1] * (1 - turnover * cost)
        port_r = float(np.dot(cur, r.iloc[i].values))
        eq.append(val * (1 + port_r))
        prev = cur
    return np.array(eq)


# ------------------------------------------------------------------ 가설


def build_weights(data: dict, prices: pd.DataFrame, mode: str,
                  shift_days: int = 1) -> pd.DataFrame:
    """
    가설별 비중 산출. 모든 지표는 확장 백분위로 정규화하고 shift로 봉인한다.
    """
    from data.onchain import align_panel, expanding_percentile

    idx, cols = prices.index, prices.columns
    n = len(cols)

    if mode == "equal":
        return pd.DataFrame(1.0 / n, index=idx, columns=cols)

    if mode == "mvrv_exposure":
        # BTC MVRV 백분위로 전체 노출 조절. 나머지는 현금.
        mv = align_panel(data, "CapMVRVCur")
        if "btc" not in mv.columns:
            return pd.DataFrame()
        pct = expanding_percentile(mv["btc"]).shift(shift_days)
        pct = pct.reindex(idx).ffill()
        # 백분위가 높을수록(비쌀수록) 노출을 줄인다. 0.2~1.0 범위
        expo = (1.0 - pct).clip(0.2, 1.0)
        w = pd.DataFrame(0.0, index=idx, columns=cols)
        for c in cols:
            w[c] = expo / n
        return w.fillna(0.0)

    if mode == "cap_per_adr":
        # On5 — 활성주소 대비 시총. 낮을수록 사용자 수 대비 저평가.
        cpa = align_panel(data, "CapPerAdr").reindex(idx).ffill()
        score = (-cpa).rank(axis=1, pct=True).shift(shift_days)
        score = score.reindex(columns=cols)
        w = score.div(score.sum(axis=1), axis=0)
        return w.fillna(1.0 / n)

    if mode in ("adr_growth", "nvt_value", "composite"):
        scores = pd.DataFrame(index=idx, columns=cols, dtype=float)

        if mode in ("adr_growth", "composite"):
            adr = align_panel(data, "AdrActCnt").reindex(idx).ffill()
            growth = adr.pct_change(30)
            g_pct = growth.rank(axis=1, pct=True).shift(shift_days)
        if mode in ("nvt_value", "composite"):
            nvt = align_panel(data, "NVTProxy").reindex(idx).ffill()
            # NVT는 낮을수록 저평가 → 순위 반전
            n_pct = (-nvt).rank(axis=1, pct=True).shift(shift_days)

        if mode == "adr_growth":
            scores = g_pct
        elif mode == "nvt_value":
            scores = n_pct
        else:
            scores = (g_pct.fillna(0.5) + n_pct.fillna(0.5)) / 2

        scores = scores.reindex(columns=cols)
        w = scores.div(scores.sum(axis=1), axis=0)
        return w.fillna(1.0 / n)

    raise ValueError(mode)


def random_weights(prices: pd.DataFrame, seed: int, rebal: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = prices.shape[1]
    w = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    cur = np.ones(n) / n
    for i, t in enumerate(prices.index):
        if i % rebal == 0:
            x = rng.random(n)
            cur = x / x.sum()
        w.loc[t] = cur
    return w


# ------------------------------------------------------------------ 명령


def cmd_discover() -> None:
    from data.onchain import DEFAULT_ASSETS, discover_metrics

    print("=" * 74)
    print("커뮤니티 티어에서 실제로 쓸 수 있는 온체인 지표 조회")
    print("=" * 74)
    avail = discover_metrics(DEFAULT_ASSETS)

    common = None
    for a, ms in avail.items():
        common = set(ms) if common is None else common & set(ms)
    print(f"\n모든 자산 공통 지표: {sorted(common) if common else '없음'}")
    print("\n가설 실행에 필요한 것:")
    for need, why in [("CapMVRVCur", "On1 MVRV"), ("AdrActCnt", "On2 네트워크 성장"),
                      ("NVTAdj", "On3 NVT"), ("PriceUSD", "가격 기준")]:
        have = [a for a, ms in avail.items() if need in ms]
        print(f"  {need:<16} {len(have)}/{len(avail)} 자산  ({why})")


def cmd_fetch() -> None:
    from data.onchain import DEFAULT_ASSETS, load_onchain

    os.makedirs("data_cache", exist_ok=True)
    print("온체인 데이터 다운로드 중...")
    data = load_onchain(DEFAULT_ASSETS, start="2016-01-01")
    if not data:
        raise SystemExit("데이터를 받지 못했습니다.")
    from data.onchain import add_derived_metrics
    data = add_derived_metrics(data)
    print("파생 지표 추가: NVTProxy(시총/트랜잭션수), CapPerAdr(시총/활성주소)")
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"\n저장: {CACHE}")


def cmd_run() -> None:
    from data.onchain import align_panel

    if not os.path.exists(CACHE):
        raise SystemExit("먼저 python exp_04_onchain.py --fetch 를 실행하세요.")
    with open(CACHE, "rb") as f:
        data = pickle.load(f)

    prices = align_panel(data, "PriceUSD").dropna()
    if prices.empty or prices.shape[1] < 3:
        raise SystemExit(f"가격 데이터 부족 (자산 {prices.shape[1]}개)")

    print("=" * 84)
    print(f"온체인 가설 검증  (자산 {prices.shape[1]} · "
          f"{prices.index[0].date()} ~ {prices.index[-1].date()} · {len(prices)}일)")
    print("=" * 84)

    # 벤치마크
    btc = prices["btc"] if "btc" in prices.columns else prices.iloc[:, 0]
    bench_btc = (btc / btc.iloc[0]).values
    eq_hold = (prices / prices.iloc[0]).mean(axis=1).values

    results = {
        "BTC 보유": stats_from_curve(bench_btc),
        "균등 방치": stats_from_curve(eq_hold),
    }

    hypotheses = [
        ("On1 MVRV 노출", "mvrv_exposure"),
        ("On2 주소 성장", "adr_growth"),
        ("On3 NVT대용 저평가", "nvt_value"),
        ("On5 주소당시총", "cap_per_adr"),
        ("On4 복합", "composite"),
        ("(대조) 균등 리밸", "equal"),
    ]
    for label, mode in hypotheses:
        w = build_weights(data, prices, mode)
        if w.empty:
            print(f"  {label}: 지표 없음 — 건너뜀")
            continue
        results[label] = stats_from_curve(run_weights(prices, w))

    # 무작위 대조군
    rnd = [stats_from_curve(run_weights(prices, random_weights(prices, s)))
           for s in range(20)]
    results["(대조) 무작위"] = {
        k: float(np.median([r[k] for r in rnd])) for k in rnd[0]
    }

    hdr = f"  {'전략':<18}{'총수익':>12}{'CAGR':>9}{'MDD':>9}{'Sharpe':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, v in results.items():
        print(f"  {k:<18}{v['total_return']:>+12.1%}{v['cagr']:>+9.1%}"
              f"{v['max_drawdown']:>9.1%}{v['sharpe']:>9.2f}")

    # 판정
    print("\n" + "=" * 84)
    print("판정 — 매수보유(BTC, 균등) 둘 다를 상대로")
    print("=" * 84)
    print("  (a) 총수익 초과  또는  (b) 수익 80% 유지 + MDD 30% 이상 축소\n")

    any_pass = False
    for label, _ in hypotheses:
        if label not in results or label.startswith("(대조)"):
            continue
        s = results[label]
        oks = []
        for bname in ("BTC 보유", "균등 방치"):
            b = results[bname]
            a = s["total_return"] > b["total_return"]
            keeps = (s["total_return"] >= b["total_return"] * 0.8
                     if b["total_return"] > 0 else s["total_return"] >= b["total_return"])
            bc = keeps and s["max_drawdown"] <= b["max_drawdown"] * 0.7
            oks.append(a or bc)
        ok = all(oks)
        any_pass = any_pass or ok
        print(f"  {label:<18} {'통과' if ok else '미달'}"
              f"   (BTC {'O' if oks[0] else 'X'} / 균등 {'O' if oks[1] else 'X'})")

    print()
    if any_pass:
        print("  통과한 가설이 있다. 다음은 기간 분할과 파라미터 민감도 검증이다.")
        print("  아직 채택이 아니다.")
    else:
        print("  전부 미달. 사전 약속대로 능동 탐색을 종료하고 C로 넘어간다.")
        print("  일곱 번째 시도도 같은 답이면, 그건 꽤 단단한 결론이다.")


def main() -> None:
    args = sys.argv[1:]
    if "--discover" in args:
        cmd_discover()
    elif "--fetch" in args:
        cmd_fetch()
    elif "--run" in args or not args:
        cmd_run()


if __name__ == "__main__":
    main()
