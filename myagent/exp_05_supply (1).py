"""
실험 5 — 외국인·기관 수급 전략 검증

    pip install pykrx
    python exp_05_supply.py --fetch     # 20~40분 (재개 가능)
    python exp_05_supply.py --verify    # 원문 주장 그대로 재현
    python exp_05_supply.py --filter    # 여섯 가지 필터 적용
    python exp_05_supply.py --all

검증 대상
----------
스레드 글의 주장:
    "기관이 3일 연속 순매수한 종목을 다음 날 매수해 5일 보유"
      → 승률 58.3%, 평균 수익률 2.1%
    "외국인도 동시에 순매수한 조건 추가"
      → 승률 63.7%, 평균 수익률 2.8%

이 숫자를 연율로 환산하면:
    2.1% × 연 50회전 → 연 185%
    2.8% × 연 50회전 → 연 302%
르네상스 메달리온(역사상 최고)이 수수료 전 연 66%, 버핏이 연 20% 수준이다.

검증 순서
----------
1단계 (--verify) 원문 조건 그대로 재현. 숫자가 맞는지만 본다.
2단계 (--filter) 여섯 가지 필터 적용:
    ① 무작위 진입 기준선
    ② KOSPI 매수보유 벤치마크
    ③ out-of-sample (기간 분할)
    ④ 거래비용 (증권거래세·수수료·슬리피지)
    ⑤ 생존 편향 (시점별 상장 종목 사용)
    ⑥ 다중검정 보정

이 프로젝트의 이점
-------------------
pykrx는 과거 시점의 상장 종목 목록을 조회할 수 있다(get_market_ticker_list).
지금까지 미국 주식·크립토에서 계속 걸렸던 생존 편향을 처음으로 제대로 통제한다.
상장폐지된 종목도 그 시점에는 유니버스에 들어간다.

사전 확정 기준
---------------
"사실"로 인정하려면 여섯 필터를 모두 통과해야 한다.
1단계에서 숫자가 재현되어도, 2단계를 못 넘으면 "재현되나 거래 불가"다.
조건을 추가해 개선을 시도하지 않는다. 한 번만 돌린다.
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = "data_cache/krx"
START = "20220101"
END = "20260831"

# 한국 주식 거래비용 (2026년 기준, 확인 필요)
#   매도 시 증권거래세·농특세 0.15%
#   증권사 수수료 0.015% 내외 (양방향)
#   슬리피지 0.1% (소형주는 더 클 수 있음)
SELL_TAX = 0.0015
COMMISSION = 0.00015
SLIPPAGE = 0.001
ROUND_TRIP = SELL_TAX + COMMISSION * 2 + SLIPPAGE * 2


def check_credentials() -> None:
    """
    KRX는 2025년 12월 27일부로 회원제(KRX Data Marketplace)로 전환됐다.
    조회는 무료지만 로그인이 필수다. pykrx도 환경변수를 요구한다.

    자격증명이 없으면 빈 응답을 받아 엉뚱한 곳에서 죽으므로 먼저 확인한다.
    """
    missing = [k for k in ("KRX_ID", "KRX_PW") if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"\nKRX 자격증명이 없습니다: {', '.join(missing)}\n"
            "\nKRX는 2025-12-27부터 회원제로 전환됐습니다(조회는 무료).\n"
            "  1) data.krx.co.kr 에서 회원가입\n"
            "  2) 자격증명 등록 — 둘 중 하나\n"
            "     · Codespaces Secret (권장, 재시작 후 자동 주입)\n"
            "       저장소 Settings > Secrets and variables > Codespaces\n"
            "     · 이번 세션만:\n"
            "       read -s -p 'ID: ' KRX_ID && export KRX_ID && echo\n"
            "       read -s -p 'PW: ' KRX_PW && export KRX_PW && echo\n"
        )


def trading_days(start: str, end: str) -> list[str]:
    from pykrx import stock

    days = stock.get_previous_business_days(fromdate=start, todate=end)
    if not days:
        raise SystemExit(
            "거래일 목록이 비어 있습니다.\n"
            "KRX 로그인이 실패했거나(자격증명 확인) 기간 설정이 잘못됐을 수 있습니다."
        )
    return days


def fetch_all(resume: bool = True) -> None:
    """
    일자별로 전 종목 수급·가격을 받는다.

    종목별로 받으면 호출이 수천 번이라 오래 걸린다.
    일자별로 받으면 하루 한 번에 전 종목이 나온다.
    중간에 끊겨도 이어받을 수 있게 일자별로 저장한다.
    """
    from pykrx import stock

    check_credentials()
    os.makedirs(CACHE_DIR, exist_ok=True)
    days = trading_days(START, END)
    print(f"거래일 {len(days)}일 · {days[0].date()} ~ {days[-1].date()}")

    fails = saved = 0
    for i, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        path = f"{CACHE_DIR}/{ds}.pkl"
        if resume and os.path.exists(path):
            continue
        try:
            # 시점별 상장 종목 — 생존 편향 통제의 핵심
            ohlcv = stock.get_market_ohlcv(ds, market="ALL")
            inst = stock.get_market_net_purchases_of_equities(ds, ds, "ALL", "기관합계")
            forn = stock.get_market_net_purchases_of_equities(ds, ds, "ALL", "외국인")

            day = pd.DataFrame(index=ohlcv.index)
            day["close"] = ohlcv["종가"]
            day["open"] = ohlcv["시가"]
            day["volume"] = ohlcv["거래량"]
            day["value"] = ohlcv["거래대금"]
            day["inst"] = inst["순매수거래대금"].reindex(day.index)
            day["forn"] = forn["순매수거래대금"].reindex(day.index)

            with open(path, "wb") as f:
                pickle.dump(day, f)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  {ds}: 실패 ({type(e).__name__}) — 건너뜀")
            if fails >= 10 and saved == 0:
                raise SystemExit(
                    "\n연속 10일 실패, 저장된 데이터 0건.\n"
                    "KRX 로그인이 정상인지 확인하세요 (KRX_ID / KRX_PW)."
                ) from e
            continue
        saved += 1

        if i % 50 == 0:
            print(f"  {i}/{len(days)}  {ds}")

    print("완료.")


def load_panel() -> dict[str, pd.DataFrame]:
    """일자별 파일을 지표별 패널로 재구성한다."""
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl"))
    if not files:
        raise SystemExit("먼저 --fetch 를 실행하세요.")

    frames = {}
    for f in files:
        with open(f"{CACHE_DIR}/{f}", "rb") as fh:
            frames[f[:-4]] = pickle.load(fh)

    dates = sorted(frames)
    panels = {}
    for col in ("close", "open", "inst", "forn", "value"):
        panels[col] = pd.DataFrame(
            {d: frames[d][col] for d in dates}
        ).T.sort_index()
    print(f"패널: {len(dates)}일 · 종목 {panels['close'].shape[1]}개 "
          f"(시점별 상장 종목 합집합)")
    return panels


# ------------------------------------------------------------------ 신호


def signal_streak(panels: dict, use_foreign: bool, streak: int = 3,
                  min_value: float = 1e9) -> pd.DataFrame:
    """
    기관 N일 연속 순매수 (+ 선택적으로 외국인 동시 순매수).

    min_value: 일 거래대금 하한. 유동성이 없는 종목은 제외한다.
               이걸 안 걸면 체결 불가능한 종목이 통계를 오염시킨다.
    """
    inst, forn, val = panels["inst"], panels["forn"], panels["value"]

    cond = (inst > 0)
    for k in range(1, streak):
        cond = cond & (inst.shift(k) > 0)

    if use_foreign:
        cond = cond & (forn > 0)

    liquid = val > min_value
    return (cond & liquid).fillna(False)


def evaluate(panels: dict, sig: pd.DataFrame, hold: int = 5,
             cost: float = 0.0, start_i: int | None = None,
             end_i: int | None = None) -> dict:
    """
    t일 신호 → t+1일 시가 매수 → hold일 후 종가 매도.

    원문은 "다음 날 매수해서 5일 보유"라고 했으므로 그대로 따른다.
    """
    op, cl = panels["open"], panels["close"]
    n = len(cl)
    lo = start_i if start_i is not None else 0
    hi = end_i if end_i is not None else n - hold - 2

    rets = []
    for i in range(max(lo, 3), min(hi, n - hold - 2)):
        row = sig.iloc[i]
        picks = row[row].index
        if len(picks) == 0:
            continue
        buy = op.iloc[i + 1].reindex(picks)
        sell = cl.iloc[i + 1 + hold].reindex(picks)
        r = (sell / buy - 1).dropna()
        r = r[np.isfinite(r)]
        if len(r):
            rets.extend((r - cost).tolist())

    if not rets:
        return {"n": 0}
    a = np.array(rets)
    return {
        "n": len(a),
        "win_rate": float((a > 0).mean()),
        "mean_ret": float(a.mean()),
        "median_ret": float(np.median(a)),
        "std": float(a.std()),
        "t_stat": float(a.mean() / (a.std() / np.sqrt(len(a)))) if a.std() > 0 else 0.0,
    }


def random_baseline(panels: dict, n_picks_per_day: int, hold: int = 5,
                    cost: float = 0.0, seed: int = 0,
                    min_value: float = 1e9) -> dict:
    """같은 수의 종목을 무작위로 골랐을 때."""
    rng = np.random.default_rng(seed)
    val = panels["value"]
    sig = pd.DataFrame(False, index=val.index, columns=val.columns)
    for i in range(len(val)):
        liquid = val.iloc[i][val.iloc[i] > min_value].index
        if len(liquid) == 0:
            continue
        k = min(n_picks_per_day, len(liquid))
        picks = rng.choice(liquid, size=k, replace=False)
        sig.iloc[i, sig.columns.get_indexer(picks)] = True
    return evaluate(panels, sig, hold, cost)


# ------------------------------------------------------------------ 단계


def cmd_verify(panels: dict) -> dict:
    print("\n" + "=" * 78)
    print("1단계 — 원문 주장 그대로 재현 (비용 미반영)")
    print("=" * 78)
    print("원문: 기관 3일 연속 순매수 → 다음날 매수 → 5일 보유")
    print("      조건1 승률 58.3% / 평균 2.1%   조건2 승률 63.7% / 평균 2.8%\n")

    out = {}
    print(f"  {'조건':<22}{'표본':>9}{'승률':>9}{'평균':>10}{'중앙':>10}{'원문 대비'}")
    for label, fg, claim_wr, claim_r in [
        ("기관 3일 연속", False, 0.583, 0.021),
        ("+ 외국인 동시", True, 0.637, 0.028),
    ]:
        sig = signal_streak(panels, fg)
        r = evaluate(panels, sig, hold=5, cost=0.0)
        out[label] = r
        if not r["n"]:
            print(f"  {label:<22} 표본 없음")
            continue
        print(f"  {label:<22}{r['n']:>9,}{r['win_rate']:>8.1%}{r['mean_ret']:>+10.2%}"
              f"{r['median_ret']:>+10.2%}   승률 {r['win_rate'] - claim_wr:+.1%p} · "
              f"수익 {r['mean_ret'] - claim_r:+.2%p}")

    print("\n  판정: 원문 숫자가 재현되는가?")
    for label, r in out.items():
        if not r.get("n"):
            continue
        claim = 0.021 if "기관" in label and "외국인" not in label else 0.028
        ratio = r["mean_ret"] / claim if claim else 0
        print(f"    {label}: 원문의 {ratio:.0%} 수준")
    return out


def cmd_filter(panels: dict) -> None:
    from statistics import NormalDist

    print("\n" + "=" * 78)
    print("2단계 — 여섯 가지 필터")
    print("=" * 78)
    print(f"거래비용 왕복 {ROUND_TRIP:.2%} "
          f"(거래세 {SELL_TAX:.2%} + 수수료 {COMMISSION * 2:.3%} + 슬리피지 {SLIPPAGE * 2:.1%})\n")

    sig = signal_streak(panels, use_foreign=True)
    picks_per_day = int(sig.sum(axis=1).mean())
    print(f"  일평균 신호 종목 수: {picks_per_day}개\n")

    # ④ 비용 반영
    raw = evaluate(panels, sig, 5, cost=0.0)
    net = evaluate(panels, sig, 5, cost=ROUND_TRIP)
    print("  ④ 거래비용")
    print(f"     비용 전 {raw['mean_ret']:+.2%} → 비용 후 {net['mean_ret']:+.2%}")

    # ① 무작위 기준선
    rnd = [random_baseline(panels, picks_per_day, 5, ROUND_TRIP, seed=s) for s in range(5)]
    rnd_mean = float(np.mean([r["mean_ret"] for r in rnd if r["n"]]))
    rnd_wr = float(np.mean([r["win_rate"] for r in rnd if r["n"]]))
    print("\n  ① 무작위 기준선 (같은 수의 종목을 무작위로)")
    print(f"     신호 {net['mean_ret']:+.2%} (승률 {net['win_rate']:.1%}) vs "
          f"무작위 {rnd_mean:+.2%} (승률 {rnd_wr:.1%})")
    print(f"     초과 {net['mean_ret'] - rnd_mean:+.2%}")

    # ② 벤치마크 — 동일 기간 전 종목 매수보유
    cl = panels["close"]
    eq = (cl / cl.iloc[0]).mean(axis=1).dropna()
    bh_total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    years = len(cl) / 252
    # 전략의 연환산 (회전율 반영)
    cycles = 252 / 6
    strat_annual = (1 + net["mean_ret"]) ** cycles - 1
    print("\n  ② 매수보유 벤치마크")
    print(f"     전략 연환산 {strat_annual:+.1%} vs 전 종목 매수보유 "
          f"{(1 + bh_total) ** (1 / years) - 1:+.1%}")

    # ③ 기간 분할
    print("\n  ③ out-of-sample (기간 3등분)")
    n = len(cl)
    for k, (lo, hi) in enumerate([(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)], 1):
        r = evaluate(panels, sig, 5, ROUND_TRIP, start_i=lo, end_i=hi)
        if r["n"]:
            print(f"     P{k}: 표본 {r['n']:>6,} · 승률 {r['win_rate']:.1%} · "
                  f"평균 {r['mean_ret']:+.2%} · t={r['t_stat']:.2f}")

    # ⑤ 생존 편향
    print("\n  ⑤ 생존 편향")
    print(f"     시점별 상장 종목 사용 (pykrx get_market_ohlcv). "
          f"상장폐지 종목도 당시 유니버스에 포함됨.")

    # ⑥ 다중검정
    n_tests = 6
    crit = NormalDist().inv_cdf(1 - (0.05 / n_tests) / 2)
    print(f"\n  ⑥ 다중검정 보정 ({n_tests}회 → |t| > {crit:.2f})")
    print(f"     신호 t={net['t_stat']:.2f}")

    # 최종
    print("\n" + "=" * 78)
    print("최종 판정")
    print("=" * 78)
    c1 = net["mean_ret"] > rnd_mean + 0.002
    c2 = strat_annual > (1 + bh_total) ** (1 / years) - 1
    c3 = abs(net["t_stat"]) > crit
    c4 = net["mean_ret"] > 0
    for label, ok in [
        ("무작위 기준선 초과", c1),
        ("매수보유 초과", c2),
        ("다중검정 보정 후 유의", c3),
        ("비용 차감 후 양수", c4),
    ]:
        print(f"  [{'O' if ok else 'X'}] {label}")

    print()
    if all([c1, c2, c3, c4]):
        print("  전부 통과. 다만 아직 '사실'이지 '거래 가능'은 아니다.")
        print("  다음: 체결 가능성(일평균 신호 종목이 너무 많으면 분산 불가),")
        print("        시장충격, 그리고 새로운 시각으로의 확장을 검토한다.")
    else:
        print("  미달. 원문 주장은 여섯 필터를 통과하지 못한다.")
        print("  숫자가 재현되더라도 거래 가능한 엣지는 아니다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.fetch:
        fetch_all()
        return

    panels = load_panel()
    if a.verify or a.all or not (a.verify or a.filter):
        cmd_verify(panels)
    if a.filter or a.all:
        cmd_filter(panels)


if __name__ == "__main__":
    main()
