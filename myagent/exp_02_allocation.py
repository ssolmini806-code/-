"""
실험 2 — 항상 투자 상태에서 신호를 배분에 쓰기 (H1 + H2)

    python exp_02_allocation.py --crypto
    python exp_02_allocation.py --stocks
    python exp_02_allocation.py --all

성공 기준 (사전 확정)
----------------------
매수보유 대비 다음 중 하나:
  (a) 총수익이 더 높다
  (b) 총수익이 비슷하면서(80% 이상) 최대낙폭이 30% 이상 작다

가설
-----
H1 tilt      균등가중을 기본으로 두고 신호 쪽으로 절반만 기울인다
H2 xs_mom    횡단면 모멘텀 — 과거 수익률 상위 5개 보유
H2 xs_mr     횡단면 평균회귀 — 가장 과매도된 5개 보유
H3 inv_vol   변동성 역가중 — 방향 예측 없음
H4 eq_rebal  균등가중 + 주기적 리밸런싱

대조군
-------
BTC(또는 SPY) 매수보유, 균등가중 방치

사전 약속
----------
이번 한 번만 돌린다. 파라미터(top_n, lookback, tilt, 리밸런싱 주기)를
바꿔가며 재시도하지 않는다. 미달하면 C로 넘어간다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

from allocation import (
    buy_and_hold,
    equal_weight_hold,
    run_allocation,
    w_equal,
    w_inverse_vol,
    w_tilt_meanrev,
    w_xs_meanrev,
    w_xs_momentum,
)

STRATEGIES = [
    ("H4 eq_rebal", w_equal),
    ("H3 inv_vol", w_inverse_vol),
    ("H2 xs_mom", w_xs_momentum),
    ("H2 xs_mr", w_xs_meanrev),
    ("H1 tilt", w_tilt_meanrev),
]


def evaluate(panel, label: str, bench_symbol: str | None, cost_rate: float,
             bench_is_equal: bool = False) -> None:
    print("\n" + "=" * 84)
    print(f"{label}  (종목 {len(panel.symbols)} · 봉 {panel.n_bars} · 회전비용 {cost_rate:.2%})")
    print("=" * 84)

    warmup = 120
    single = buy_and_hold(panel, bench_symbol, start_bar=warmup)
    eqhold = equal_weight_hold(panel, start_bar=warmup)
    # 벤치마크 선택: 크립토는 BTC 보유, 주식은 균등가중 보유
    results = [eqhold, single] if bench_is_equal else [single, eqhold]
    for name, fn in STRATEGIES:
        results.append(
            run_allocation(panel, fn, name, rebalance_every=5,
                           cost_rate=cost_rate, warmup=warmup)
        )

    hdr = f"  {'전략':<16}{'총수익':>11}{'CAGR':>9}{'MDD':>9}{'Sharpe':>9}{'Calmar':>9}{'회전율':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    stats = {}
    for r in results:
        s = r.stats()
        stats[r.name] = s
        print(f"  {r.name:<16}{s['total_return']:>+11.1%}{s['cagr']:>+9.1%}"
              f"{s['max_drawdown']:>9.1%}{s['sharpe']:>9.2f}{s['calmar']:>9.2f}"
              f"{s['turnover']:>9.2f}")

    # 판정
    bench_name = results[0].name
    b = stats[bench_name]
    print(f"\n  판정 기준: {bench_name} 대비")
    print(f"    (a) 총수익 초과  또는  (b) 수익 80% 이상 유지 + MDD 30% 이상 축소\n")

    any_pass = False
    for name, _ in STRATEGIES:
        s = stats[name]
        a = s["total_return"] > b["total_return"]
        keeps = (
            s["total_return"] >= b["total_return"] * 0.8
            if b["total_return"] > 0
            else s["total_return"] >= b["total_return"]
        )
        dd_better = s["max_drawdown"] <= b["max_drawdown"] * 0.7
        bcond = keeps and dd_better
        ok = a or bcond
        any_pass = any_pass or ok
        tag = "통과" if ok else "미달"
        why = "(a) 수익 우위" if a else ("(b) 낙폭 축소" if bcond else "")
        print(f"    {name:<16} {tag}  {why}")

    if not any_pass:
        print("\n    → 전부 미달")


def load_crypto():
    path = "data_cache/crypto_1d.pkl"
    if not os.path.exists(path):
        raise SystemExit("먼저 python exp_01_daily.py --fetch 를 실행하세요.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_stocks():
    from run_real import DEFAULT_TICKERS, load_panel

    return load_panel(DEFAULT_TICKERS, "2015-01-01", "2026-09-01")


def main() -> None:
    args = sys.argv[1:]
    do_all = not args or "--all" in args

    if do_all or "--crypto" in args:
        # 크립토: 회전비용 0.07% (maker 0.02% + 슬리피지 0.05%)
        evaluate(load_crypto(), "크립토 일봉", None, cost_rate=0.0007)

    if do_all or "--stocks" in args:
        # 미국 주식: 수수료 0 + 슬리피지 0.05%
        evaluate(load_stocks(), "미국 주식 일봉", None, cost_rate=0.0005, bench_is_equal=True)

    print("\n" + "=" * 84)
    print("해석 주의")
    print("=" * 84)
    print("  1. 두 유니버스 모두 생존 편향이 있다. 사라진 코인·종목이 빠져 있다.")
    print("  2. 횡단면 전략은 특히 편향에 취약하다 — '결국 오른 것'을 고르게 되므로.")
    print("  3. 파라미터를 바꿔 재시도하지 않는다. 사전 약속이다.")


if __name__ == "__main__":
    main()
