"""
1단계 통과 기준 검증: 무작위 진입 스트레스 테스트

목적: "진입 신호가 완전히 무작위여도 계좌가 폭사하지 않는가"

이걸 통과해야 2단계(진입 신호 개발)로 갈 수 있다.
무작위 진입으로 계좌가 반토막 나면, 그건 진입 신호 문제가 아니라
리스크 엔진이 고장난 것이다. 아무리 좋은 신호를 얹어도 소용없다.

기대 결과:
  - 총수익률은 마이너스 (수수료·슬리피지 때문에 당연)
  - 최대 낙폭은 서킷 브레이커 근처에서 멈춤
  - 계좌가 0으로 가지 않음
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from risk.indicators import atr
from risk.rules import RiskConfig, Side, plan_trade
from risk.portfolio import Portfolio


def make_driftless_prices(
    n_symbols: int = 30,
    n_bars: int = 750,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    드리프트를 제거한 순수 널(null) 데이터.

    이것이 진짜 통과 기준이다. 상승 드리프트가 있는 데이터에서는
    추적 손절만으로도 수익이 나기 때문에 엔진 검증이 되지 않는다.
    여기서는 무작위 진입이 '비용만큼만' 잃어야 정상이다.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        vol = rng.uniform(0.010, 0.045)
        shocks = rng.normal(0.0, vol, n_bars)
        shocks -= shocks.mean()          # 표본 드리프트까지 제거
        close = 100.0 * np.exp(np.cumsum(shocks))
        intrabar = np.abs(rng.normal(0, vol * 0.6, n_bars))
        out[f"SYM{i:02d}"] = pd.DataFrame(
            {"high": close * (1 + intrabar), "low": close * (1 - intrabar), "close": close}
        )
    return out


def make_synthetic_prices(
    n_symbols: int = 30,
    n_bars: int = 750,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    기하 브라운 운동 기반 합성 가격.

    합성 데이터를 쓰는 이유: 실제 데이터의 특정 국면(예: 2020~2021 상승장)에
    맞춰 리스크 엔진을 튜닝하는 것을 방지하기 위해서다.
    엔진은 어떤 국면에서도 폭사하지 않아야 한다.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}

    for i in range(n_symbols):
        symbol = f"SYM{i:02d}"
        # 종목마다 변동성을 다르게 → 사이징이 제대로 작동하는지 확인
        daily_vol = rng.uniform(0.010, 0.045)
        drift = rng.uniform(-0.0004, 0.0006)

        shocks = rng.normal(drift, daily_vol, n_bars)
        # 가끔 갭 발생 (손절 미끄러짐 테스트)
        gap_idx = rng.choice(n_bars, size=max(1, n_bars // 120), replace=False)
        shocks[gap_idx] += rng.normal(0, daily_vol * 4, len(gap_idx))

        close = 100.0 * np.exp(np.cumsum(shocks))
        intrabar = np.abs(rng.normal(0, daily_vol * 0.6, n_bars))
        high = close * (1 + intrabar)
        low = close * (1 - intrabar)

        out[symbol] = pd.DataFrame({"high": high, "low": low, "close": close})

    return out


def run_random_entry_test(
    cfg: RiskConfig,
    data: dict[str, pd.DataFrame],
    entry_probability: float = 0.02,
    seed: int = 7,
) -> Portfolio:
    rng = random.Random(seed)
    symbols = list(data.keys())
    n_bars = len(next(iter(data.values())))

    atr_series = {
        s: atr(df["high"], df["low"], df["close"], cfg.atr_period)
        for s, df in data.items()
    }

    pf = Portfolio(initial_equity=100_000.0, cfg=cfg)
    warmup = cfg.atr_period + 5

    for bar in range(warmup, n_bars):
        pf.bar_index = bar

        bars_now = {
            s: {
                "high": float(df["high"].iloc[bar]),
                "low": float(df["low"].iloc[bar]),
                "close": float(df["close"].iloc[bar]),
            }
            for s, df in data.items()
        }
        atr_now = {
            s: float(series.iloc[bar])
            for s, series in atr_series.items()
            if not np.isnan(series.iloc[bar])
        }

        # 1) 보유 포지션 청산 점검
        pf.process_bar(bars_now, atr_now)

        # 2) 무작위 진입 시도
        for symbol in symbols:
            if rng.random() > entry_probability:
                continue
            a = atr_now.get(symbol)
            if not a or a <= 0:
                continue

            price = bars_now[symbol]["close"]
            plan = plan_trade(
                symbol=symbol,
                side=Side.LONG,
                equity=pf.equity,
                entry_price=price,
                atr_value=a,
                cfg=cfg,
                available_cash=pf.cash,
                conviction=1.0,
            )
            pf.open_position(plan, {s: b["close"] for s, b in bars_now.items()})

        # 3) 평가
        pf.mark_to_market({s: b["close"] for s, b in bars_now.items()})

    return pf


def main() -> None:
    cfg = RiskConfig()

    print("=" * 66)
    print("1단계 검증 — 리스크 엔진 스트레스 테스트")
    print("=" * 66)
    print(f"1회 위험 {cfg.risk_per_trade:.1%} · 손절 {cfg.atr_stop_multiple}xATR · "
          f"손익비 {cfg.reward_risk_ratio} · 시간손절 {cfg.time_stop_bars}봉 · "
          f"서킷 {cfg.max_drawdown_halt:.0%}")

    scenarios = {
        "드리프트 없음 (널 테스트)": make_driftless_prices(),
        "드리프트 있음 (참고용)": make_synthetic_prices(),
    }

    summary = {}
    for label, data in scenarios.items():
        print("-" * 66)
        print(f"[{label}]  종목 {len(data)} · 봉 {len(next(iter(data.values())))}")
        rows = []
        for seed in range(1, 7):
            pf = run_random_entry_test(cfg, data, seed=seed)
            st = pf.stats()
            rows.append(st)
            print(f"  seed {seed}: 거래 {st['trades']:>4} · 평균 {st['avg_r']:>+5.2f}R · "
                  f"수익 {st['total_return']:>+7.2%} · MDD {st['max_drawdown']:>6.2%} · "
                  f"최종 {st['final_equity']:>9,.0f}")
        summary[label] = {
            "avg_r": sum(r["avg_r"] for r in rows) / len(rows),
            "ret": sum(r["total_return"] for r in rows) / len(rows),
            "worst_dd": max(r["max_drawdown"] for r in rows),
            "worst_eq": min(r["final_equity"] for r in rows),
        }

    null = summary["드리프트 없음 (널 테스트)"]
    print("=" * 66)
    print("통과 기준 (널 테스트 기준)")
    checks = [
        ("계좌가 폭사하지 않음 (최종자본 > 초기의 70%)", null["worst_eq"] > 70_000),
        ("최대 낙폭이 서킷 근처에서 통제됨 (< 15%)", null["worst_dd"] < 0.15),
        ("무작위 진입의 기댓값이 음수 (비용만큼만 손실)", -0.20 < null["avg_r"] < 0.0),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print()
    print("2단계 진입 신호의 비교 기준선(baseline):")
    print(f"  드리프트 없음: 평균 {null['avg_r']:+.3f}R / 수익 {null['ret']:+.2%}")
    d = summary["드리프트 있음 (참고용)"]
    print(f"  드리프트 있음: 평균 {d['avg_r']:+.3f}R / 수익 {d['ret']:+.2%}")
    print()
    print("  ※ 진입 신호는 '0'이 아니라 위 기준선을 이겨야 한다.")
    print("     추적 손절만으로도 추세장에서는 수익이 나기 때문이다.")


if __name__ == "__main__":
    main()
