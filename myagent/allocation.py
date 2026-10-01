"""
비중 배분 백테스터.

지금까지의 엔진과 근본적으로 다르다.

    기존: 진입/청산 → 대부분의 시간을 현금으로 보냄
    여기: 항상 100% 투자 → 신호는 '무엇을 얼마나'만 결정

이렇게 바꾸는 이유
-------------------
크립토 일봉 결과에서 meanrev는 무작위를 이겼다(+0.017R vs -0.067R).
신호에 뭔가 있다는 뜻이다. 그런데 총수익은 +0.11%, BTC 보유는 +39.26%였다.

실패 원인은 신호가 아니라 '시장 밖에 있던 시간'이다.
그렇다면 시장 밖에 나가지 않으면서 신호를 쓰면 된다.

시점 봉인
----------
t 시점 비중은 t까지의 데이터로만 계산하고, t→t+1 수익률에 적용한다.
t+1 데이터로 t 비중을 정하면 백테스트가 통째로 거짓말이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.contract import PricePanel


@dataclass
class AllocResult:
    name: str
    equity_curve: np.ndarray
    weights_history: list[np.ndarray]
    turnover: list[float]

    @property
    def total_return(self) -> float:
        return float(self.equity_curve[-1] / self.equity_curve[0] - 1)

    @property
    def cagr(self) -> float:
        years = len(self.equity_curve) / 252
        if years <= 0 or self.equity_curve[0] <= 0:
            return 0.0
        return float((self.equity_curve[-1] / self.equity_curve[0]) ** (1 / years) - 1)

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity_curve)
        return float(np.max(1 - self.equity_curve / peak))

    @property
    def sharpe(self) -> float:
        r = np.diff(self.equity_curve) / self.equity_curve[:-1]
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover)) if self.turnover else 0.0

    def stats(self) -> dict:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "turnover": self.avg_turnover,
            "calmar": self.cagr / self.max_drawdown if self.max_drawdown > 0 else 0.0,
        }


# --------------------------------------------------------------- 비중 산출기
# 모든 함수는 (returns_history, prices_history) → 비중 벡터.
# 입력은 t 시점까지만 담긴 배열이다. 미래를 볼 방법이 구조적으로 없다.


def w_equal(hist: pd.DataFrame) -> np.ndarray:
    n = hist.shape[1]
    return np.ones(n) / n


def w_inverse_vol(hist: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """H3 — 변동성 역가중. 방향 예측이 전혀 없다."""
    vol = hist.pct_change().iloc[-lookback:].std().values
    vol = np.where((vol > 0) & np.isfinite(vol), vol, np.nanmedian(vol[vol > 0]) if (vol > 0).any() else 1.0)
    inv = 1.0 / vol
    return inv / inv.sum()


def w_xs_momentum(hist: pd.DataFrame, lookback: int = 90, top_n: int = 5,
                  skip: int = 5) -> np.ndarray:
    """
    H2 — 횡단면 모멘텀. 과거 수익률 상위 N개를 균등 보유.

    skip: 최근 며칠을 제외한다. 단기 반전 효과와 섞이는 것을 막기 위한
          표준적인 처리다(1개월 반전).
    """
    n = hist.shape[1]
    if len(hist) < lookback + skip:
        return w_equal(hist)
    past = hist.iloc[-(lookback + skip)]
    recent = hist.iloc[-(skip + 1)]
    ret = (recent / past - 1).values
    ret = np.where(np.isfinite(ret), ret, -np.inf)
    idx = np.argsort(ret)[::-1][:top_n]
    w = np.zeros(n)
    w[idx] = 1.0 / len(idx)
    return w


def w_xs_meanrev(hist: pd.DataFrame, rsi_period: int = 3, top_n: int = 5,
                 trend_ma: int = 100) -> np.ndarray:
    """
    H2 — 횡단면 평균회귀. 가장 과매도된 N개를 보유.

    추세 필터를 통과한 자산 중에서만 고른다.
    통과 자산이 부족하면 균등가중으로 물러난다(현금으로 빠지지 않는다).
    """
    n = hist.shape[1]
    if len(hist) < trend_ma + rsi_period + 5:
        return w_equal(hist)

    scores = np.full(n, np.inf)
    ma = hist.rolling(trend_ma).mean().iloc[-1]
    for i, col in enumerate(hist.columns):
        s = hist[col]
        if float(s.iloc[-1]) <= float(ma[col]):
            continue
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean().iloc[-1]
        rs = gain / loss if loss > 0 else np.inf
        scores[i] = 100 - 100 / (1 + rs)

    eligible = np.isfinite(scores)
    if eligible.sum() < 2:
        return w_equal(hist)
    idx = np.argsort(scores)[: min(top_n, int(eligible.sum()))]
    w = np.zeros(n)
    w[idx] = 1.0 / len(idx)
    return w


def w_tilt_meanrev(hist: pd.DataFrame, rsi_period: int = 3, trend_ma: int = 100,
                   tilt: float = 0.5) -> np.ndarray:
    """
    H1 — 균등가중을 기본으로 두고 신호 쪽으로 기울인다.

    tilt=0이면 순수 균등가중, tilt=1이면 신호에 100% 의존.
    0.5는 절반만 기울이는 것. 신호가 틀려도 절반은 시장에 남아 있다.
    """
    base = w_equal(hist)
    signal = w_xs_meanrev(hist, rsi_period, top_n=5, trend_ma=trend_ma)
    w = (1 - tilt) * base + tilt * signal
    return w / w.sum()


# --------------------------------------------------------------- 백테스트


def run_allocation(
    panel: PricePanel,
    weight_fn,
    name: str,
    *,
    rebalance_every: int = 5,
    cost_rate: float = 0.0007,
    start_bar: int | None = None,
    end_bar: int | None = None,
    warmup: int = 120,
) -> AllocResult:
    """
    rebalance_every: 며칠마다 비중을 다시 맞출지. 잦을수록 비용이 커진다.
    cost_rate: 회전율 1단위당 비용(편도 수수료+슬리피지).
    """
    closes = pd.DataFrame({s: panel.frames[s]["close"] for s in panel.symbols})
    start = max(warmup, start_bar or 0)
    end = min(len(closes) - 1, end_bar if end_bar is not None else len(closes) - 1)

    n = closes.shape[1]
    w = np.ones(n) / n
    equity = [1.0]
    hist_w, turnovers = [], []

    for t in range(start, end):
        # t 시점 비중 결정 — t까지의 데이터만 사용
        if (t - start) % rebalance_every == 0:
            target = weight_fn(closes.iloc[: t + 1])
            target = np.nan_to_num(target, nan=0.0)
            if target.sum() <= 0:
                target = np.ones(n) / n
            target = target / target.sum()
            to = float(np.abs(target - w).sum())
            turnovers.append(to)
            equity[-1] *= 1 - to * cost_rate
            w = target

        # t → t+1 수익률 적용
        r = (closes.iloc[t + 1] / closes.iloc[t] - 1).values
        r = np.nan_to_num(r, nan=0.0)
        port_r = float(np.dot(w, r))
        equity.append(equity[-1] * (1 + port_r))

        # 가격 변동에 따른 비중 드리프트 반영
        grown = w * (1 + r)
        w = grown / grown.sum() if grown.sum() > 0 else np.ones(n) / n
        hist_w.append(w.copy())

    return AllocResult(name, np.array(equity), hist_w, turnovers)


def buy_and_hold(panel: PricePanel, symbol: str | None = None,
                 start_bar: int = 120, end_bar: int | None = None) -> AllocResult:
    """단일 자산 매수보유 (기본은 BTC). 리밸런싱도 비용도 없다."""
    closes = pd.DataFrame({s: panel.frames[s]["close"] for s in panel.symbols})
    end = min(len(closes) - 1, end_bar if end_bar is not None else len(closes) - 1)
    sym = symbol or next((s for s in panel.symbols if s.startswith("BTC")), panel.symbols[0])
    px = closes[sym].iloc[start_bar : end + 1].values
    return AllocResult(f"{sym} 보유", px / px[0], [], [])


def equal_weight_hold(panel: PricePanel, start_bar: int = 120,
                      end_bar: int | None = None) -> AllocResult:
    """균등가중 후 방치. 리밸런싱 없음 — H4의 대조군."""
    closes = pd.DataFrame({s: panel.frames[s]["close"] for s in panel.symbols})
    end = min(len(closes) - 1, end_bar if end_bar is not None else len(closes) - 1)
    px = closes.iloc[start_bar : end + 1]
    norm = px / px.iloc[0]
    return AllocResult("균등 방치", norm.mean(axis=1).values, [], [])
