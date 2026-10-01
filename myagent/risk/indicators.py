"""
결정론적 지표 계산.

이 모듈에는 LLM이 절대 개입하지 않는다.
모든 함수는 같은 입력에 같은 출력을 반환해야 한다(재현성).
"""

from __future__ import annotations

import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range = max(고가-저가, |고가-전일종가|, |저가-전일종가|)

    전일 종가를 쓰기 때문에 갭(gap)을 변동성에 반영한다.
    단순 (고가-저가)만 쓰면 갭 하락 종목의 위험을 과소평가하게 된다.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (Wilder 방식 지수평활).

    주의: 이 함수는 t 시점 값을 계산할 때 t 시점 데이터까지만 사용한다.
    호출하는 쪽에서 as_of 이후 데이터를 잘라서 넘겨야 룩어헤드가 없다.
    """
    if period < 1:
        raise ValueError("period는 1 이상이어야 합니다")

    tr = true_range(high, low, close)
    # Wilder smoothing == alpha = 1/period 인 EMA
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """일간 로그수익률의 표준편차(연율화하지 않은 값)."""
    import numpy as np

    log_ret = (close / close.shift(1)).apply(lambda x: np.log(x) if x > 0 else float("nan"))
    return log_ret.rolling(period).std()
