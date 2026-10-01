"""
진입 신호 (AlphaModel).

인터페이스 계약:
  - generate()는 방향과 확신도(0~1)만 반환한다.
  - 손절가, 목표가, 수량은 절대 반환하지 않는다. 그건 리스크 엔진의 일이다.
  - as_of 이후 데이터에 접근하지 않는다(PricePanel이 강제한다).

3단계에서 LLM 필터를 추가할 때도 같은 인터페이스를 따른다.
LLM은 새 신호를 만드는 게 아니라 기존 후보를 거르는 역할이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.contract import PricePanel
from risk.rules import Side


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    conviction: float          # 0.0 ~ 1.0. 리스크 엔진이 위험금액을 '줄이는' 데만 쓴다
    reason: str                # 로그·사후 진단용

    def __post_init__(self) -> None:
        if not 0.0 <= self.conviction <= 1.0:
            raise ValueError("conviction은 0~1 사이여야 합니다")


class AlphaModel(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, panel: PricePanel, as_of: int) -> list[Signal]:
        """as_of 봉 종가 기준 판단. 진입은 다음 봉 시가에 이루어진다."""

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """지표 계산에 필요한 최소 봉 수."""


class RandomEntry(AlphaModel):
    """
    기준선(baseline). 반드시 이겨야 하는 대상.

    1단계에서 확인했듯 추적 손절만으로도 추세장에서는 수익이 난다.
    진입 신호의 성과를 0이 아니라 이 모델과 비교해야
    출구 로직이 만든 수익을 신호의 공로로 착각하지 않는다.
    """

    name = "random"

    def __init__(self, probability: float = 0.02, seed: int = 0) -> None:
        self.probability = probability
        self._rng = np.random.default_rng(seed)

    @property
    def warmup_bars(self) -> int:
        return 20

    def generate(self, panel: PricePanel, as_of: int) -> list[Signal]:
        out = []
        for symbol in panel.symbols:
            if self._rng.random() < self.probability:
                out.append(Signal(symbol, Side.LONG, 1.0, "random"))
        return out


class MomentumBreakout(AlphaModel):
    """
    추세추종: N일 신고가 돌파 + 장기 추세 정배열.

    가장 오래되고 가장 많이 검증된 유형이다.
    화려하지 않지만 기준선 역할을 하기에 적합하다.
    """

    name = "momentum"

    def __init__(
        self,
        breakout_lookback: int = 55,
        fast_ma: int = 20,
        slow_ma: int = 100,
        min_slope_bars: int = 20,
    ) -> None:
        self.breakout_lookback = breakout_lookback
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.min_slope_bars = min_slope_bars

    @property
    def warmup_bars(self) -> int:
        return max(self.slow_ma, self.breakout_lookback) + self.min_slope_bars + 5

    def generate(self, panel: PricePanel, as_of: int) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in panel.symbols:
            h = panel.history(symbol, as_of, lookback=self.warmup_bars)
            if len(h) < self.warmup_bars:
                continue

            close = h["close"]
            price = float(close.iloc[-1])

            # 직전 봉까지의 신고가와 비교(당일 고가를 쓰면 룩어헤드)
            prior_high = float(h["close"].iloc[-self.breakout_lookback - 1 : -1].max())
            if price <= prior_high:
                continue

            ma_f = float(close.rolling(self.fast_ma).mean().iloc[-1])
            ma_s = float(close.rolling(self.slow_ma).mean().iloc[-1])
            if not (price > ma_f > ma_s):
                continue

            slope = ma_s - float(close.rolling(self.slow_ma).mean().iloc[-self.min_slope_bars])
            if slope <= 0:
                continue

            # 확신도: 돌파 폭을 변동성으로 정규화. 과대 해석하지 않도록 상한을 둔다
            vol = float(close.pct_change().rolling(20).std().iloc[-1] or 0)
            edge = (price / prior_high - 1) / vol if vol > 0 else 0.0
            conviction = float(np.clip(0.5 + edge / 4.0, 0.3, 1.0))

            signals.append(
                Signal(symbol, Side.LONG, conviction, f"{self.breakout_lookback}봉 신고가 돌파")
            )
        return signals


class MeanReversion(AlphaModel):
    """
    평균회귀: 장기 상승 추세 안에서의 단기 과매도.

    추세 필터 없이 과매도만 보면 하락 추세에서 계속 물린다.
    """

    name = "meanrev"

    def __init__(self, rsi_period: int = 3, rsi_threshold: float = 15.0, trend_ma: int = 200) -> None:
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.trend_ma = trend_ma

    @property
    def warmup_bars(self) -> int:
        return self.trend_ma + 10

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else np.inf
        return float(100 - 100 / (1 + rs))

    def generate(self, panel: PricePanel, as_of: int) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in panel.symbols:
            h = panel.history(symbol, as_of, lookback=self.warmup_bars)
            if len(h) < self.warmup_bars:
                continue

            close = h["close"]
            price = float(close.iloc[-1])
            ma = float(close.rolling(self.trend_ma).mean().iloc[-1])
            if price <= ma:
                continue

            rsi = self._rsi(close, self.rsi_period)
            if rsi > self.rsi_threshold:
                continue

            conviction = float(np.clip((self.rsi_threshold - rsi) / self.rsi_threshold + 0.4, 0.3, 1.0))
            signals.append(Signal(symbol, Side.LONG, conviction, f"RSI({self.rsi_period})={rsi:.0f} 과매도"))
        return signals
