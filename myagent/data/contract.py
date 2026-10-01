"""
데이터 계약 (point-in-time).

핵심: as_of 시점 이후의 데이터에 접근하면 조용히 통과시키지 않고 예외를 던진다.
룩어헤드는 "조심하면 되는 것"이 아니라 타입 시스템처럼 강제해야 하는 것이다.

The Memorization Problem(arXiv 2504.14765)에서 GPT-4o가 훈련 창 내
S&P500 종가를 1% 미만 오차로 회상한다는 것이 보고됐다. 백테스트가
"예측"이 아니라 "기억"을 측정하고 있을 수 있다는 뜻이다.
가격 데이터에서도 같은 원리로, 미래 봉이 한 줄만 새어들어와도
백테스트 전체가 거짓말이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """as_of 이후 데이터 접근 시도. 절대 무시하지 말 것."""


@dataclass
class PricePanel:
    """
    종목별 OHLC 패널. 모든 접근은 as_of로 잘린 뷰를 통해서만 이루어진다.

    frames: {symbol: DataFrame(index=정수 봉번호, columns=[open,high,low,close])}
    """

    frames: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        required = {"open", "high", "low", "close"}
        lengths = set()
        for symbol, df in self.frames.items():
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{symbol}: 필수 컬럼 누락 {missing}")
            lengths.add(len(df))
        if len(lengths) > 1:
            raise ValueError("모든 종목의 봉 개수가 같아야 합니다 (정렬 후 전달하세요)")
        self._n_bars = lengths.pop() if lengths else 0

    @property
    def symbols(self) -> list[str]:
        return list(self.frames.keys())

    @property
    def n_bars(self) -> int:
        return self._n_bars

    def history(self, symbol: str, as_of: int, lookback: int | None = None) -> pd.DataFrame:
        """
        as_of 봉까지의 데이터만 반환한다(as_of 포함).

        as_of 봉의 종가로 판단하고 다음 봉에 진입하는 것이 기본 가정이다.
        as_of 봉의 종가에 즉시 진입한다고 가정하면 미세한 룩어헤드가 생긴다.
        """
        if as_of < 0 or as_of >= self._n_bars:
            raise LookaheadError(f"as_of={as_of}가 유효 범위를 벗어났습니다")
        df = self.frames[symbol]
        end = as_of + 1
        start = 0 if lookback is None else max(0, end - lookback)
        return df.iloc[start:end]

    def bar(self, symbol: str, index: int, as_of: int) -> dict[str, float]:
        """
        특정 봉을 읽는다. index > as_of이면 예외.

        체결 엔진이 '다음 봉'을 읽어야 할 때는 as_of를 함께 진행시켜야 한다.
        """
        if index > as_of:
            raise LookaheadError(
                f"룩어헤드 차단: {symbol} bar[{index}]를 as_of={as_of}에서 읽으려 했습니다"
            )
        row = self.frames[symbol].iloc[index]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }


def make_synthetic_panel(
    n_symbols: int = 40,
    n_bars: int = 1500,
    seed: int = 42,
    drift: bool = True,
) -> PricePanel:
    """
    합성 OHLC 패널.

    실제 데이터를 쓸 수 없는 환경(이 컨테이너)에서 엔진을 검증하기 위한 것이다.
    실운용에서는 yfinance 등 실제 데이터로 교체해야 하며,
    합성 데이터에서 나온 성과 수치는 전략의 증거가 아니다.
    """
    rng = np.random.default_rng(seed)
    frames: dict[str, pd.DataFrame] = {}

    for i in range(n_symbols):
        vol = rng.uniform(0.012, 0.038)
        mu = rng.uniform(-0.0003, 0.0006) if drift else 0.0

        # 변동성 클러스터링(GARCH 유사) — 실제 시장의 주요 특징
        shocks = np.empty(n_bars)
        sigma = vol
        for t in range(n_bars):
            sigma = np.sqrt(0.90 * sigma**2 + 0.10 * vol**2 + 0.05 * (shocks[t - 1] ** 2 if t else 0))
            sigma = min(sigma, vol * 3)
            shocks[t] = rng.normal(mu, sigma)

        if not drift:
            shocks -= shocks.mean()

        close = 100.0 * np.exp(np.cumsum(shocks))
        gap = rng.normal(0, 0.002, n_bars)
        open_ = np.concatenate([[close[0]], close[:-1] * (1 + gap[1:])])
        span = np.abs(rng.normal(0, vol * 0.7, n_bars))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)

        frames[f"SYM{i:02d}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}
        )

    return PricePanel(frames)


def load_yfinance_panel(
    tickers: list[str],
    start: str,
    end: str,
) -> PricePanel:
    """
    실제 데이터 로더. 이 컨테이너에서는 네트워크가 막혀 있어 동작하지 않는다.
    로컬에서 `pip install yfinance` 후 사용할 것.

    주의: yfinance의 수정주가(auto_adjust)는 과거 값이 소급 변경되므로
    엄밀한 point-in-time이 아니다. 실운용 검증에는 유료 PIT 데이터를 권장한다.
    """
    import yfinance as yf  # noqa: PLC0415

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = pd.DataFrame(
                {
                    "open": raw["Open"][t],
                    "high": raw["High"][t],
                    "low": raw["Low"][t],
                    "close": raw["Close"][t],
                }
            ).dropna()
        except Exception:
            continue
        if len(df) > 0:
            frames[t] = df.reset_index(drop=True)

    lengths = {len(v) for v in frames.values()}
    if len(lengths) > 1:
        n = min(lengths)
        frames = {k: v.iloc[-n:].reset_index(drop=True) for k, v in frames.items()}

    return PricePanel(frames)
