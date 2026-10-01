"""
크립토 데이터 로더 (ccxt).

미국 주식 파이프라인과 동일한 PricePanel 계약을 따른다.
ATR·사이징·서킷·부트스트랩·walk-forward는 전부 그대로 재사용된다.
바뀌는 것은 데이터 소스와 비용 파라미터뿐이다.

설치:
    pip install ccxt

주의사항
--------
1. 생존 편향이 주식보다 훨씬 심하다.
   사라진 코인이 수천 개다. "오늘의 시총 상위 N개"로 백테스트하면
   대형주보다 편향이 크다. 깨끗한 대조군(ETF 같은)도 없다.
   → 완화책: 상장 시점이 오래된 자산 위주로 구성하고, 결과를 크게 할인한다.

2. 24시간 거래이므로 봉 사이 공백이 없다.
   대신 주말·공휴일 효과가 없어 주식과 통계 성질이 다르다.

3. 거래소마다 가격이 미세하게 다르다. 하나로 고정해서 재현성을 확보한다.
"""

from __future__ import annotations

import time

import pandas as pd

from data.contract import PricePanel

# 상장이 오래되고 유동성이 큰 자산 위주.
# 신규 상장 코인을 넣으면 생존 편향이 급증한다.
DEFAULT_CRYPTO = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "SOL/USDT", "DOGE/USDT", "DOT/USDT", "AVAX/USDT", "LINK/USDT",
    "LTC/USDT", "ATOM/USDT", "ETC/USDT", "XLM/USDT", "BCH/USDT",
    "FIL/USDT", "TRX/USDT", "NEAR/USDT", "ALGO/USDT", "VET/USDT",
]


def load_crypto_panel(
    symbols: list[str] | None = None,
    timeframe: str = "4h",
    since_days: int = 1095,
    exchange_id: str = "binance",
    max_retries: int = 3,
) -> PricePanel:
    """
    ccxt로 OHLCV를 받아 PricePanel로 변환한다.

    timeframe: '1h', '4h', '1d'
      4h를 기본으로 두는 이유는 같은 기간에 일봉의 6배 관측치를 얻기 때문이다.
      다만 매매가 잦아지므로 비용 부담도 6배가 된다. 공짜가 아니다.
    """
    import ccxt  # noqa: PLC0415

    symbols = symbols or DEFAULT_CRYPTO
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since = ex.milliseconds() - since_days * 24 * 60 * 60 * 1000

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        rows: list[list] = []
        cursor = since
        for _ in range(200):  # 페이지네이션 상한
            batch = None
            for attempt in range(max_retries):
                try:
                    batch = ex.fetch_ohlcv(sym, timeframe, since=cursor, limit=1000)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == max_retries - 1:
                        print(f"  {sym}: 실패 ({type(e).__name__})")
                    time.sleep(1.5 * (attempt + 1))
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            cursor = batch[-1][0] + 1

        if len(rows) < 200:
            print(f"  {sym}: 봉 {len(rows)}개 — 제외")
            continue

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        frames[sym.replace("/", "")] = (
            df[["open", "high", "low", "close"]].astype(float).reset_index(drop=True)
        )
        print(f"  {sym}: 봉 {len(df)}개")

    if not frames:
        raise SystemExit("데이터를 받지 못했습니다.")

    # 봉 개수를 최소값으로 정렬 (PricePanel 요구사항)
    n = min(len(v) for v in frames.values())
    frames = {k: v.iloc[-n:].reset_index(drop=True) for k, v in frames.items()}
    print(f"  정렬 후: 종목 {len(frames)} · 봉 {n}")
    return PricePanel(frames)


def cost_in_r_units(atr_pct: float, atr_stop_multiple: float,
                    commission: float, slippage: float) -> float:
    """
    거래비용을 R 단위로 환산한다.

    이 값이 기대 엣지에 비해 크면 전략이 성립할 수 없다.
    크립토 4시간봉에서는 보통 0.05~0.13R이고,
    미국 주식 일봉의 0.025R보다 2~5배 크다.
    """
    stop_distance_pct = atr_stop_multiple * atr_pct
    round_trip = 2 * (commission + slippage)
    return round_trip / stop_distance_pct if stop_distance_pct > 0 else float("inf")
