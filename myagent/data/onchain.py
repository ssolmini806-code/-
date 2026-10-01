"""
온체인 데이터 로더 (Coin Metrics Community API).

API 키 불필요. 비상업적 사용 무료 (Creative Commons).
    https://community-api.coinmetrics.io/v4

왜 온체인인가
--------------
지금까지 여섯 번의 시도는 전부 OHLC만 썼다. 전 세계가 보는 데이터이고,
기관은 더 빨리 본다. 여기서 이기는 건 구조적으로 어렵다.

온체인 데이터는 다르다. 공개돼 있지만 가공이 번거로워 상대적으로 덜 소비된다.
정보 우위가 아니라 노동 우위다.

다만 기대는 낮게 잡아야 한다. MVRV, NVT 같은 지표는 이미 널리 알려져 있다.
"덜 소비된다"는 것이지 "아무도 안 본다"가 아니다.

주의사항
---------
1. 커뮤니티 티어에 어떤 지표가 있는지는 자산마다 다르다.
   하드코딩하지 않고 catalog로 먼저 조회한다.
2. 온체인 지표는 발표 지연이 있을 수 있다. 시점 봉인을 반드시 지킨다.
3. 지표 정의가 소급 변경되는 경우가 있다(재계산). 엄밀한 PIT이 아니다.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE = "https://community-api.coinmetrics.io/v4"

# 온체인 지표 후보. 커뮤니티 티어에 없으면 자동으로 제외된다.
# --discover 결과 확인된 커뮤니티 티어 공통 지표 5종.
# NVTAdj, CapRealUSD, TxTfrValAdjUSD 등은 무료 티어에 없어 제외했다.
CANDIDATE_METRICS = [
    "PriceUSD",           # 기준 가격
    "CapMrktCurUSD",      # 시가총액
    "CapMVRVCur",         # MVRV = 시총/실현시총
    "AdrActCnt",          # 활성 주소 수
    "TxCnt",              # 트랜잭션 수
]

# 온체인 커버리지가 좋은 자산들. 알트는 지표가 부실한 경우가 많다.
DEFAULT_ASSETS = ["btc", "eth", "ltc", "bch", "etc", "xrp", "ada", "doge", "xlm", "link"]


def _get(path: str, params: dict, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            return {"error": f"HTTP {r.status_code}", "body": r.text[:200]}
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                return {"error": str(e)}
            time.sleep(2 * (i + 1))
    return {"error": "재시도 초과"}


def discover_metrics(assets: list[str]) -> dict[str, list[str]]:
    """
    자산별로 커뮤니티 티어에서 실제 사용 가능한 지표를 조회한다.

    하드코딩하면 티어 정책이 바뀔 때 조용히 깨진다.
    무엇이 있는지 먼저 물어보는 편이 안전하다.
    """
    out: dict[str, list[str]] = {}
    for a in assets:
        res = _get("catalog-v2/asset-metrics", {"assets": a, "page_size": 1000})
        if "error" in res:
            print(f"  {a}: 조회 실패 ({res['error']})")
            out[a] = []
            continue
        available = set()
        for row in res.get("data", []):
            for m in row.get("metrics", []):
                available.add(m.get("metric"))
        out[a] = [m for m in CANDIDATE_METRICS if m in available]
        print(f"  {a}: {len(out[a])}/{len(CANDIDATE_METRICS)}개 사용 가능")
    return out


def load_onchain(
    assets: list[str] | None = None,
    metrics: list[str] | None = None,
    start: str = "2016-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    자산별 온체인 지표 시계열을 받는다.

    반환: {asset: DataFrame(index=날짜, columns=지표)}
    """
    assets = assets or DEFAULT_ASSETS
    out: dict[str, pd.DataFrame] = {}

    for a in assets:
        use = metrics or CANDIDATE_METRICS
        rows: list[dict] = []
        token = None
        for _ in range(60):
            params = {
                "assets": a,
                "metrics": ",".join(use),
                "frequency": "1d",
                "start_time": start,
                "page_size": 10000,
            }
            if end:
                params["end_time"] = end
            if token:
                params["next_page_token"] = token
            res = _get("timeseries/asset-metrics", params)
            if "error" in res:
                print(f"  {a}: {res['error']}")
                break
            rows.extend(res.get("data", []))
            token = res.get("next_page_token")
            if not token:
                break

        if not rows:
            print(f"  {a}: 데이터 없음")
            continue

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        df = df.drop(columns=["asset"], errors="ignore").set_index("time").sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all")
        out[a] = df
        print(f"  {a}: {len(df)}일 · 지표 {len(df.columns)}개 "
              f"({df.index[0].date()} ~ {df.index[-1].date()})")

    return out


def align_panel(data: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """여러 자산의 같은 지표를 하나의 DataFrame으로 정렬한다."""
    series = {}
    for a, df in data.items():
        if metric in df.columns:
            series[a] = df[metric]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all")


def expanding_percentile(s: pd.Series, min_periods: int = 365) -> pd.Series:
    """
    확장 윈도우 백분위. t 시점 값이 t까지의 역사에서 몇 퍼센타일인지.

    전체 기간 백분위를 쓰면 미래를 보게 된다. 이건 룩어헤드의 전형이고,
    온체인 지표 백테스트에서 가장 흔한 오류다.
    (예: "MVRV 3.5 이상이면 고점" — 3.5라는 임계값 자체가 미래 데이터에서 나왔다)
    """
    return s.expanding(min_periods=min_periods).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else 0.5, raw=True
    )


def add_derived_metrics(data: dict) -> dict:
    """
    커뮤니티 티어 가용 지표로 파생 지표를 만든다.

    NVTAdj가 무료 티어에 없으므로 직접 계산한다.
    원본과 정의가 다르므로 이름을 구분해 둔다(Proxy).
    """
    for asset, df in data.items():
        if "CapMrktCurUSD" in df.columns and "TxCnt" in df.columns:
            tx = df["TxCnt"].replace(0, float("nan"))
            df["NVTProxy"] = df["CapMrktCurUSD"] / tx
        if "CapMrktCurUSD" in df.columns and "AdrActCnt" in df.columns:
            adr = df["AdrActCnt"].replace(0, float("nan"))
            df["CapPerAdr"] = df["CapMrktCurUSD"] / adr
        data[asset] = df
    return data
