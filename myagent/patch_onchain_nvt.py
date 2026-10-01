"""
패치 — On3 대체 지표 (NVTAdj 부재 대응)

    python patch_onchain_nvt.py     # 적용
    python exp_04_onchain.py --fetch
    python exp_04_onchain.py --run

상황
-----
--discover 결과, 커뮤니티 티어 공통 지표는 다섯 개다.
    AdrActCnt, CapMVRVCur, CapMrktCurUSD, PriceUSD, TxCnt

NVTAdj는 0/10 자산으로 사용 불가.

대체
-----
원래 NVT = 시가총액 / 온체인 전송액.
전송액(TxTfrValAdjUSD)이 없으므로 트랜잭션 수로 대체한다.

    NVT_proxy = CapMrktCurUSD / TxCnt

"트랜잭션 1건당 시가총액"이다. 이 값이 높으면 사용량 대비 비싸다는 뜻.
원본과 정확히 같지는 않다 — 전송액은 금액을, 트랜잭션 수는 건수를 센다.
같은 사용량이라도 큰 거래 몇 건과 작은 거래 여러 건이 다르게 잡힌다.
그래서 결과를 원본 NVT의 검증으로 읽으면 안 된다. 별개 지표다.

추가로 On5를 넣는다.
    On5 활성주소 대비 시총 = CapMrktCurUSD / AdrActCnt
    메트칼프 법칙 계열 발상. 사용자 수 대비 가치.
    TxCnt보다 조작에 덜 취약하다는 점에서 보완적이다.
"""

from __future__ import annotations

import os
import sys

PATCH = '''

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
'''


def patch_loader() -> None:
    p = "data/onchain.py"
    s = open(p, encoding="utf-8").read()

    if "add_derived_metrics" in s:
        print("  data/onchain.py: 이미 적용됨")
        return

    # 후보 지표 목록에서 없는 것 제거, 있는 것만 남김
    s = s.replace(
        '''CANDIDATE_METRICS = [
    "PriceUSD",           # 기준 가격
    "CapMrktCurUSD",      # 시가총액
    "CapRealUSD",         # 실현 시총 — 마지막 이동 시점 가격 기준
    "CapMVRVCur",         # MVRV = 시총/실현시총. 대표적 밸류에이션 지표
    "AdrActCnt",          # 활성 주소 수 — 네트워크 사용량
    "TxCnt",              # 트랜잭션 수
    "TxTfrValAdjUSD",     # 조정 전송액 — NVT 계산용
    "NVTAdj",             # NVT 비율
    "SplyAdrBalUSD1M",    # 100만 달러 이상 보유 주소 공급량 (고래)
    "FeeTotUSD",          # 총 수수료 — 블록스페이스 수요
    "HashRate",
    "VtyDayRet30d",       # 30일 실현 변동성
]''',
        '''# --discover 결과 확인된 커뮤니티 티어 공통 지표 5종.
# NVTAdj, CapRealUSD, TxTfrValAdjUSD 등은 무료 티어에 없어 제외했다.
CANDIDATE_METRICS = [
    "PriceUSD",           # 기준 가격
    "CapMrktCurUSD",      # 시가총액
    "CapMVRVCur",         # MVRV = 시총/실현시총
    "AdrActCnt",          # 활성 주소 수
    "TxCnt",              # 트랜잭션 수
]''',
    )
    open(p, "w", encoding="utf-8").write(s + PATCH)
    print("  data/onchain.py: 지표 목록 정리 + 파생 지표 함수 추가")


def patch_experiment() -> None:
    p = "exp_04_onchain.py"
    s = open(p, encoding="utf-8").read()

    if "NVTProxy" in s:
        print("  exp_04_onchain.py: 이미 적용됨")
        return

    # fetch 시 파생 지표 생성
    s = s.replace(
        """    data = load_onchain(DEFAULT_ASSETS, start="2016-01-01")
    if not data:
        raise SystemExit("데이터를 받지 못했습니다.")""",
        """    data = load_onchain(DEFAULT_ASSETS, start="2016-01-01")
    if not data:
        raise SystemExit("데이터를 받지 못했습니다.")
    from data.onchain import add_derived_metrics
    data = add_derived_metrics(data)
    print("파생 지표 추가: NVTProxy(시총/트랜잭션수), CapPerAdr(시총/활성주소)")""",
    )

    # NVT를 프록시로 교체
    s = s.replace(
        '''        if mode in ("nvt_value", "composite"):
            nvt = align_panel(data, "NVTAdj").reindex(idx).ffill()''',
        '''        if mode in ("nvt_value", "composite"):
            nvt = align_panel(data, "NVTProxy").reindex(idx).ffill()''',
    )

    # On5 추가
    s = s.replace(
        '''    if mode in ("adr_growth", "nvt_value", "composite"):''',
        '''    if mode == "cap_per_adr":
        # On5 — 활성주소 대비 시총. 낮을수록 사용자 수 대비 저평가.
        cpa = align_panel(data, "CapPerAdr").reindex(idx).ffill()
        score = (-cpa).rank(axis=1, pct=True).shift(shift_days)
        score = score.reindex(columns=cols)
        w = score.div(score.sum(axis=1), axis=0)
        return w.fillna(1.0 / n)

    if mode in ("adr_growth", "nvt_value", "composite"):''',
    )

    s = s.replace(
        '''        ("On3 NVT 저평가", "nvt_value"),''',
        '''        ("On3 NVT대용 저평가", "nvt_value"),
        ("On5 주소당시총", "cap_per_adr"),''',
    )
    open(p, "w", encoding="utf-8").write(s)
    print("  exp_04_onchain.py: On3를 프록시로 교체, On5 추가")


def main() -> None:
    if not os.path.exists("data/onchain.py"):
        sys.exit("myagent/ 폴더 안에서 실행하세요.")
    print("패치 적용 중...")
    patch_loader()
    patch_experiment()
    print()
    print("완료. 다음을 실행하세요:")
    print("    python exp_04_onchain.py --fetch")
    print("    python exp_04_onchain.py --run")


if __name__ == "__main__":
    main()
