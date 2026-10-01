"""
패치 — SEC 접근 실패 진단 및 우회

    python patch_monitor_sec.py
    python -m monitor.sec_check        # 원인 진단 먼저
    python -m monitor.run --dry

상황
-----
SEC_UA를 설정했는데도 CIK 조회가 실패한다.
Google News는 되는데 SEC만 안 되는 상황이므로 네트워크 문제는 아니다.

가능한 원인
------------
1. SEC가 데이터센터 IP를 차단 — Codespaces는 클라우드 IP다. 흔한 경우.
2. User-Agent 형식 불일치 — SEC는 "회사명 이메일" 형식을 권장한다.
3. 요청 빈도 제한 (초당 10회 초과)
4. company_tickers.json 엔드포인트 변경

대응
-----
- sec_check 모듈로 실제 HTTP 상태코드와 예외를 출력한다.
  "실패했다"가 아니라 "왜 실패했는지"를 봐야 고칠 수 있다.
- CIK를 직접 지정할 수 있게 한다. 조회가 막혀도 공시 자체는 받을 수 있다.
- 그래도 안 되면 SEC는 포기하고 뉴스만 쓴다. 공시가 더 좋지만 필수는 아니다.
"""

from __future__ import annotations

import os
import sys

SEC_CHECK = '''"""
SEC 접근 진단.

    python -m monitor.sec_check

무엇이 막혔는지 단계별로 확인한다.
"""

from __future__ import annotations

import os

import requests

UA = os.environ.get("SEC_UA", "")

ENDPOINTS = [
    ("티커→CIK 매핑", "https://www.sec.gov/files/company_tickers.json"),
    ("제출 이력 (Alphabet)", "https://data.sec.gov/submissions/CIK0001652044.json"),
    ("EDGAR 메인", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL&type=8-K&dateb=&owner=include&count=5&output=atom"),
]


def main() -> None:
    print("=" * 66)
    print("SEC 접근 진단")
    print("=" * 66)
    print(f"SEC_UA = {UA!r}")
    if not UA:
        print("  ⚠ 비어 있음. SEC는 User-Agent를 요구한다.")
        print('    export SEC_UA="이름 your@email.com"')
    elif "example.com" in UA:
        print("  ⚠ 기본값. 실제 이메일로 바꿀 것.")
    elif "@" not in UA:
        print("  ⚠ 이메일이 없음. SEC는 연락 가능한 주소를 요구한다.")
    print()

    headers = {"User-Agent": UA or "test test@test.com",
               "Accept-Encoding": "gzip, deflate",
               "Host": "www.sec.gov"}

    for name, url in ENDPOINTS:
        h = dict(headers)
        if "data.sec.gov" in url:
            h["Host"] = "data.sec.gov"
        try:
            r = requests.get(url, headers=h, timeout=20)
            size = len(r.content)
            print(f"  {name}")
            print(f"    HTTP {r.status_code} · {size:,} bytes")
            if r.status_code == 200 and size > 100:
                print("    → 정상")
            elif r.status_code == 403:
                print("    → 차단됨. SEC가 이 IP 또는 User-Agent를 거부.")
                print(f"       응답: {r.text[:150]}")
            elif r.status_code == 429:
                print("    → 요청 과다. 잠시 후 재시도.")
            else:
                print(f"       응답: {r.text[:150]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}")
            print(f"    예외: {type(e).__name__}: {e}")
        print()

    print("=" * 66)
    print("해석")
    print("=" * 66)
    print("  전부 403 → SEC가 Codespaces 같은 데이터센터 IP를 차단 중일 수 있다.")
    print("            로컬 PC에서 돌리면 될 가능성이 높다.")
    print("  일부만 200 → 되는 엔드포인트만 쓰면 된다.")
    print("  전부 200인데 run에서 실패 → 코드 문제. CIK 수동 지정으로 우회.")
    print()
    print("  SEC가 끝내 안 되면 공시는 포기하고 뉴스만 쓴다.")
    print("  공시가 더 좋지만 필수는 아니다.")


if __name__ == "__main__":
    main()
'''


def patch_sources() -> None:
    p = "monitor/sources.py"
    s = open(p, encoding="utf-8").read()

    if "KNOWN_CIK" in s:
        print("  monitor/sources.py: 이미 적용됨")
        return

    s = s.replace(
        '''def resolve_cik(ticker: str) -> str | None:
    """티커 → CIK. SEC가 전체 매핑을 공개한다."""
    r = _get("https://www.sec.gov/files/company_tickers.json", SEC_HEADERS)
    if not r:
        return None
    for row in r.json().values():
        if row.get("ticker", "").upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    return None''',
        '''# CIK 조회가 막혀도 공시는 받을 수 있다. 알려진 값을 미리 둔다.
# 다른 종목을 추가하려면 sec.gov에서 "EDGAR full-text search"로 찾으면 된다.
KNOWN_CIK = {
    "GOOGL": "0001652044",   # Alphabet Inc.
    "GOOG": "0001652044",
}


def resolve_cik(ticker: str, verbose: bool = False) -> str | None:
    """
    티커 → CIK.

    SEC 매핑 조회가 실패하는 경우가 있다(데이터센터 IP 차단 등).
    알려진 값이 있으면 그걸 먼저 쓰고, 없을 때만 조회한다.
    """
    t = ticker.upper()
    if t in KNOWN_CIK:
        return KNOWN_CIK[t]

    r = _get("https://www.sec.gov/files/company_tickers.json", SEC_HEADERS)
    if not r:
        if verbose:
            print(f"      SEC 매핑 조회 실패 (UA={SEC_UA[:40]!r})")
        return None
    try:
        for row in r.json().values():
            if row.get("ticker", "").upper() == t:
                return str(row["cik_str"]).zfill(10)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"      응답 파싱 실패: {type(e).__name__}")
    return None''',
    )

    # data.sec.gov는 Host 헤더가 다르다
    s = s.replace(
        '''    r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json", SEC_HEADERS)
    if not r:
        return []''',
        '''    h = dict(SEC_HEADERS)
    h["Host"] = "data.sec.gov"
    r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json", h)
    if not r:
        return []''',
    )

    open(p, "w", encoding="utf-8").write(s)
    print("  monitor/sources.py: CIK 수동 지정 + Host 헤더 수정")


def patch_positions() -> None:
    """포지션에 CIK를 직접 적을 수 있게 한다."""
    p = "monitor/positions.py"
    s = open(p, encoding="utf-8").read()
    if "cik:" in s:
        print("  monitor/positions.py: 이미 적용됨")
        return

    s = s.replace(
        """    must_mention: list[str] = field(default_factory=list)
    special_metric: str | None = None""",
        """    must_mention: list[str] = field(default_factory=list)
    special_metric: str | None = None
    cik: str | None = None   # SEC 조회가 막힐 때 직접 지정""",
    )
    s = s.replace(
        '''        must_mention=["Alphabet", "GOOGL", "GOOG", "Google"],
    ),''',
        '''        must_mention=["Alphabet", "GOOGL", "GOOG"],
        cik="0001652044",
    ),''',
    )
    open(p, "w", encoding="utf-8").write(s)
    print("  monitor/positions.py: cik 필드 추가, GOOGL CIK 지정")
    print("    (must_mention에서 'Google' 제거 — 너무 넓어 기사 103건이 통과했음)")


def patch_run() -> None:
    p = "monitor/run.py"
    s = open(p, encoding="utf-8").read()
    if "pos.cik" in s:
        print("  monitor/run.py: 이미 적용됨")
        return

    s = s.replace(
        """        cik = resolve_cik(pos.ticker)
        if cik:""",
        """        cik = pos.cik or resolve_cik(pos.ticker, verbose=True)
        if cik:""",
    )
    open(p, "w", encoding="utf-8").write(s)
    print("  monitor/run.py: 지정된 CIK 우선 사용")


def main() -> None:
    if not os.path.isdir("monitor"):
        sys.exit("myagent/ 폴더 안에서 실행하세요.")
    print("패치 적용 중...")
    with open("monitor/sec_check.py", "w", encoding="utf-8") as f:
        f.write(SEC_CHECK)
    print("  monitor/sec_check.py: 진단 모듈 생성")
    patch_sources()
    patch_positions()
    patch_run()
    print()
    print("완료. 진단부터 돌리세요:")
    print("    python -m monitor.sec_check")
    print("    python -m monitor.run --dry")


if __name__ == "__main__":
    main()
