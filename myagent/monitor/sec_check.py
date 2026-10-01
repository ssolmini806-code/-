"""
SEC 접근 진단.

    python -m monitor.sec_check

무엇이 막혔는지 단계별로 확인한다.
"""

from __future__ import annotations

import os

import requests

def _ascii_safe(v: str) -> str:
    try:
        v.encode("latin-1")
        return v
    except UnicodeEncodeError:
        mail = next((p for p in v.split() if "@" in p), "")
        print(f"  ⚠ SEC_UA에 비-ASCII(한글 등) 문자가 있습니다.")
        print(f"    HTTP 헤더는 latin-1만 허용하므로 요청이 전송되지 않습니다.")
        print(f'    해결: export SEC_UA="Solmin Kim {mail or "your@email.com"}"')
        print(f"    이번 진단은 이메일만으로 진행합니다.\n")
        return f"research-tool {mail}" if mail else "research-tool test@test.com"


UA = _ascii_safe(os.environ.get("SEC_UA", ""))

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
