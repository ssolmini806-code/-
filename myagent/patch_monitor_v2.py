"""
패치 — 모니터 신호 품질 개선

    python patch_monitor_v2.py
    python -m monitor.run --dry

첫 실행에서 드러난 문제
------------------------
1. SEC 공시가 하나도 안 옴 (가장 중요한 소스인데 조용히 실패)
   → 원인 진단 추가. SEC는 User-Agent에 실제 연락처를 요구한다.

2. 노이즈가 신호를 덮음
   GOOGL 기사 147건 중 '논리 관련' 10건, 그중 8건이 Gemini 4 출시 중복.
   같은 사건을 매체별로 다시 센 것이다.

3. 엉뚱한 기사가 걸림
   DXYZ에서 VCX 기사가 걸렸다. DXYZ 얘기가 아닌데 SpaceX·Anthropic이
   언급돼서 매칭됐다.

4. 데이터 스크래퍼 페이지 유입
   TradingView의 'HAN:W8K' 재무 페이지 같은 것들.

고치는 방법
------------
- 키워드를 2등급으로: critical(논리를 깰 수 있는 것) / context(배경)
  critical만 상단에 올리고 context는 접는다.
- 같은 사건 묶기: 제목 토큰 겹침으로 클러스터링. 대표 1건만 표시.
- 종목명이 제목에 없으면 제외 (VCX 사례 차단)
- 데이터 페이지 도메인 제외
"""

from __future__ import annotations

import os
import sys

POSITIONS_NEW = '''"""
보유 종목과 매수 논리.

이 파일이 모니터의 기준점이다.
"주가가 떨어졌다"가 아니라 "샀던 이유가 흔들렸는가"를 묻기 위해,
그 이유를 먼저 적어둔다.

키워드 2등급
-------------
critical: 이게 걸리면 논리가 흔들릴 수 있다. 상단에 올린다.
context : 관련은 있지만 그 자체로는 논리를 깨지 않는다. 접어서 보여준다.

첫 실행에서 'Gemini' 하나로 8건이 걸려 신호가 묻혔다.
제품 출시는 context, 규제·소송·점유율 하락은 critical이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    ticker: str
    name: str
    thesis: str
    thesis_depends_on: list[str]
    critical_terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    # 제목에 이 중 하나는 반드시 있어야 한다 (엉뚱한 기사 차단)
    must_mention: list[str] = field(default_factory=list)
    special_metric: str | None = None

    @property
    def watch_terms(self) -> list[str]:
        return self.critical_terms + self.context_terms


POSITIONS = [
    Position(
        ticker="DXYZ",
        name="Destiny Tech100",
        thesis=(
            "OpenAI·Anthropic·SpaceX 등 지금 가장 주목받는 비상장 기업들을 "
            "모아둔 펀드라서 충분히 살 만한 가치가 있다고 판단"
        ),
        thesis_depends_on=[
            "보유 비상장 기업들의 가치가 유지·상승할 것",
            "⚠ 주가가 보유 자산 가치(NAV)를 따라갈 것 ← 폐쇄형 펀드라 보장되지 않음",
            "희석(ATM 발행)이 과도하지 않을 것",
        ],
        critical_terms=[
            # 프리미엄·희석 — 이 종목 손익을 지배하는 요인
            "premium to nav", "discount to nav", "premium", "discount",
            "at-the-market", "ATM program", "dilution", "offering",
            "share issuance", "tender offer", "repurchase",
            "net asset value", "NAV per share",
            "expense ratio", "write-down", "markdown", "valuation cut",
        ],
        context_terms=[
            "OpenAI", "Anthropic", "SpaceX", "portfolio company",
            "private company", "pre-IPO", "secondary market",
        ],
        must_mention=["DXYZ", "Destiny Tech"],
        special_metric="premium_to_nav",
    ),
    Position(
        ticker="GOOGL",
        name="Alphabet",
        thesis="AI 쪽에서는 약세지만 그 외 모든 분야에서 너무 안정적이라 매수",
        thesis_depends_on=[
            "검색·광고 사업의 수익성과 점유율이 유지될 것",
            "클라우드·YouTube 등 비AI 사업의 안정성이 유지될 것",
            "AI 약세가 핵심 사업으로 번지지 않을 것",
            "규제·반독점 판결이 사업 구조를 깨지 않을 것",
        ],
        critical_terms=[
            # 사업 구조를 깰 수 있는 것
            "antitrust", "remedy", "divest", "breakup", "DOJ",
            "European Commission", "fine", "ruling", "injunction",
            # 핵심 사업 지표
            "search market share", "ad revenue decline", "advertising revenue",
            "guidance cut", "margin compression", "subpoena",
            # 수익성 훼손
            "capex guidance", "write-off", "impairment", "layoff",
        ],
        context_terms=[
            "Gemini", "cloud", "YouTube", "AI Overviews", "capex",
            "data center", "TPU", "Waymo",
        ],
        must_mention=["Alphabet", "GOOGL", "GOOG", "Google"],
    ),
]


MATERIAL_FORMS = {
    "8-K": "주요 경영사항 (가장 중요)",
    "10-Q": "분기 실적",
    "10-K": "연간 보고서",
    "424B5": "증권 발행 (희석 가능성)",
    "424B3": "증권 발행",
    "S-3": "일괄신고 (발행 준비)",
    "N-CSR": "펀드 연차보고서",
    "N-CSRS": "펀드 반기보고서",
    "N-PORT": "펀드 포트폴리오 명세",
    "DEF 14A": "주주총회 안건",
    "SC 13D": "5% 이상 지분 변동 (경영참여)",
}

# 데이터 스크래퍼·시세 페이지 — 기사가 아니다
JUNK_SOURCES = [
    "tradingview", "kalkine", "simplywall", "marketbeat", "stocktwits",
    "investing.com/equities", "wallmine", "stockinvest", "barchart",
    "zacks.com/stock/quote", "finance.yahoo.com/quote",
]

JUNK_TITLE_PATTERNS = [
    "income statement", "revenue breakdown", "balance sheet",
    "share price, ", "stock news,", "ratio of", "financial statements",
    "after-hours movers", "pre-market movers", "stocks to watch",
    "market wrap", "biggest movers",
]
'''


DEDUP_FN = '''

def _tokens(title: str) -> set[str]:
    import re

    stop = {"the", "a", "an", "of", "on", "in", "to", "for", "and", "as",
            "with", "at", "by", "is", "are", "its", "after", "says", "new"}
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def cluster_stories(items: list, threshold: float = 0.5) -> list:
    """
    같은 사건을 다룬 기사를 묶는다.

    첫 실행에서 'Gemini 4 Argon' 출시 하나로 8건이 걸려 신호가 묻혔다.
    제목 토큰의 자카드 유사도로 묶고, 각 묶음의 대표 1건만 남긴다.
    """
    clusters: list[list] = []
    for it in items:
        t = _tokens(it.title)
        placed = False
        for c in clusters:
            ref = _tokens(c[0].title)
            if not (t | ref):
                continue
            if len(t & ref) / len(t | ref) >= threshold:
                c.append(it)
                placed = True
                break
        if not placed:
            clusters.append([it])

    out = []
    for c in clusters:
        rep = c[0]
        if len(c) > 1:
            rep.summary = f"(유사 보도 {len(c)}건)"
        out.append(rep)
    return out


def is_junk(item) -> bool:
    """데이터 스크래퍼 페이지와 시세 나열 기사를 걸러낸다."""
    from monitor.positions import JUNK_SOURCES, JUNK_TITLE_PATTERNS

    url = (item.url or "").lower()
    title = (item.title or "").lower()
    if any(s in url for s in JUNK_SOURCES):
        return True
    if any(p in title for p in JUNK_TITLE_PATTERNS):
        return True
    return False


def mentions_subject(item, must_mention: list[str]) -> bool:
    """
    제목에 종목명이 없으면 제외.

    첫 실행에서 DXYZ 검색에 VCX 기사가 걸렸다.
    DXYZ 얘기가 아닌데 SpaceX·Anthropic이 언급돼서 매칭된 것이다.
    """
    if not must_mention:
        return True
    t = (item.title or "").lower()
    return any(m.lower() in t for m in must_mention)


def diagnose_sec(ticker: str) -> str:
    """
    SEC 수집이 왜 실패했는지 알려준다.

    첫 실행에서 공시가 하나도 안 왔다. 가장 중요한 소스인데 조용히 실패하면
    '이번 주에 공시가 없었나 보다'로 오해하게 된다.
    """
    import os

    from monitor.sources import SEC_UA, resolve_cik, fetch_sec_filings

    if "example.com" in SEC_UA:
        return ("SEC_UA 환경변수가 기본값입니다. SEC는 실제 연락처를 요구하며 "
                "example.com은 차단될 수 있습니다.\\n"
                '      export SEC_UA="이름 your@email.com"')
    cik = resolve_cik(ticker)
    if not cik:
        return f"{ticker}의 CIK를 찾지 못했습니다 (SEC 응답 실패 또는 티커 불일치)"
    n = len(fetch_sec_filings(ticker, cik, days=30))
    if n == 0:
        return f"{ticker} CIK={cik} — 최근 30일 공시 0건 (응답은 정상)"
    return f"{ticker} CIK={cik} — 최근 30일 {n}건 (정상 동작)"
'''


def patch_positions() -> None:
    with open("monitor/positions.py", "w", encoding="utf-8") as f:
        f.write(POSITIONS_NEW)
    print("  monitor/positions.py: 키워드 2등급화 + 잡음 필터 목록 추가")


def patch_sources() -> None:
    p = "monitor/sources.py"
    s = open(p, encoding="utf-8").read()
    if "cluster_stories" in s:
        print("  monitor/sources.py: 이미 적용됨")
        return
    open(p, "w", encoding="utf-8").write(s + DEDUP_FN)
    print("  monitor/sources.py: 중복 묶기·잡음 필터·SEC 진단 추가")


def patch_run() -> None:
    p = "monitor/run.py"
    s = open(p, encoding="utf-8").read()
    if "cluster_stories" in s:
        print("  monitor/run.py: 이미 적용됨")
        return

    s = s.replace(
        """from monitor.sources import (
    Item,
    fetch_dxyz_premium,
    fetch_google_news,
    fetch_sec_filings,
    resolve_cik,
)""",
        """from monitor.sources import (
    Item,
    cluster_stories,
    diagnose_sec,
    fetch_dxyz_premium,
    fetch_google_news,
    fetch_sec_filings,
    is_junk,
    mentions_subject,
    resolve_cik,
)""",
    )

    s = s.replace(
        '''def match_terms(item: Item, terms: list[str]) -> list[str]:
    text = f"{item.title} {item.summary}".lower()
    return [t for t in terms if t.lower() in text]''',
        '''def match_terms(item: Item, terms: list[str]) -> list[str]:
    text = f"{item.title} {item.summary}".lower()
    return [t for t in terms if t.lower() in text]


def classify(item: Item, pos) -> str:
    """critical > context > none"""
    if match_terms(item, pos.critical_terms):
        return "critical"
    if match_terms(item, pos.context_terms):
        return "context"
    return "none"''',
    )

    s = s.replace(
        '''        # 중복 제거 (제목 기준)
        seen, uniq = set(), []
        for it in sorted(items, key=lambda x: x.published, reverse=True):
            key = it.title.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            it.matched_terms = match_terms(it, pos.watch_terms)
            uniq.append(it)

        out[pos.ticker] = {"position": pos, "items": uniq}''',
        '''        # 잡음 제거 → 주제 확인 → 같은 사건 묶기
        news = [i for i in items if i.source == "NEWS"]
        filings = [i for i in items if i.source == "SEC"]

        before = len(news)
        news = [i for i in news if not is_junk(i)]
        news = [i for i in news if mentions_subject(i, pos.must_mention)]
        news = cluster_stories(sorted(news, key=lambda x: x.published, reverse=True))
        print(f"    기사 {before} → {len(news)} (잡음·중복 제거), 공시 {len(filings)}")

        for i in news:
            i.matched_terms = match_terms(i, pos.watch_terms)
            i.form_type = classify(i, pos)   # 등급을 여기에 담는다

        out[pos.ticker] = {"position": pos, "items": filings + news}''',
    )

    # 렌더링: critical / context 분리
    s = s.replace(
        '''        flagged = [i for i in items if i.matched_terms]''',
        '''        critical = [i for i in items if i.source == "NEWS" and i.form_type == "critical"]
        context = [i for i in items if i.source == "NEWS" and i.form_type == "context"]
        flagged = critical''',
    )

    s = s.replace(
        '''        if flagged:
            L.append("  [논리 관련 기사] — 아래 단어가 포함된 것")
            for i in flagged[:10]:
                if i.source == "SEC":
                    continue
                terms = ", ".join(i.matched_terms[:4])
                L.append(f"    · {i.title}")
                L.append(f"      관련: {terms}")
                L.append(f"      {i.url}")
            L.append("")

        others = [i for i in items if not i.matched_terms and i.source == "NEWS"]
        if others:
            L.append(f"  [기타 기사 {len(others)}건]")
            for i in others[:5]:
                L.append(f"    · {i.title}")
            L.append("")''',
        '''        if critical:
            L.append("  [논리를 흔들 수 있는 것] ★ 먼저 볼 것")
            for i in critical[:6]:
                terms = ", ".join(i.matched_terms[:3])
                L.append(f"    · {i.title} {i.summary}")
                L.append(f"      관련: {terms}")
                L.append(f"      {i.url}")
            L.append("")
        else:
            L.append("  [논리를 흔들 수 있는 것] 없음")
            L.append("")

        if context:
            L.append(f"  [배경 — 관련 있으나 논리를 깨지는 않음, {len(context)}건]")
            for i in context[:5]:
                L.append(f"    · {i.title} {i.summary}")
            L.append("")

        others = [i for i in items if i.source == "NEWS" and i.form_type == "none"]
        if others:
            L.append(f"  [기타 {len(others)}건 — 생략]")
            L.append("")''',
    )

    # SEC 진단 추가
    s = s.replace(
        '''    print("수집 시작...")
    data = collect()''',
        '''    print("수집 시작...")
    data = collect()

    if not any(i.source == "SEC" for d in data.values() for i in d["items"]):
        print("\\n⚠ 공시가 하나도 수집되지 않았습니다. 진단:")
        for p in POSITIONS:
            print(f"    {diagnose_sec(p.ticker)}")
        print()''',
    )

    open(p, "w", encoding="utf-8").write(s)
    print("  monitor/run.py: 등급 분리 + SEC 진단 연결")


def main() -> None:
    if not os.path.isdir("monitor"):
        sys.exit("myagent/ 폴더 안에서 실행하세요 (monitor/ 가 보여야 합니다).")
    print("패치 적용 중...")
    patch_positions()
    patch_sources()
    patch_run()
    print()
    print("완료. 실행 전에 SEC 연락처를 설정하세요:")
    print('    export SEC_UA="이름 your@email.com"')
    print("    python -m monitor.run --dry")


if __name__ == "__main__":
    main()
