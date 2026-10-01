"""
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
    cik: str | None = None   # SEC 조회가 막힐 때 직접 지정

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
            "[주의] 주가가 보유 자산 가치(NAV)를 따라갈 것 ← 폐쇄형 펀드라 보장되지 않음",
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
        must_mention=["Alphabet", "GOOGL", "GOOG"],
        cik="0001652044",
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
