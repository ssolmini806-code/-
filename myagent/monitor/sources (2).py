"""
공시·뉴스 수집.

우선순위
---------
1. SEC EDGAR — 공시는 기사보다 먼저 나오고 법적 구속력이 있다.
   DXYZ의 ATM 발행과 NAV 공시도 전부 여기서 먼저 나왔다.
   기사는 공시를 보고 쓰는 것이므로, 공시를 보면 하루 이틀 앞선다.

2. Google News RSS — 커버리지가 넓다. 공시가 아닌 사건(소송 보도, 경영진 발언,
   업계 동향)을 잡는다.

둘 다 무료이고 API 키가 필요 없다.

SEC 요구사항: User-Agent에 연락처를 넣어야 한다(차단 방지).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

# SEC는 User-Agent에 식별 가능한 연락처를 요구한다.
# 환경변수 SEC_UA로 덮어쓸 수 있다.
import os

def _ascii_safe(v: str, fallback: str) -> str:
    """
    HTTP 헤더는 latin-1만 허용한다.
    한글 이름을 SEC_UA에 넣으면 요청이 전송조차 되지 않고
    UnicodeEncodeError가 난다 — SEC가 차단한 것처럼 보이지만 아니다.
    비-latin1 문자가 있으면 이메일만 남기고 이름은 버린다.
    """
    try:
        v.encode("latin-1")
        return v
    except (UnicodeEncodeError, AttributeError):
        parts = [p for p in (v or "").split() if "@" in p]
        if parts:
            print(f"  [알림] SEC_UA에 비-ASCII 문자가 있어 이메일만 사용: {parts[0]}")
            return f"research-tool {parts[0]}"
        print("  [알림] SEC_UA를 ASCII로 사용할 수 없어 기본값으로 대체합니다.")
        return fallback


SEC_UA = _ascii_safe(
    os.environ.get("SEC_UA", "personal-research-tool contact@example.com"),
    "personal-research-tool contact@example.com",
)
SEC_HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}


@dataclass
class Item:
    source: str          # "SEC" | "NEWS"
    ticker: str
    title: str
    url: str
    published: datetime
    form_type: str | None = None
    summary: str = ""
    matched_terms: list[str] | None = None

    @property
    def age_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.published).total_seconds() / 3600


def _get(url: str, headers: dict | None = None, retries: int = 3) -> requests.Response | None:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers or {}, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                return None
            time.sleep(2)
    return None


# ------------------------------------------------------------------ SEC


# CIK 조회가 막혀도 공시는 받을 수 있다. 알려진 값을 미리 둔다.
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
    return None


def fetch_sec_filings(ticker: str, cik: str, days: int = 7) -> list[Item]:
    """최근 공시 목록. submissions API는 전체 이력을 한 번에 준다."""
    h = dict(SEC_HEADERS)
    h["Host"] = "data.sec.gov"
    r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json", h)
    if not r:
        return []

    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    out: list[Item] = []
    for form, d, acc, doc in zip(forms, dates, accs, docs):
        try:
            fd = datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < cutoff:
            break  # 최신순 정렬이므로 중단
        acc_clean = acc.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{acc_clean}/{doc}")
        out.append(Item(
            source="SEC",
            ticker=ticker,
            title=f"[{form}] {ticker}",
            url=url,
            published=datetime.combine(fd, datetime.min.time(), tzinfo=timezone.utc),
            form_type=form,
        ))
    return out


# ------------------------------------------------------------------ 뉴스


def fetch_google_news(query: str, ticker: str, days: int = 3) -> list[Item]:
    """
    Google News RSS. 키 불필요.

    when:3d 같은 연산자로 기간을 제한할 수 있다.
    """
    q = quote(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = _get(url, {"User-Agent": "Mozilla/5.0"})
    if not r:
        return []

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []

    out: list[Item] = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate") or ""
        src = it.find("{http://search.yahoo.com/mrss/}credit")
        try:
            dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            dt = datetime.now(timezone.utc)
        out.append(Item(
            source="NEWS",
            ticker=ticker,
            title=title,
            url=link,
            published=dt,
            summary=src.text if src is not None and src.text else "",
        ))
    return out


# ------------------------------------------------------- 종목 고유 지표


def fetch_dxyz_premium() -> dict | None:
    """
    DXYZ의 NAV 대비 프리미엄.

    왜 별도로 보는가: DXYZ는 폐쇄형 펀드라 주가가 보유 자산 가치와 따로 논다.
    2025년에 NAV 수익률 +210%인데 주가는 -48%였다. 프리미엄이 꺼진 것이다.

    "OpenAI·SpaceX를 담고 있으니 좋다"는 논리는 보유 자산에 대한 것이고,
    실제 손익은 프리미엄 변동이 지배한다. 그래서 이걸 따로 추적한다.

    NAV는 분기마다 공시되므로 자동 조회가 어렵다.
    여기서는 가격만 가져오고, NAV는 사용자가 직접 갱신한다.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        px = yf.Ticker("DXYZ").history(period="5d")
        if px.empty:
            return None
        last = float(px["Close"].iloc[-1])
        prev = float(px["Close"].iloc[0])
        return {"price": last, "change_5d": last / prev - 1}
    except Exception:  # noqa: BLE001
        return None


def _tokens(title: str) -> set[str]:
    import re

    stop = {"the", "a", "an", "of", "on", "in", "to", "for", "and", "as",
            "with", "at", "by", "is", "are", "its", "after", "says", "new"}
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def cluster_stories(items: list, threshold: float = 0.33) -> list:
    """
    같은 사건을 다룬 기사를 묶는다.

    첫 실행에서 'Gemini 4 Argon' 출시 하나로 8건이 걸려 신호가 묻혔다.
    제목 토큰의 자카드 유사도로 묶고, 각 묶음의 대표 1건만 남긴다.

    임계값 0.33: 처음에 0.5로 뒀더니 'Google Unveils Gemini 4 Argon'과
    'Alphabet Stock Jumps as Google Releases Gemini 4 Argon'이 0.45로
    묶이지 않았다. 매체마다 제목 어순과 수식어가 달라 겹침이 생각보다 낮다.
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
                "example.com은 차단될 수 있습니다.\n"
                '      export SEC_UA="이름 your@email.com"')
    cik = resolve_cik(ticker)
    if not cik:
        return f"{ticker}의 CIK를 찾지 못했습니다 (SEC 응답 실패 또는 티커 불일치)"
    n = len(fetch_sec_filings(ticker, cik, days=30))
    if n == 0:
        return f"{ticker} CIK={cik} — 최근 30일 공시 0건 (응답은 정상)"
    return f"{ticker} CIK={cik} — 최근 30일 {n}건 (정상 동작)"
