"""
보유 종목 논리 모니터 — 메일 발송.

    python -m monitor.run --dry       # 메일 없이 터미널 출력
    python -m monitor.run             # 메일 발송

설계 원칙
----------
이 도구는 "무슨 일이 있었나"만 말한다. "그래서 사라/팔아라"는 말하지 않는다.

이유: 예측형 시스템을 일곱 번 검증해서 전부 실패했다. 뉴스로 타이밍을
맞히는 것은 가격으로 맞히는 것보다 더 어렵다 — 기사가 나온 시점에는
이미 반영돼 있기 때문이다.

대신 이 도구가 하는 일은 기록 대조다.
    "샀던 이유를 적어뒀다 → 그 이유를 흔드는 일이 생겼는가"
이건 예측이 아니라 확인이고, 검증 가능하다.

메일 설정 (환경변수)
---------------------
    SMTP_USER   보내는 Gmail 주소
    SMTP_PASS   Gmail 앱 비밀번호 (계정 비밀번호 아님)
    MAIL_TO     받을 주소 (없으면 SMTP_USER로)

Gmail 앱 비밀번호는 2단계 인증 설정 후 발급받는다.
계정 비밀번호를 넣지 말 것.
"""

from __future__ import annotations

import argparse
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from monitor.positions import MATERIAL_FORMS, POSITIONS
from monitor.sources import (
    Item,
    cluster_stories,
    diagnose_sec,
    fetch_dxyz_premium,
    fetch_google_news,
    fetch_sec_filings,
    is_junk,
    mentions_subject,
    resolve_cik,
)


def match_terms(item: Item, terms: list[str]) -> list[str]:
    text = f"{item.title} {item.summary}".lower()
    return [t for t in terms if t.lower() in text]


def classify(item: Item, pos) -> str:
    """critical > context > none"""
    if match_terms(item, pos.critical_terms):
        return "critical"
    if match_terms(item, pos.context_terms):
        return "context"
    return "none"


def collect() -> dict:
    out = {}
    for pos in POSITIONS:
        print(f"  {pos.ticker} 수집 중...")
        items: list[Item] = []

        cik = pos.cik or resolve_cik(pos.ticker, verbose=True)
        if cik:
            # 7일은 짧다. 분기 보고 중심 종목은 걸리지 않는다.
            items.extend(fetch_sec_filings(pos.ticker, cik, days=14))
        else:
            print(f"    CIK를 찾지 못함 — 공시 건너뜀")

        for q in (pos.ticker, pos.name):
            items.extend(fetch_google_news(q, pos.ticker, days=3))

        # 잡음 제거 → 주제 확인 → 같은 사건 묶기
        news = [i for i in items if i.source == "NEWS"]
        filings = [i for i in items if i.source == "SEC"]

        before = len(news)
        news = [i for i in news if not is_junk(i)]
        news = [i for i in news if mentions_subject(i, pos.must_mention)]
        news = cluster_stories(sorted(news, key=lambda x: x.published, reverse=True))
        mat = [i for i in filings if i.form_type in MATERIAL_FORMS]
        routine = [i for i in filings if i.form_type not in MATERIAL_FORMS]
        kinds = {}
        for i in routine:
            kinds[i.form_type] = kinds.get(i.form_type, 0) + 1
        kind_s = ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())) or "없음"
        print(f"    기사 {before} → {len(news)} (잡음·중복 제거)")
        print(f"    공시 {len(filings)}건 — 중요 {len(mat)}건 / 일상 {len(routine)}건 ({kind_s})")

        for i in news:
            i.matched_terms = match_terms(i, pos.watch_terms)
            i.form_type = classify(i, pos)   # 등급을 여기에 담는다

        out[pos.ticker] = {"position": pos, "items": filings + news}
    return out


def render(data: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"보유 종목 논리 점검 — {now}", "=" * 64, ""]

    for ticker, d in data.items():
        pos, items = d["position"], d["items"]
        L.append(f"■ {ticker} · {pos.name}")
        L.append(f"  매수 논리: {pos.thesis}")
        L.append("")

        # 종목 고유 지표
        if pos.special_metric == "premium_to_nav":
            px = fetch_dxyz_premium()
            L.append("  [NAV 대비 프리미엄 — 이 종목의 핵심 지표]")
            if px:
                L.append(f"    현재가 ${px['price']:.2f} (5일 {px['change_5d']:+.1%})")
            L.append("    NAV는 분기 공시이므로 수동 확인 필요.")
            L.append("    참고: 2025년 NAV 수익률 +210%, 주가 수익률 -48%.")
            L.append("          보유 자산이 아니라 프리미엄이 손익을 지배한다.")
            L.append("")

        # 논리를 건드리는 항목
        critical = [i for i in items if i.source == "NEWS" and i.form_type == "critical"]
        context = [i for i in items if i.source == "NEWS" and i.form_type == "context"]
        flagged = critical
        material = [i for i in items
                    if i.source == "SEC" and i.form_type in MATERIAL_FORMS]

        all_filings = [i for i in items if i.source == "SEC"]
        routine = [i for i in all_filings if i.form_type not in MATERIAL_FORMS]

        if material:
            L.append("  [공시] ★ 기사보다 먼저, 법적 구속력 있음")
            for i in material[:8]:
                desc = MATERIAL_FORMS.get(i.form_type, "")
                L.append(f"    · {i.form_type} ({i.published.date()}) {desc}")
                L.append(f"      {i.url}")
            L.append("")
        elif all_filings:
            kinds = {}
            for i in routine:
                kinds[i.form_type] = kinds.get(i.form_type, 0) + 1
            ks = ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items()))
            L.append(f"  [공시] 최근 14일 중요 공시 없음 (일상 서류만: {ks})")
            L.append("")
        else:
            L.append("  [공시] 최근 14일 제출 없음")
            L.append("")

        if critical:
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
            L.append("")

        L.append("  [내가 세운 전제 — 아직 유효한가]")
        for dep in pos.thesis_depends_on:
            L.append(f"    □ {dep}")
        L.append("")
        L.append("-" * 64)
        L.append("")

    L.append("이 메일은 '무슨 일이 있었나'만 전달한다.")
    L.append("매수·매도 판단은 하지 않는다. 판단은 본인 몫이다.")
    return "\n".join(L)


def send_mail(body: str) -> bool:
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to = os.environ.get("MAIL_TO") or user

    if not user or not pw:
        print("\nSMTP_USER / SMTP_PASS 환경변수가 없습니다.")
        print("  Gmail 2단계 인증 후 '앱 비밀번호'를 발급받아 넣으세요.")
        print("  계정 비밀번호를 넣지 마세요.")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"[보유 종목 논리 점검] {datetime.now().strftime('%m/%d')}"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.send_message(msg)
        print(f"발송 완료 → {to}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"발송 실패: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="메일 없이 출력만")
    a = ap.parse_args()

    print("수집 시작...")
    data = collect()

    if not any(i.source == "SEC" for d in data.values() for i in d["items"]):
        print("\n⚠ 공시가 하나도 수집되지 않았습니다. 진단:")
        for p in POSITIONS:
            print(f"    {diagnose_sec(p.ticker)}")
        print()
    body = render(data)

    if a.dry:
        print()
        print(body)
    else:
        send_mail(body)


if __name__ == "__main__":
    main()
