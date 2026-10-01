"""
실행 품질 계산기 (C)

    python exec_quality.py --demo          # 예시로 구조 보기
    python exec_quality.py --holdings holdings.csv

무엇을 하는가
--------------
시장을 이기려 하지 않는다. 이미 얻은 수익에서 새는 것을 막는다.
연 0.5~1%p 수준이지만 확률 100%로 실현되고, 복리로 쌓인다.

세 가지 지렛대
---------------
1. 기본공제 연 250만원 (소득세법 §118의2)
   해외주식 양도차익에서 연 250만원을 공제하고 나머지에 22%
   (양도소득세 20% + 지방소득세 2%). 다음 해 5월 확정신고.
   → 공제는 해마다 리셋된다. 큰 이익을 한 해에 몰아 실현하면 공제를 한 번만
     쓰고, 두 해에 나누면 두 번 쓴다.

2. 손익통산
   같은 연도 안에서 실현한 차익과 차손을 합산한다.
   평가손실은 12월 31일까지 팔지 않으면 그해 계산에 안 들어간다.
   → 이익 종목만 팔면 세금을 더 낸다. 손실 종목을 함께 실현하면 통산된다.

3. 거래·환전 비용
   증권사 수수료, 환전 스프레드, 리밸런싱 빈도.

주의
-----
이 계산기는 구조를 보여줄 뿐이며 세무 자문이 아니다.
실제 적용 전 세무 전문가 확인이 필요하다. 특히:
  - 국내 대주주 양도소득이 있으면 250만원 공제를 나눠 쓴다
  - 배당소득은 별도 체계다(종합과세 여부 확인 필요)
  - 매도 후 동일 종목 재매수 시 취득단가가 바뀌어 다음 해 부담이 달라진다
  - 제도는 바뀐다. 신고 전 국세청 기준을 확인할 것
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

DEDUCTION = 2_500_000      # 연간 기본공제
TAX_RATE = 0.22            # 양도세 20% + 지방소득세 2%


@dataclass
class Holding:
    name: str
    quantity: float
    cost_basis_krw: float      # 취득가액 (원화 환산 총액)
    current_value_krw: float   # 현재 평가액 (원화 환산 총액)

    @property
    def unrealized(self) -> float:
        return self.current_value_krw - self.cost_basis_krw


def tax_on(gain: float, deduction_used: float = 0.0) -> float:
    """양도차익에 대한 세액. 손실이면 0."""
    remaining = max(0.0, DEDUCTION - deduction_used)
    taxable = max(0.0, gain - remaining)
    return taxable * TAX_RATE


# --------------------------------------------------------------- 지렛대 1


def split_realization(total_gain: float, years: int) -> dict:
    """
    이익 실현을 여러 해로 나눌 때의 절세액.

    공제는 해마다 리셋되므로, 한 해에 몰면 250만원을 한 번만 쓴다.
    """
    lump = tax_on(total_gain)
    per_year = total_gain / years
    split = sum(tax_on(per_year) for _ in range(years))
    return {
        "total_gain": total_gain,
        "years": years,
        "tax_lump": lump,
        "tax_split": split,
        "saved": lump - split,
        "saved_pct": (lump - split) / total_gain if total_gain > 0 else 0.0,
    }


# --------------------------------------------------------------- 지렛대 2


def loss_harvest(holdings: list[Holding], realized_gain_ytd: float = 0.0) -> dict:
    """
    연말 손익통산 시나리오.

    평가손실 종목을 연내에 실현하면 그해 차익과 통산된다.
    실현하지 않으면 그해 계산에 들어가지 않는다.
    """
    losers = [h for h in holdings if h.unrealized < 0]
    total_loss = sum(h.unrealized for h in losers)

    tax_without = tax_on(realized_gain_ytd)
    tax_with = tax_on(realized_gain_ytd + total_loss)

    return {
        "realized_gain_ytd": realized_gain_ytd,
        "harvestable_loss": total_loss,
        "losers": [(h.name, h.unrealized) for h in losers],
        "tax_without": tax_without,
        "tax_with": tax_with,
        "saved": tax_without - tax_with,
        "deduction_headroom": max(0.0, DEDUCTION - max(0.0, realized_gain_ytd + total_loss)),
    }


# --------------------------------------------------------------- 지렛대 3


def cost_drag(
    portfolio_krw: float,
    rebalances_per_year: int,
    turnover_per_rebalance: float,
    commission: float,
    fx_spread: float,
    fx_conversions_per_year: int = 2,
) -> dict:
    """
    연간 비용 드래그.

    리밸런싱은 세금과 상충한다. 자주 하면 비중은 잘 맞지만
    거래비용과 실현 이익(=과세)이 늘어난다.
    """
    trade_cost = (
        portfolio_krw * turnover_per_rebalance * rebalances_per_year * commission * 2
    )
    fx_cost = portfolio_krw * fx_spread * fx_conversions_per_year
    total = trade_cost + fx_cost
    return {
        "trade_cost": trade_cost,
        "fx_cost": fx_cost,
        "total": total,
        "drag_pct": total / portfolio_krw if portfolio_krw > 0 else 0.0,
    }


def compound_impact(annual_drag_pct: float, years: int, gross_return: float = 0.08) -> dict:
    """비용 드래그가 복리로 미치는 영향."""
    with_drag = (1 + gross_return - annual_drag_pct) ** years
    without = (1 + gross_return) ** years
    return {
        "years": years,
        "with_drag": with_drag - 1,
        "without_drag": without - 1,
        "gap_pct": (without - with_drag) / without,
    }


# --------------------------------------------------------------- 출력


def won(x: float) -> str:
    return f"{x:>13,.0f}원"


def demo() -> None:
    print("=" * 74)
    print("실행 품질 계산기 — 구조 확인용 예시")
    print("=" * 74)
    print("아래 숫자는 예시다. 본인 수치로 바꿔 넣어야 의미가 있다.\n")

    # ---- 지렛대 1
    print("─" * 74)
    print("1. 기본공제 250만원 — 실현을 나누면 공제를 여러 번 쓴다")
    print("─" * 74)
    for gain in (3_000_000, 6_000_000, 10_000_000):
        r2 = split_realization(gain, 2)
        r3 = split_realization(gain, 3)
        print(f"  양도차익 {won(gain)}")
        print(f"    한 해에 전부   세금 {won(r2['tax_lump'])}")
        print(f"    2년 분할      세금 {won(r2['tax_split'])}  절세 {won(r2['saved'])}")
        print(f"    3년 분할      세금 {won(r3['tax_split'])}  절세 {won(r3['saved'])}")
        print()
    print("  주의: 분할하려면 매도를 미뤄야 하고, 그동안 가격이 변한다.")
    print("        세금을 아끼려다 더 큰 가격 변동을 감수하는 것은 본말전도다.")

    # ---- 지렛대 2
    print()
    print("─" * 74)
    print("2. 손익통산 — 평가손실은 팔아야 계산에 들어간다")
    print("─" * 74)
    holdings = [
        Holding("A", 10, 5_000_000, 8_000_000),
        Holding("B", 20, 4_000_000, 3_200_000),
        Holding("C", 5, 3_000_000, 2_100_000),
    ]
    for realized in (0, 5_000_000):
        r = loss_harvest(holdings, realized)
        print(f"  올해 이미 실현한 차익 {won(realized)}")
        print(f"    실현 가능한 손실   {won(r['harvestable_loss'])}"
              f"  ({', '.join(n for n, _ in r['losers'])})")
        print(f"    통산 안 하면       세금 {won(r['tax_without'])}")
        print(f"    통산 하면          세금 {won(r['tax_with'])}"
              f"   절세 {won(r['saved'])}")
        print()
    print("  주의: 손실 실현 후 같은 종목을 다시 사면 취득단가가 낮아진다.")
    print("        올해 아낀 세금을 내년에 더 낼 수 있다. 이연이지 소멸이 아니다.")

    # ---- 지렛대 3
    print()
    print("─" * 74)
    print("3. 거래·환전 비용 — 리밸런싱 빈도의 상충")
    print("─" * 74)
    pf = 10_000_000
    print(f"  포트폴리오 {won(pf)} · 수수료 0.07% · 환전 스프레드 0.1% 가정\n")
    print(f"  {'리밸런싱':<12}{'거래비용':>14}{'환전비용':>14}{'합계':>14}{'드래그':>9}")
    for n, label in [(1, "연 1회"), (4, "분기"), (12, "월간"), (52, "주간")]:
        c = cost_drag(pf, n, 0.20, 0.0007, 0.001, fx_conversions_per_year=2)
        print(f"  {label:<12}{c['trade_cost']:>13,.0f}원{c['fx_cost']:>13,.0f}원"
              f"{c['total']:>13,.0f}원{c['drag_pct']:>8.2%}")

    print()
    print("  10년 복리 영향 (연 8% 총수익 가정)")
    for drag in (0.001, 0.005, 0.01, 0.02):
        r = compound_impact(drag, 10)
        print(f"    연 {drag:.1%} 드래그 → 10년 후 {r['with_drag']:+.1%} "
              f"(무비용 {r['without_drag']:+.1%}, 격차 {r['gap_pct']:.1%})")

    print()
    print("=" * 74)
    print("정리")
    print("=" * 74)
    print("  · 리밸런싱을 자주 하면 비중은 잘 맞지만 비용과 과세가 늘어난다.")
    print("  · 연 1~2회가 대체로 균형점이다. 주간 리밸런싱은 근거가 필요하다.")
    print("  · 세금은 '아끼는' 것보다 '미루는' 것이다. 이연된 세금은 언젠가 낸다.")
    print("  · 가장 확실한 절감은 환전 횟수를 줄이는 것이다. 세금과 무관하다.")
    print()
    print("  ※ 세무 자문이 아니다. 실제 적용 전 전문가 확인이 필요하다.")


def from_csv(path: str) -> None:
    import csv

    holdings = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            holdings.append(
                Holding(
                    row["name"],
                    float(row.get("quantity", 0)),
                    float(row["cost_basis_krw"]),
                    float(row["current_value_krw"]),
                )
            )

    print("=" * 74)
    print("보유 종목 분석")
    print("=" * 74)
    print(f"  {'종목':<12}{'취득가액':>14}{'평가액':>14}{'평가손익':>14}")
    for h in holdings:
        print(f"  {h.name:<12}{h.cost_basis_krw:>13,.0f}원{h.current_value_krw:>13,.0f}원"
              f"{h.unrealized:>+13,.0f}원")

    total_gain = sum(h.unrealized for h in holdings if h.unrealized > 0)
    r = loss_harvest(holdings, 0)
    print(f"\n  평가이익 합계 {won(total_gain)}")
    print(f"  평가손실 합계 {won(r['harvestable_loss'])}")
    print(f"\n  전량 실현 시 세금 {won(tax_on(total_gain + r['harvestable_loss']))}")
    print(f"  이익만 실현 시   {won(tax_on(total_gain))}")
    print(f"  → 통산 효과 {won(tax_on(total_gain) - tax_on(total_gain + r['harvestable_loss']))}")

    if total_gain > DEDUCTION:
        s = split_realization(total_gain + r["harvestable_loss"], 2)
        print(f"\n  2년 분할 시 추가 절세 {won(s['saved'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--holdings", help="CSV: name,quantity,cost_basis_krw,current_value_krw")
    a = ap.parse_args()
    if a.holdings:
        from_csv(a.holdings)
    else:
        demo()


if __name__ == "__main__":
    main()
