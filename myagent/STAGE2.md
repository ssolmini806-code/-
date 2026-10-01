# 2단계 — 로컬 실행 가이드

이 컨테이너는 금융 데이터 사이트 접근이 막혀 있어 **실제 데이터 검증을 끝내지 못했다.**
아래는 로컬에서 마무리하는 절차다.

## 준비

```bash
pip install yfinance pandas numpy
```

## 실행 순서

### 1. 시간축 유효성 검사 (먼저 이걸 통과해야 함)

```python
from configs import LIVE_SMALL
from data.contract import load_yfinance_panel
from eval.harness import check_timescale_validity
from signals.models import MomentumBreakout

panel = load_yfinance_panel(
    ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","JPM","V","UNH",
     "XOM","JNJ","WMT","PG","MA","HD","CVX","ABBV","KO","PEP"],
    start="2015-01-01", end="2026-09-01",
)
print(check_timescale_validity(panel, MomentumBreakout(), LIVE_SMALL))
```

- `valid: True` → 2단계로 진행
- `valid: False` → 신호에 고유 시간축이 없다는 뜻. 시간 손절값이 임의값이 되므로
  신호 자체를 바꾸거나 시간 손절을 끄는 것을 고려

### 2. MAE/MFE로 time_stop_bars 확정

```bash
python run_stage2.py    # 내부 make_synthetic_panel을 load_yfinance_panel로 교체
```

확정된 값을 `configs.py`의 `time_stop_bars`에 반영하고, 잠정값 주석을 지운다.

### 3. Walk-forward 판정

통과 조건 (셋 다 충족해야 3단계로):

1. 무작위 기준선 대비 평균 R이 **+0.05R 이상** 높을 것
2. test 구간 과반에서 기준선을 이길 것 (한 구간의 대박이 아닐 것)
3. 매수 후 보유 대비 위험조정수익(Sharpe)이 열등하지 않을 것

---

## 이 컨테이너에서 확인된 것 / 확인 못 한 것

| 항목 | 상태 |
|---|---|
| 백테스터가 룩어헤드 없이 도는가 | 확인 (t봉 판단 → t+1봉 시가 체결) |
| walk-forward 분할이 제대로 도는가 | 확인 (3구간) |
| MAE/MFE 측정 기계가 도는가 | 확인 |
| 리스크 엔진이 백테스터와 맞물리는가 | 확인 |
| **time_stop_bars 확정** | **미완** — 합성 데이터에서 시간축 무효 판정 |
| **신호가 기준선을 이기는가** | **미확인** — 합성 데이터로는 판정 불가 |

합성 데이터(GARCH 붙인 기하 브라운 운동)에는 모멘텀도 평균회귀도 구조적으로 존재하지
않는다. 그런 데이터에서 두 신호가 기준선에 미달한 것은 신호가 나쁘다는 증거가 아니라
**데이터에 잡을 것이 없다**는 뜻이다. 판정은 실제 데이터에서만 유효하다.

반대로, 합성 데이터에서 신호가 **잘 나왔다면** 그건 버그 신호였을 것이다.
그런 의미에서 이번 결과는 백테스터가 정직하게 작동한다는 증거이기도 하다.
