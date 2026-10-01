# 실행 가이드

## 파일 배치

받은 파일을 이 구조로 놓는다. 폴더 이름이 임포트 경로라 그대로 맞춰야 한다.

```
myagent/
├── configs.py
├── cycle.py
├── run_real.py            ← 실데이터 실행 (이걸 돌린다)
├── run_stage2.py          ← 합성데이터 실행 (참고용)
├── stress_random.py       ← 1단계 스트레스 테스트
├── test_invariants.py     ← 회귀 테스트 (먼저 돌릴 것)
├── requirements.txt
├── risk/
│   ├── __init__.py        ← 빈 파일
│   ├── indicators.py
│   ├── rules.py
│   └── portfolio.py
├── data/
│   ├── __init__.py        ← 빈 파일
│   └── contract.py
├── signals/
│   ├── __init__.py        ← 빈 파일
│   └── models.py
└── eval/
    ├── __init__.py        ← 빈 파일
    └── harness.py
```

`__init__.py` 4개는 빈 파일로 만들면 된다:

```bash
mkdir -p risk data signals eval
touch risk/__init__.py data/__init__.py signals/__init__.py eval/__init__.py
```

---

## 어디서 돌릴까

### GitHub Codespaces (권장)

브라우저만 있으면 되고, 무료 티어로 충분하다. 저장소에 위 파일을 올리고 Codespace를 띄운 뒤:

```bash
pip install -r requirements.txt
python test_invariants.py
python run_real.py --check
```

`data_cache/`에 다운로드가 캐시되므로 Codespace를 껐다 켜도 저장소에 커밋해두면 재사용된다
(단, `.gitignore`에 `data_cache/`가 있으니 캐시를 유지하려면 그 줄을 빼야 한다).

### 로컬 PC

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_invariants.py
```

### Google Colab

무료지만 세션이 끊기면 파일이 날아간다. 짧게 확인만 할 때 쓴다.
파일을 업로드하거나 Google Drive를 마운트한 뒤 `!python run_real.py --check`.

---

## 실행 순서

### 0단계 — 회귀 테스트 (필수)

```bash
python test_invariants.py
```

14개 전부 PASS여야 한다. 하나라도 실패하면 그것부터 고친다.
백테스트 결과를 믿으려면 백테스터를 먼저 믿을 수 있어야 한다.

### 1단계 — 시간축 유효성 검사

```bash
python run_real.py --check
```

첫 실행은 yfinance 다운로드에 1~3분 걸린다. 이후는 캐시를 쓴다.

- **유효** → 2단계로
- **무효** → 신호에 고유 시간축이 없다는 뜻.
  `time_stop_bars`를 확정할 근거가 없으므로, 시간 손절을 끄거나(`time_stop_bars`를 매우 크게)
  신호 자체를 다른 종류로 바꾸는 것을 고려한다.

### 2단계 — time_stop_bars 확정

```bash
python run_real.py --calibrate
```

출력된 값을 `configs.py`의 `time_stop_bars`에 반영하고, "잠정" 주석을 지운다.

`⚠ 20% 기준 미달성, 최댓값 폴백`이 뜨면 관측 범위(60봉)로도 부족하다는 뜻이다.
`run_real.py`의 `horizon=60`을 90이나 120으로 올려 다시 본다.

### 3단계 — walk-forward 판정

```bash
python run_real.py --walkforward
```

통과 조건 3개가 자동으로 판정된다.

---

## 결과를 어떻게 읽을까

### 통과한 경우

축하할 일이지만 바로 실거래로 가지 않는다. 다음은 3단계 LLM 필터를 붙여
"LLM이 실제로 기여하는가"를 같은 방식으로 검증하는 것이다.
그 다음이 페이퍼 트레이딩 3개월이다.

### 미달한 경우 (이쪽이 더 가능성 높다)

**하지 말아야 할 것:** 파라미터를 바꿔가며 통과할 때까지 돌리기.
`breakout_lookback`을 55에서 20, 34, 89로 바꿔보고 제일 좋은 걸 고르는 순간
그 결과는 과적합이고, 실거래에서 재현되지 않는다.

**할 수 있는 것:**

1. **신호 종류를 바꾼다.** 55일 신고가 돌파나 RSI(3) 과매도는 널리 알려져
   초과수익이 대부분 소멸했다. 다른 축(변동성 수축 후 확장, 섹터 상대강도,
   실적 발표 후 드리프트 등)을 시도한다.
2. **출구만 남긴다.** 무작위 진입 + 좋은 출구가 신호보다 나았다면,
   그 자체가 결과다. 진입을 정교하게 만드는 대신 출구를 개선하는 쪽이 나을 수 있다.
3. **3단계로 넘어간다.** 결정론적 신호가 약해도 LLM 필터가 지뢰(실적 발표 임박,
   소송·회계 이슈)를 걸러 개선할 여지가 있다. 다만 이때도 검증 방식은 같다.

### 어느 경우든

결과는 생존 편향이 섞여 있어 실제보다 좋게 나온다. 유니버스가 "오늘의 대형주"라
2015년에 망했거나 상장폐지된 회사가 빠져 있기 때문이다. 통과선을 아슬아슬하게
넘겼다면 실제로는 미달일 가능성이 높다.

---

## 자주 걸리는 문제

**`ModuleNotFoundError: No module named 'risk'`**
`myagent/` 디렉토리 안에서 실행하고 있는지, `__init__.py` 4개가 있는지 확인.

**yfinance가 빈 데이터를 반환**
티커 오타이거나 일시적 차단이다. 티커 수를 10개로 줄여 다시 시도.

**`데이터가 짧아 walk-forward 구간을 만들 수 없습니다`**
`--start`를 더 과거로 당긴다. train 750 + test 250 = 최소 1,000봉(약 4년)이 필요하다.

**거래가 0건**
신호 조건이 빡빡하거나, 계좌가 작아 `min_notional`에 걸린 것이다.
`run_real.py` 실행 시 출력되는 거부 사유를 확인한다.
