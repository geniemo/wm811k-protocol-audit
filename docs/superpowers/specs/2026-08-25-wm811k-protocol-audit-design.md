# WM-811K 평가 프로토콜 감사 — 설계 문서

작성일: 2026-08-25 · 상태: 승인된 설계 (사용자 2026-08-25 "추천대로 진행")

## 0. 한 줄 정의

모델·학습 절차·seed 집합을 고정한 채 **평가 프로토콜(분할 방식 / 클래스 구성 / 표본 선택)만** 바꿔, 보고되는 성능이 얼마나 이동하는지 측정한다. 측정 대상은 모델이 아니라 프로토콜의 기여도다.

이 문서는 지원서·면접에 들어갈 결과를 만드는 실험의 규격이다. 여기 적힌 고정값은 실험 도중 바꾸지 않는다. 바꿔야 한다면 문서를 먼저 고치고 전체를 다시 돌린다.

## 1. 범위

**한다**
- LSWMD.pkl → 정규화된 라벨 데이터(64×64, 메타데이터) 1회 변환
- lot-disjoint gold 테스트셋 1개 고정
- 18개 프로토콜 셀 × 3 seed = 54 run (+ 보조 lot-순서 분할 3 run, + stretch 두 번째 고정 모델 9 run)
- 셀마다 두 열 보고: as-reported(자기 테스트셋) / on-gold(공통 테스트셋)
- 누수 진단(정확 중복, lot 공유, 최근접 거리), 원본 분할의 lot 겹침 검사
- 축별 기여도 분해 표, 핵심 비교 두 쌍, 대표 confusion matrix, seed 분산
- README(프로토콜 카드 포함), 면접 노트(숫자 채운 90초 요약)

**하지 않는다** (계획서 §8 그대로)
- 모델 튜닝, 데이터 증강, 백본 비교, 미라벨 데이터 사용, raw accuracy 단독 보고, "기존 연구가 틀렸다" 류 서술, 소수 클래스 단정

## 2. 데이터

### 2.1 원천
- `http://mirlab.org/dataSet/public/MIR-WM811K.zip` (329 MB, 로그인 불필요) 안의 `LSWMD.pkl`. Kaggle 사본과 동일 파일.
- 원본 필드: `waferMap`, `dieSize`, `lotName`, `waferIndex`, 학습/테스트 라벨, `failureType`. 배포본마다 표기가 다르다(Kaggle 사본은 `trianTestLabel` 오타와 `[['Training']]` 중첩 배열이 보고됨). **런타임에 컬럼명을 정규식 `tr.*test.*label`로 찾고, 찾은 이름을 로그에 남긴다.**
- 2026-08-25 MIR 배포본(`MIR-WM811K/Python/WM811K.pkl`, 2.02 GB) 확인 결과: pandas 3.0.5로 `read_pickle` 정상(8초). 컬럼 `['dieSize','failureType','lotName','trainTestLabel','waferIndex','waferMap']`. 라벨 값은 문자열(`'Training'`/`'Test'`, `'none'`/결함명), **미라벨은 정수 `0`**. `waferMap`은 uint8, 값 {0,1,2}. `lotName` 결측 0건(singleton 로직은 유지).
- 언랩 규칙: 문자열이면 그대로, 중첩 배열/리스트면 첫 원소를 재귀 언랩, 그 외(정수 0, 빈 배열, NaN)는 미라벨. 테스트로 고정.
- **원본 분할 사실(확인)**: 라벨 172,950 = Training 54,355 + Test 118,595 (통념과 반대로 Test가 더 큼). **lot이 두 집합에 걸치는 경우 0건 — 원본 분할은 lot-disjoint다.** 클래스 비율은 두 집합에서 다르다(Training 결함 약 32%, Test 결함 약 6.7%) — 층화되지 않았다. 따라서 A1은 "lot-disjoint + 비층화 + 31/69"인 프로토콜이다.
- **원본 분할의 lot 순서 구조(확인)**: Training lot 번호 1~46,729(중앙값 36,921), Test lot 번호 40,328~47,542(중앙값 44,422). 대체로 앞 lot이 Training, 뒤 lot이 Test이며 40,328~46,729 구간에서만 교차(순서상 구간 16개). 즉 원저자 분할은 유사-시간 분할에 가깝다. A4(보조)는 이 성질을 층화·비율 교란 없이 재현하는 통제 조건이다.
- **lot 구조(확인)**: 라벨 lot 10,762개, lot당 라벨 웨이퍼 중앙값 23(라벨링이 lot 단위로 이뤄짐). 결함 웨이퍼 25,519 중 21,141(83%)이 "같은 lot에 결함 웨이퍼가 2장 이상"인 lot에 속하고, 그런 lot의 51.5%는 결함 클래스가 하나뿐이다 → 랜덤 분할 시 형제 웨이퍼 누수가 성립할 구조.
- **정확 중복(확인)**: 라벨 172,950 중 6,502행(3.76%)이 중복 그룹(3,250개)에 속하며 그중 6,304행이 none. 중복 그룹의 99.6%가 여러 lot에 걸침(전부 양품인 동일 레이아웃 맵으로 추정) → lot-disjoint 분할에서도 none 중복은 남는다. 진단은 전체와 결함 클래스 한정(`dup_rate_defect`)을 따로 보고한다.
- 맵 크기(라벨 기준): 고유 shape 346개, 중앙값 33×33, H 또는 W가 64를 넘는 맵은 2.0% → 거의 전부 업샘플.

### 2.2 1회 변환 (`scripts/convert_data.py`) → `data/processed/`
- 라벨된 행만 추출 (failureType 비어 있지 않음). 기대치 172,950 — 실제 수는 직접 세서 `data/processed/summary.json`에 기록.
- `labeled_maps64.npy`: uint8 `[N, 64, 64]`, 값 ∈ {0, 1, 2}
- `labeled_meta.parquet`: `row_id, lot_name, lot_id, lot_num, is_singleton_lot, wafer_index, die_size, orig_h, orig_w, failure_type, label9, orig_split, raw_hash, map64_hash`
- 클래스 인코딩(고정 순서): `Center=0, Donut=1, Edge-Loc=2, Edge-Ring=3, Loc=4, Random=5, Scratch=6, Near-full=7, none=8`. 결함 클래스 = 0..7.
- `lot_id`: `lotName` 문자열. 결측·빈 문자열·NaN이면 `__singleton_<row_id>` (각 표본이 단독 그룹). `is_singleton_lot=True`.
- `lot_num`: `lotName`에서 정수 파싱(`lot12345` → 12345). 파싱 불가면 결측.
- `raw_hash`: 원본 맵의 `shape` + 바이트의 sha1. `map64_hash`: 64×64 맵의 sha1.
- 미라벨 행은 변환하지 않는다 (확장 1은 별도 작업).

### 2.3 리사이즈 (고정, 이후 수정 금지)
- 중심 정렬 nearest-neighbour: 출력 좌표 `i`(0..63)에 대해 원본 인덱스 `floor((i + 0.5) * H / 64)`, `H-1`로 클립. 열도 동일.
- 종횡비 보존 없음 (선행 문헌과 동일한 단순 리사이즈). 원본 크기는 메타에 남겨 분석에 쓴다.
- 이유: 픽셀값 0/1/2는 이산 의미(die 없음/정상/불량). 보간은 존재하지 않는 중간값을 만든다.
- 테스트: 출력값 ⊂ {0,1,2}; 64×64 입력은 항등; 정수배 업샘플은 블록 복제; 다운샘플은 원본에 있던 값만 등장.

### 2.4 입력 인코딩 (고정)
- 배치 시점에 GPU에서 3채널 one-hot(`[B, 3, 64, 64]`, float32). 스칼라 1채널은 die 없음 < 정상 < 불량이라는 서열을 강제하므로 쓰지 않는다.

## 3. 실험 설계

### 3.1 통제변수 (고정값)

| 항목 | 값 |
|---|---|
| 모델 | `SmallCNN`: conv3×3(3→32)-BN-ReLU-MaxPool2 → conv(32→64)-BN-ReLU-MaxPool2 → conv(64→128)-BN-ReLU-MaxPool2 → conv(128→128)-BN-ReLU → GlobalAvgPool → Dropout(0.3) → Linear(128→n_classes). 약 0.3M 파라미터 |
| 옵티마이저 | AdamW, lr 1e-3, weight_decay 1e-4, betas 기본값 |
| 스케줄 | cosine decay to 0, warmup 없음 |
| 배치 | 256, 복원 없는 셔플, 에폭 경계 넘어 스텝 카운트 |
| 학습량 | **8,000 gradient steps** (에폭이 아님) |
| 손실 | 가중치 없는 cross-entropy |
| 검증셋 / early stopping | 없음. 마지막 가중치를 평가 |
| 증강 | 없음 |
| 입력 | 64×64, nearest, 3채널 one-hot |
| seed | {0, 1, 2}. python/numpy/torch 시드, `cudnn.benchmark=False`, deterministic 최선 노력 |
| 정밀도 | float32 (AMP 없음) |

왜 step 고정인가: 에폭 고정은 표본 규모와 교락된다(C3에서는 1 에폭 ≈ 4 step). 동일 연산량이 비교 단위로 가장 단순하다. 1,000 step마다 own-test/gold 지표를 로그하지만 **선택에 쓰지 않는다** (진단용 학습 곡선).

### 3.2 gold 테스트셋 (실험 전체에서 하나)
- 라벨 전체(N≈172,950)에 대해 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260825)`, `y=label9`, `groups=lot_id`. **fold 0의 test 인덱스 = gold** (≈20%), 나머지 = **pool**.
- gold는 어떤 셀의 학습·분할·cap 추출에도 쓰이지 않는다. 모든 셀은 pool 안에서만 산다.
- `gold_defect` = gold 중 `label9 ≠ 8`. 모든 셀의 공통 평가 집합.
- 검증: gold와 pool의 `lot_id` 교집합 = ∅ (singleton 포함); gold에 9개 클래스 모두 존재; 각 클래스의 gold 비중이 15~25% 범위.
- gold 인덱스는 `data/processed/gold_indices.npy`로 저장하고 해시를 README에 기록한다.

### 3.3 조작변수

**축 A — 분할 (pool 내부)**
- `A1 original`: `orig_split=='Training'` → train, `orig_split=='Test'` → own-test. 비율은 원본 그대로. 둘 다 아닌 라벨 행이 있으면 A1에서 제외하고 개수를 기록. seed는 초기화와 cap 추출에만 영향.
- `A2 random`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`, fold 0 test (20%).
- `A3 lot`: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, `groups=lot_id`, fold 0 test.
- `A4 lot-ordered` (보조, B1×C1에서만): `lot_num` 결측 행 제외, lot_num 오름차순으로 lot을 정렬, 누적 웨이퍼 수 기준 뒤 20%에 해당하는 lot들을 test. lot 경계 유지. 분할은 seed 무관(초기화만 변동). lotName 번호가 시간순이라는 문서 근거는 없으므로 "유사-시간 분할"로만 표기.

**축 B — 클래스 구성**
- `B1 9cls`: 전부.
- `B2 8cls`: `label9==8`(none) 행을 pool에서 제거한 뒤 동일 파이프라인. 모델 출력 8.

**축 C — 클래스당 cap (표본 선택 규칙)**
- `C1 full`: cap 없음.
- `C2 cap5000`: 클래스당 최대 5,000장.
- `C3 balanced`: cap = (축 B 적용 후) pool 내 최소 클래스 표본 수. 실제 값은 데이터에서 계산해 기록 (Near-full 기준 ≈ 120 예상).
- cap 초과 클래스는 `np.random.default_rng(seed)`로 비복원 추출. **cap은 분할 전에 pool 전체에 적용**한다 (관행대로 "서브셋을 만든 뒤 나눈다"). A1도 cap 후 남은 행의 원본 라벨로 나눈다.

**적용 순서 (고정)**: pool → B 클래스 필터 → C cap(seed) → A 분할(seed) → 학습(seed) → 평가.

**셀 수**: 3×2×3 = 18, cell_id = `A{1,2,3}-B{1,2}-C{1,2,3}`. run_id = `cell_id-s{seed}`. 본 실험 54 run + 보조 `A4-B1-C1` 3 run. Stretch: `SmallCNN` 대신 `ResNet18Adapted`(torchvision resnet18, conv1을 3ch·3×3·stride1로, maxpool 제거, 동일 하이퍼파라미터)로 `A2-B1-C1`, `A3-B1-C1`, `A3-B1-C3` × 3 seed = 9 run. stretch는 본 실험과 결과 파일을 분리한다(`model` 컬럼).

### 3.4 평가 (run당)

| 집합 | 대상 셀 | 지표 |
|---|---|---|
| **own-test** (as-reported) | 전부 | 셀의 클래스 집합으로 macro-F1, balanced accuracy, accuracy, 클래스별 F1/precision/recall, confusion matrix |
| **gold-defect** (common) | 전부 | 결함 8클래스 평균 F1(`defect_f1`), 8클래스 평균 recall(`defect_bacc`), 클래스별 F1. B1 모델이 none(8)으로 예측한 결함 표본은 해당 클래스의 FN이며 어떤 결함 클래스의 FP도 아니다. confusion은 B1: 8×9, B2: 8×8 |
| **gold-full** | B1만 | 9클래스 macro-F1, balanced accuracy, none F1, confusion 9×9 |

- **대표값**: `A3-B1-C1`의 gold-full macro-F1, 3 seed mean±std. 배포 상황에 가장 가깝다.
- 모든 지표는 sklearn 정의(`f1_score(average='macro')`, `balanced_accuracy_score`)와 일치해야 하며 테스트로 고정한다.

### 3.5 누수 진단 (분할당 1회, 모델 무관)
- `dup_rate`: own-test 중 train에 `raw_hash`가 같은 표본이 있는 비율. `dup_rate64`: `map64_hash` 기준. `dup_rate_defect`: own-test의 결함 표본만 대상으로 같은 계산.
- `lot_share_rate`: own-test 중 자기 lot의 다른 웨이퍼가 train에 있는 비율 (A3에서 0이어야 함 — 테스트로 강제).
- `nn_hamming`: own-test 각 표본에서 train 최근접 표본까지의 64×64 해밍 거리. one-hot 평탄화 후 GPU 행렬곱으로 `4096 − max(dot)`. 중앙값·10/25/75/90 분위수 저장, 히스토그램용 raw 값은 run 디렉토리에 npy.
- 전역(EDA 1회): 원본 Training/Test 간 lot 겹침(겹치는 lot 수, Test 웨이퍼 중 lot이 Training에도 있는 비율); 전체 정확 중복률(원본/64×64), 중복이 같은 lot 내인지 비율; lot당 라벨 웨이퍼 수 분포; lot 내 라벨 동질성(한 lot에 결함 라벨이 2장 이상일 때 같은 클래스인 비율); 맵 크기 분포와 64보다 큰 맵(다운샘플) 비율; 클래스별 개수.

## 4. 실행·저장·재개
- 진입점: `python -m wm811k_audit.run --cells all --seeds 0 1 2` (`--cells A3-B1-C1 ...`, `--model resnet18`, `--aux-a4` 옵션).
- run 디렉토리 `results/runs/<run_id>/`: `config.json`, `metrics.json`(3.4 지표 전부 + confusion + 3.5 진단), `train_log.csv`(1,000 step마다 loss·own-test·gold 지표), `nn_hamming.npy`. 가중치는 저장하지 않는다(재현은 seed로).
- `results/results.csv`: 1행 = 1 run, 요약 지표 평탄화. 실행 시작 시 이미 있는 run_id는 건너뛴다(조합 단위 재개).
- 데이터는 run 시작 시 GPU에 uint8로 상주(≈0.7 GB). DataLoader 없이 인덱싱.
- 예상 시간: run당 1~2분 → 본 실험 2시간 이내.

## 5. 분석·산출물 (`python -m wm811k_audit.analyze`)
1. **표 1**: 18셀 × {as-reported macro-F1, gold defect_f1, (B1) gold-full macro-F1} mean±std.
2. **표 2 축별 기여도**: 지표별로 각 축의 수준 평균과 범위(max−min), 가산 모형 잔차로 상호작용 크기. as-reported와 gold 각각.
3. **핵심 쌍 ①** `A2-B1-C1` vs `A3-B1-C1`: as-reported 격차; `A2` 모델 내부 (own-test − gold-defect) 격차 = 같은 가중치로 측정한 누수 부풀림. `A3`는 own ≈ gold여야 함(sanity). 보조 `A4-B1-C1`은 같은 표에 참고 행으로 붙인다.
4. **핵심 쌍 ②** `A3-B1-C1` vs `A3-B1-C3`: as-reported와 gold의 격차 비교 → 표본 선택 효과를 학습 데이터 효과와 테스트셋 효과로 분리.
5. **seed 분산 대 셀 격차**: 각 격차를 seed std와 함께 제시. 격차 < 2·std면 "노이즈와 구분 불가"로 서술.
6. **그림**: 핵심 쌍 막대(seed 오차막대) 2장, 축 주효과 플롯, `A3-B1-C1` gold-full confusion matrix, A2 vs A3 최근접 해밍 히스토그램, 진단 표(dup_rate, lot_share_rate).
7. **README**: 프로토콜 카드(분할·클래스·cap·리사이즈·인코딩·학습량·지표 전부 명시), 결과 표·그림, 재현 명령, 한계.
8. **`docs/interview_notes.md`**(gitignore, 개인용): 숫자 채운 90초 요약, 예상 질문 답, 말하지 말아야 할 것.

## 6. 테스트 전략
- 합성 fixture(`tests/conftest.py`): 수백 행짜리 DataFrame — 다양한 맵 크기(6×21, 26×26, 64×64, 120×80), 9클래스, lot 크기 1~25, lotName 결측, 원본 split 라벨, 의도적 정확 중복 쌍(같은 lot / 다른 lot).
- 단위: 라벨 컬럼 탐지·언랩, 리사이즈(2.3 성질), one-hot, lot_id/singleton, 해시, gold carve(lot-disjoint·전 클래스 존재), A1/A2/A3/A4 성질(A3·A4 lot-disjoint, A1 라벨 준수, A2 층화), cap 규칙(≤cap, seed 재현, C3 균등), 지표(sklearn과 일치, none 예측 처리), 진단(dup/lot_share/nn_hamming의 brute-force 대조), 재개 로직, 모델 출력 shape, 학습 루프 스모크(CPU, 20 step, 손실 감소).
- 통합: fixture로 전체 파이프라인 CPU 실행 → results.csv 행 생성.
- 실데이터 검증(변환 직후 1회): 행 수·클래스 수·값 집합·gold 성질을 `summary.json`에 기록하고 assert.

## 7. 시나리오별 해석 (계획서 §5 유지)

| 시나리오 | 해석 |
|---|---|
| A3에서 뚜렷한 하락, A2의 own−gold 격차 큼 | 누수 실재. 검증 조건과 배포 조건의 괴리를 정량화 |
| 하락 미미 | 이 데이터셋에서 lot 상관이 약함. 보고된 성능 차이는 실제 모델 차이일 가능성 높음 |
| C 효과 > A 효과 | 표본 선택이 1순위 교란 요인 |
| 상호작용 큼 | 가산이 아님 — 조합 실험이라서 보이는 결과 |
| A1(원본) ≠ A3(lot) | 둘 다 lot-disjoint이므로 격차는 누수가 아니라 클래스 사전확률 이동(비층화)과 학습량(31%) 때문. as-reported와 gold 두 열이 이를 가른다 |

## 8. 한계·방어선
- lot 분할도 배포 조건의 상한이 아니다(신제품·새 레이아웃 변화는 반영 못 함).
- 패턴–원인 매핑은 공개 문헌 일반론이지 특정 팹 지식이 아니다.
- Donut(555)·Near-full(149)은 gold에 수십 장뿐 — 클래스별 F1을 단정하지 않는다.
- 선행연구 검색은 웹·arXiv·GitHub 범위. "없다"가 아니라 "확인 못 했다".
- 원본 Training/Test 라벨의 분할 기준은 공개 자료에 없다. "원본 제공 분할"로만 표기.

## 9. 저장소 구조
```
src/wm811k_audit/   data.py preprocess.py splits.py model.py train.py metrics.py diagnostics.py run.py analyze.py
scripts/            convert_data.py eda.py
tests/              conftest.py test_*.py
data/               raw/ processed/   (gitignore)
results/            results.csv runs/ figures/ tables/
docs/               superpowers/specs, superpowers/plans, interview_notes.md(gitignore)
```
