# Experiment Plan: Qwen3-VL Retrieval Evaluation on 1000 Similar Products

## 0. Purpose

이 문서는 1000개 제품 데이터셋을 이용해 새 검색 파이프라인을 검증하기 위한 실험 계획서다.

핵심 질문은 다음과 같다.

1. OCR 없이도 `Qwen3-VL` 중심 파이프라인이 충분한 검색 품질을 낼 수 있는가?
2. `BGE-M3`를 유지한 텍스트 채널이 실제로 도움이 되는가?
3. `Qwen3-VL-Reranker-2B`가 near-duplicate 구분에 의미 있게 기여하는가?

이 계획서는 단순한 retrieval 비교만이 아니라, 최종보고서의 목표(`O1`~`O5`)를 실제로 뒷받침할 수 있는 실험 묶음을 정의하는 것을 목적으로 한다.

---

## 0.1 Objective Coverage

보고서 목표와 실험 매핑은 다음과 같다.

- `O1. Working MVP workflow`
  - E0. End-to-end scenario validation
- `O2. Retrieval effectiveness`
  - E1. 1000-item fixed-split retrieval ablation
- `O3. OCR robustness`
  - E2. OCR identifier benchmark
- `O4. Interactive feasibility`
  - E3. Latency and resource benchmark
- `O5. User usefulness and trust`
  - E4. Human-review pilot study

추가로 보고서의 평가 장에서 강조하는 engineering reliability를 위해 다음을 보조 실험으로 둔다.

- E5. Safety and reliability validation

---

## 1. Dataset Design

### 1.1 Dataset Size

- 전체 제품 수: 1000
- Index set: 900
- Test/query set: 100

### 1.2 Dataset Characteristics

- 비슷하게 생긴 상품군 중심으로 구성
- 가능한 한 visually similar but textually different 사례를 포함
- maker/model/part number가 혼재한 실제형 이미지 우선

### 1.3 Data Fields

각 제품은 최소 다음 정보를 갖는다.

- `model_id`
- `image_path`
- `maker`
- `part_number`
- `category`
- `description`
- 정답 매칭용 canonical label

### 1.4 Split Rule

- 900/100 split은 무작위가 아니라, category와 near-duplicate group을 고려해 분할
- 테스트셋에는 index셋과 visually similar한 항목이 반드시 존재해야 함
- 동일 제품의 중복 이미지는 index/test 양쪽으로 누수되지 않도록 제어

---

## 2. Compared Configurations

### C1. Current Mixed Qwen Pipeline

- image embedding: `Qwen3-VL-Embedding-2B`
- product info extraction: `Qwen3-VL-2B-Instruct`
- caption: `Qwen3-VL-2B-Instruct`
- text embedding: `BGE-M3`
- reranker: `Qwen3-VL-Reranker-2B`
- OCR: 없음

### C2. Mixed Qwen Pipeline + OCR

- image embedding: `Qwen3-VL-Embedding-2B`
- OCR enabled
- product info extraction: `Qwen3-VL-2B-Instruct`
- caption: `Qwen3-VL-2B-Instruct`
- text embedding: `BGE-M3`
- reranker: `Qwen3-VL-Reranker-2B`

### C3. Ablation Without Reranker

- C1과 동일
- reranker만 제거

### C4. Text-Light Ablation

- image embedding only + reranker
- text channel 최소화

목적:

- image-only vs hybrid
- OCR on/off
- reranker on/off
- text-channel contribution

### Important Note on Current Code State

- 현재 코드베이스 기본 동작은 사실상 `OCR enabled baseline`에 가깝다.
- 따라서 공정한 `C1` vs `C2` 비교를 위해서는 OCR on/off 스위치를 먼저 구현해야 한다.
- 특히 `C1`은 query-time OCR만 끄는 것으로 충분하지 않고, OCR 없이 별도 컬렉션에 다시 인덱싱해야 한다.
- 즉 `exp_c1_*`와 `exp_c2_*`는 서로 독립적으로 구축해야 한다.

---

## 3. Experiment Tracks

### E0. End-to-End Scenario Validation

목적:

- `O1`을 방어하기 위한 기능 검증

시나리오:

1. 이미지 업로드
2. 인덱싱
3. 검색
4. 사용자 확인
5. 승인 후 저장
6. 이후 재검색 시 캐시/저장 결과 재사용 확인

기록:

- 단계별 성공 여부
- 실패 지점
- operator action 수
- evidence 표시 여부

### E1. Retrieval Effectiveness Ablation

목적:

- `O2` 충족 여부 판단
- `C1`~`C4` 비교

핵심 출력:

- Accuracy@1
- Accuracy@5
- Recall@5
- MRR
- exact identifier hit rate

### E2. OCR Identifier Benchmark

목적:

- `O3` 충족 여부 판단
- OCR을 기본 경로에서 제거해도 되는지 정량적으로 판단

데이터:

- 1000개 전체 중 identifier가 실제로 보이는 샘플을 별도 subset으로 선정
- 권장 규모: 150~250장
- 각 샘플에 대해 ground-truth identifier 문자열을 수동 정답으로 고정

조건:

- clean single-object
- blur / glare
- cluttered background
- multi-object / partial view
- small label / far distance

측정:

- CER
- exact full-string match rate
- part_number token recall
- maker token recall

비교:

- PaddleOCR path
- Qwen3-VL extracted identifier text
- OCR + Qwen merged evidence

### E3. Latency and Resource Benchmark

목적:

- `O4` 충족 여부 판단

측정 대상:

- indexing latency
- query latency
- preprocessing / image search / OCR search / caption search / text search / rerank / total
- GPU memory peak
- cold start vs warm start

방법:

- index set 구축 시 900개 전체에 대한 평균 및 p50/p90/p95 측정
- test query 100개를 최소 3회 반복 실행
- 첫 회는 cold-ish run, 이후는 warm run으로 구분

### E4. Human-Review Pilot Study

목적:

- `O5` 충족 여부 판단
- shortlist + evidence가 실제로 listing effort를 줄이는지 확인

권장 설계:

- 소규모 pilot: 5~10명 또는 최소 1명의 도메인 익숙한 operator 반복 실험
- baseline: 수동 검색/수동 작성
- assisted: Smart Vision shortlist + evidence 사용

측정:

- task completion time
- final answer correctness
- number of manual edits
- number of external searches
- self-reported confidence / trust

### E5. Safety and Reliability Validation

목적:

- engineering reliability 근거 보강

측정:

- pytest regression pass
- writeback default off 확인
- explicit confirmation 없을 때 저장 불가 확인
- config별 evaluation runner 재현 가능 여부

---

## 4. Evaluation Questions

### Q1. Retrieval Quality

- 정답 제품이 Top-1에 오는가?
- 정답 제품이 Top-5에 오는가?
- visually similar product 간 순위 구분이 되는가?

### Q2. Text Sensitivity

- maker / part_number / short model string이 검색 품질에 얼마나 기여하는가?
- OCR 없이도 instruct+caption 기반 text embedding이 충분한가?

### Q3. Reranker Effect

- reranker가 Top-5를 개선하는가?
- reranker가 false positive를 줄이는가?

### Q4. Operational Cost

- indexing latency
- query latency
- GPU memory footprint

### Q5. Human Review Value

- shortlist와 evidence가 실제 확인 시간을 줄이는가?
- 사용자가 결과를 더 신뢰하는가?
- 승인 전 수정이 필요한 빈도는 어느 정도인가?

### Q6. OCR Decision

- OCR은 exact identifier 확보에 실제로 얼마나 기여하는가?
- OCR이 clutter/noise 조건에서 false evidence를 얼마나 늘리는가?

---

## 5. Metrics

### Primary Metrics

- Accuracy@1
- Accuracy@5
- Recall@5
- MRR
- CER
- exact full-string identifier match rate

### Secondary Metrics

- exact maker match rate
- exact part_number match rate
- reranker lift on Top-1 / Top-5
- 평균 latency
- p50 / p90 / p95 latency
- cold/warm latency gap
- GPU memory peak
- task completion time
- edit count
- external search count
- user confidence / trust score

### Qualitative Review

- 실패 사례 taxonomy
- visually similar confusion 사례
- text omission / hallucinated description 사례
- OCR contamination 사례
- human-review correction pattern

---

## 6. Experiment Procedure

### Step 1. Dataset Freeze

- 1000개 제품 목록 확정
- metadata 정규화
- train/index/test leakage 점검

### Step 2. Index Build

- index set 900개를 각 configuration 별로 별도 컬렉션에 인덱싱
- 컬렉션명은 config별 suffix를 붙여 분리

예:

- `exp_c1_image_parts`
- `exp_c1_model_texts`
- `exp_c2_image_parts`

### Step 3. Query Run

- test set 100개를 동일한 평가 스크립트로 질의
- image-only query
- image + text query
- 필요 시 text-only query 보조 측정

### Step 4. Result Logging

- rank list
- score decomposition
- reranker score
- latency breakdown
- final predicted item

### Step 5. Error Analysis

- Top-1 실패 / Top-5 성공
- Top-5 실패
- visually similar mismatch
- text evidence missing
- OCR noise contamination

### Step 6. OCR Benchmark Run

- identifier subset에 대해 OCR/Qwen 추출 문자열 수집
- ground-truth 문자열과 CER/exact match 계산
- 이미지 조건별 버킷 통계 산출

### Step 7. Latency Benchmark Run

- 900개 인덱싱 과정의 단계별 latency 수집
- 100개 query를 반복 실행하여 p50/p90/p95 집계
- cold/warm 및 config별 리소스 차이 비교

### Step 8. Human-Review Pilot

- 동일한 샘플 태스크를 baseline/assisted 조건으로 수행
- 시간, 수정 수, 외부 검색 수, confidence를 기록

### Step 9. Reliability Validation

- pytest / smoke / writeback safety 검증
- 평가 스크립트 재실행성 점검

---

## 7. Hypotheses

### H1

`Qwen3-VL image embedding + reranker + BGE-M3 text` 구성은 기존 BGE-VL 기반 구조보다 near-duplicate 구분에서 더 나은 Top-5 품질을 보일 가능성이 높다.

### H2

OCR을 기본 경로에서 제거해도 깔끔한 단일 제품 사진 조건에서는 retrieval quality가 크게 떨어지지 않을 수 있다.

### H3

복잡한 배경, 작은 라벨, 여러 물체가 섞인 이미지에서는 OCR이 오히려 false evidence를 추가해 retrieval을 악화시킬 수 있다.

### H4

`Qwen3-VL-Reranker-2B`는 top candidate 재정렬에서 명확한 이득을 줄 것이다.

### H5

OCR은 clean single-object 조건에서는 exact identifier 확보에 이득을 주지만, cluttered scene에서는 false evidence를 늘릴 가능성이 높다.

### H6

shortlist + evidence 기반 assisted workflow는 완전 수동 검색보다 task time과 외부 검색 횟수를 줄일 것이다.

---

## 8. Deliverables

실험 완료 시 산출물은 다음과 같다.

- config별 metrics table
- top-1/top-5 비교표
- OCR CER / exact match 표
- latency summary table
- human-review pilot 결과표
- reliability check summary
- 실패 사례 이미지 모음
- qualitative error taxonomy
- 최종 채택안 recommendation

---

## 9. Decision Criteria

다음 기준으로 기본 파이프라인을 선택한다.

- Accuracy@5가 가장 중요
- exact identifier match 성능이 실사용에 충분해야 함
- Accuracy@1과 exact identifier match를 함께 고려
- latency가 실사용 가능 범위에 있어야 함
- OCR이 추가 complexity 대비 의미 있는 개선을 보여야만 기본 경로에 남긴다
- assisted workflow가 수동 baseline보다 의미 있는 시간 절감을 보여야 함

즉:

- OCR on이 quality 이득이 작고 latency/오염 리스크가 크면 제거
- OCR on이 exact part_number retrieval에서 분명한 이득을 보이면 optional expert mode로 유지

---

## 10. Implementation Work Needed Before Running

- OCR on/off 스위치 구현 (`index-time`, `query-time`)
- config별 Milvus collection naming 분리
- evaluation runner script 작성
- OCR benchmark ground-truth schema 정의
- latency aggregation script 작성
- human-review pilot task sheet 작성
- safety/writeback smoke test checklist 작성
- result logging schema 정의
- metadata freeze CSV 작성
- test set 100개 ground-truth 검증

---

## 11. Immediate Next Step

가장 먼저 할 일은 아래 세 가지다.

1. OCR on/off와 config별 컬렉션 분리를 먼저 구현한다.
2. 실험용 1000개 데이터셋 메타 CSV와 identifier subset 정답을 고정한다.
3. retrieval / OCR / latency / human-review 평가 스크립트 요구사항을 별도 작업 항목으로 분해한다.
