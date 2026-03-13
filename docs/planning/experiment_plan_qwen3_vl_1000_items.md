# Experiment Plan: Qwen3-VL Retrieval Evaluation on 1000 Similar Products

## 0. Purpose

이 문서는 1000개 제품 데이터셋을 이용해 새 검색 파이프라인을 검증하기 위한 실험 계획서다.

핵심 질문은 다음과 같다.

1. OCR 없이도 `Qwen3-VL` 중심 파이프라인이 충분한 검색 품질을 낼 수 있는가?
2. `BGE-M3`를 유지한 텍스트 채널이 실제로 도움이 되는가?
3. `Qwen3-VL-Reranker-2B`가 near-duplicate 구분에 의미 있게 기여하는가?

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

---

## 3. Evaluation Questions

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

---

## 4. Metrics

### Primary Metrics

- Accuracy@1
- Accuracy@5
- Recall@5
- MRR

### Secondary Metrics

- exact maker match rate
- exact part_number match rate
- reranker lift on Top-1 / Top-5
- 평균 latency
- p50 / p90 / p95 latency

### Qualitative Review

- 실패 사례 taxonomy
- visually similar confusion 사례
- text omission / hallucinated description 사례

---

## 5. Experiment Procedure

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

---

## 6. Hypotheses

### H1

`Qwen3-VL image embedding + reranker + BGE-M3 text` 구성은 기존 BGE-VL 기반 구조보다 near-duplicate 구분에서 더 나은 Top-5 품질을 보일 가능성이 높다.

### H2

OCR을 기본 경로에서 제거해도 깔끔한 단일 제품 사진 조건에서는 retrieval quality가 크게 떨어지지 않을 수 있다.

### H3

복잡한 배경, 작은 라벨, 여러 물체가 섞인 이미지에서는 OCR이 오히려 false evidence를 추가해 retrieval을 악화시킬 수 있다.

### H4

`Qwen3-VL-Reranker-2B`는 top candidate 재정렬에서 명확한 이득을 줄 것이다.

---

## 7. Deliverables

실험 완료 시 산출물은 다음과 같다.

- config별 metrics table
- top-1/top-5 비교표
- latency summary table
- 실패 사례 이미지 모음
- qualitative error taxonomy
- 최종 채택안 recommendation

---

## 8. Decision Criteria

다음 기준으로 기본 파이프라인을 선택한다.

- Accuracy@5가 가장 중요
- Accuracy@1과 exact identifier match를 함께 고려
- latency가 실사용 가능 범위에 있어야 함
- OCR이 추가 complexity 대비 의미 있는 개선을 보여야만 기본 경로에 남긴다

즉:

- OCR on이 quality 이득이 작고 latency/오염 리스크가 크면 제거
- OCR on이 exact part_number retrieval에서 분명한 이득을 보이면 optional expert mode로 유지

---

## 9. Implementation Work Needed Before Running

- configuration별 Milvus collection naming 분리
- evaluation runner script 작성
- result logging schema 정의
- metadata freeze CSV 작성
- test set 100개 ground-truth 검증

---

## 10. Immediate Next Step

가장 먼저 할 일은 아래 두 가지다.

1. 실험용 1000개 데이터셋 메타 CSV를 고정한다.
2. configuration별 indexing/search evaluation 스크립트 요구사항을 별도 작업 항목으로 분해한다.
