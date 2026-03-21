# 실험 실행 가이드

이 문서는 현재 준비된 데이터셋과 평가 입력을 사용해 실험을 실제로 수행하는 방법을 단계별로 정리한 실행 가이드다.

## 1. 현재 준비 상태

이미 준비된 항목:

- 통합 데이터셋: [`data/datasets/unified_v1`](/Users/mac/project/Smart_vision/data/datasets/unified_v1)
- 전체 manifest: [`unified_all.jsonl`](/Users/mac/project/Smart_vision/data/datasets/unified_v1/unified_all.jsonl)
- train/test split:
  - [`train.jsonl`](/Users/mac/project/Smart_vision/data/datasets/unified_v1/train.jsonl)
  - [`test.jsonl`](/Users/mac/project/Smart_vision/data/datasets/unified_v1/test.jsonl)
- retrieval 평가 입력:
  - [`index_manifest.jsonl`](/Users/mac/project/Smart_vision/data/datasets/unified_v1/eval_v1/index_manifest.jsonl)
  - [`query_manifest.jsonl`](/Users/mac/project/Smart_vision/data/datasets/unified_v1/eval_v1/query_manifest.jsonl)
- metric 계산 스크립트:
  - [`prepare_retrieval_eval_inputs.py`](/Users/mac/project/Smart_vision/data/scripts/prepare_retrieval_eval_inputs.py)
  - [`evaluate_retrieval_predictions.py`](/Users/mac/project/Smart_vision/data/scripts/evaluate_retrieval_predictions.py)

## 2. 실험 우선순위

마감 기준 최소 우선순위:

1. E1 Retrieval effectiveness
2. E3 Latency benchmark
3. E2 OCR benchmark
4. E4 Usability pilot

시간이 부족하면 `E1 + E3`는 반드시 끝내고, `E2`와 `E4`는 가능한 범위에서 수행한다.

## 3. E1 Retrieval Effectiveness 실행 절차

### 3.1 목적

서로 다른 검색 구성을 비교해 shortlist 품질을 측정한다.

### 3.2 비교 구성

- `C1`: OCR off, reranker on, text channel on
- `C2`: OCR on, reranker on, text channel on
- `C3`: OCR off, reranker off
- `C4`: OCR off, text-light

### 3.3 준비물

- GPU 또는 안정적인 실행 환경
- Milvus 실행
- 각 구성별 독립 컬렉션 이름

예:

```env
HYBRID_IMAGE_COLLECTION=exp_c1_image_parts
HYBRID_TEXT_COLLECTION=exp_c1_text_parts
HYBRID_ATTRS_COLLECTION=exp_c1_attrs_parts
HYBRID_MODEL_COLLECTION=exp_c1_model_texts
HYBRID_CAPTION_COLLECTION=exp_c1_caption_parts
ENABLE_OCR=0
```

### 3.4 절차

1. `C1` 환경변수로 설정
2. `index_manifest.jsonl`의 모든 항목을 인덱싱
3. `query_manifest.jsonl`의 모든 항목을 질의
4. 각 query에 대해 Top-K 결과를 `predictions_c1.jsonl`로 저장
5. 같은 방식으로 `C2`, `C3`, `C4` 반복

### 3.5 저장해야 할 예측 형식

각 query당 최소 다음 필드가 필요하다.

```json
{
  "query_id": "auto_gparts_cat40_000001",
  "predicted_ids": [
    "auto_gparts_cat40_000102",
    "auto_gparts_cat50_000044",
    "semi_1090481_000031"
  ]
}
```

### 3.6 metric 계산

```bash
python3 data/scripts/evaluate_retrieval_predictions.py \
  --queries data/datasets/unified_v1/eval_v1/query_manifest.jsonl \
  --index data/datasets/unified_v1/eval_v1/index_manifest.jsonl \
  --predictions predictions_c1.jsonl
```

구성별로 반복한다.

### 3.7 산출물

- `Accuracy@1`
- `Accuracy@5`
- `Recall@5`
- `MRR`
- exact identifier hit rate

## 4. E2 OCR Benchmark 실행 절차

### 4.1 목적

OCR을 유지해야 하는지 정량적으로 판단한다.

### 4.2 subset 만들기

전체 query 중 identifier가 실제로 보이는 샘플 `150~250개`를 선정한다.

조건을 고르게 섞는다.

- clean single-object
- blur / glare
- cluttered background
- small label
- partial occlusion

### 4.3 정답 만들기

각 샘플에 대해 사람이 직접 다음 정답을 적는다.

- `ground_truth_identifier`
- `ground_truth_part_number`
- `ground_truth_maker`

CSV나 JSONL로 보관한다.

### 4.4 실행

조건별로 OCR 결과를 수집한다.

- PaddleOCR path
- Qwen-based extracted identifier text
- OCR + Qwen merged evidence

### 4.5 산출물

- CER
- exact full-string match rate
- part number recall
- maker recall

## 5. E3 Latency Benchmark 실행 절차

### 5.1 목적

interactive support 수준의 응답성을 갖는지 확인한다.

### 5.2 측정 대상

- indexing latency
- query latency
- preprocessing time
- embedding time
- retrieval time
- rerank time
- total time

### 5.3 절차

1. 동일 환경에서 query set 299개를 최소 3회 반복
2. 첫 회는 cold run으로 기록
3. 이후 2회 이상은 warm run으로 기록
4. 모든 요청의 latency를 CSV 또는 JSONL로 저장

### 5.4 산출물

- mean
- p50
- p90
- p95

## 6. E4 Usability Pilot 실행 절차

### 6.1 참가자

- 최소 3명
- 가능하면 5명 이상

### 6.2 준비물

- 외부 접속 가능한 웹 링크
- task sheet
- Google Form

### 6.3 태스크

1. 이미지 업로드 후 part identification shortlist 확인
2. metadata preview 확인 및 수정
3. 결과 저장 또는 listing-ready 상태 확인

### 6.4 기록할 항목

- task completion time
- task success 여부
- manual edit count
- external search usage 여부
- 설문 점수

### 6.5 산출물

- mean task time
- mean edit count
- external search usage rate
- usability/trust average score

## 7. 결과 정리 순서

실험이 끝나면 다음 순서로 정리한다.

1. 숫자를 표로 정리
2. 각 표 아래 핵심 해석 2~3문장 작성
3. 실패 사례 이미지 2~4개 선정
4. `experiments/qwen3_vl_1000_sample_final_report_ko.md`와 `docs/reports/final_report_docx_ready_ko.md`에 수치 반영
5. 최종 리포트 본문에 복사

## 8. 최소 마감 전략

시간이 부족하면 아래만 먼저 끝낸다.

- Retrieval results 표 1개
- Latency 표 1개
- OCR 실패 사례 figure 1개
- usability pilot summary 표 1개 또는 protocol pending 문단

이 4개만 있어도 Evaluation 장의 핵심 뼈대는 충분히 완성된다.
