# 실험 결과표 입력 템플릿

이 문서는 실험 결과 숫자를 바로 입력할 수 있도록 만든 **표 전용 템플릿**이다.  
실험이 끝나면 각 표의 빈칸만 채우고, 필요하면 캡션과 해석 문단을 최종 보고서에 옮기면 된다.

## 실험 구조 요약

- `E1` Retrieval effectiveness ablation
  - `C1` OCR off, reranker on, text channel on
  - `C2` OCR on, reranker on, text channel on
  - `C3` OCR off, reranker off
  - `C4` OCR off, text-light
- `E2` OCR robustness benchmark
- `E3` Latency and resource benchmark
- `E4` Pilot usability study
- `E5` Engineering reliability validation

즉, **평가 실험은 크게 5개**이고, 그중 `E1` 안에서 `C1~C4` 비교가 들어간다.

---

## Table E1-1. Retrieval effectiveness comparison

캡션:

`Table E1-1. Retrieval effectiveness comparison across hybrid-search configurations.`

| Configuration | Accuracy@1 | Accuracy@5 | Recall@5 | MRR | Exact identifier hit rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| C1 |  |  |  |  |  |  |
| C2 |  |  |  |  |  |  |
| C3 |  |  |  |  |  |  |
| C4 |  |  |  |  |  |  |

짧은 해석 메모:
- Best config:
- Worst config:
- 핵심 원인:

---

## Table E1-2. Retrieval effectiveness by domain

캡션:

`Table E1-2. Retrieval effectiveness by domain.`

| Configuration | Domain | Accuracy@1 | Accuracy@5 | Recall@5 | MRR | Notes |
|---|---|---:|---:|---:|---:|---|
| C1 | auto_part |  |  |  |  |  |
| C1 | semiconductor_equipment_part |  |  |  |  |  |
| C2 | auto_part |  |  |  |  |  |
| C2 | semiconductor_equipment_part |  |  |  |  |  |
| C3 | auto_part |  |  |  |  |  |
| C3 | semiconductor_equipment_part |  |  |  |  |  |
| C4 | auto_part |  |  |  |  |  |
| C4 | semiconductor_equipment_part |  |  |  |  |  |

짧은 해석 메모:
- 자동차 부품 쪽 특징:
- 반도체 장비 부품 쪽 특징:

---

## Table E1-3. Error analysis summary

캡션:

`Table E1-3. Summary of common retrieval failure patterns.`

| Failure type | Count | Representative cause | Most affected configuration | Notes |
|---|---:|---|---|---|
| Visually similar variant confusion |  |  |  |  |
| OCR text mismatch |  |  |  |  |
| Missing metadata evidence |  |  |  |  |
| Label too small / blurred |  |  |  |  |
| Wrong top-1 but correct in top-5 |  |  |  |  |

---

## Table E2-1. OCR robustness results

캡션:

`Table E2-1. OCR robustness results on the identifier-visible subset.`

| Method | CER | Exact full-string match | Part number recall | Maker recall | Notes |
|---|---:|---:|---:|---:|---|
| PaddleOCR |  |  |  |  |  |
| Qwen extracted text |  |  |  |  |  |
| OCR + Qwen merged evidence |  |  |  |  |  |

짧은 해석 메모:
- 가장 강한 방법:
- 가장 취약한 조건:

---

## Table E2-2. OCR robustness by condition

캡션:

`Table E2-2. OCR robustness by image condition.`

| Condition | CER | Exact match | Part number recall | Maker recall | Notes |
|---|---:|---:|---:|---:|---|
| Clean single-object |  |  |  |  |  |
| Blur / glare |  |  |  |  |  |
| Cluttered background |  |  |  |  |  |
| Small label / far distance |  |  |  |  |  |
| Partial occlusion |  |  |  |  |  |

---

## Table E3-1. Query latency benchmark

캡션:

`Table E3-1. Query latency benchmark.`

| Operation | Condition | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---|---:|---:|---:|---:|---|
| Query total | Cold |  |  |  |  |  |
| Query total | Warm |  |  |  |  |  |
| Preprocessing | Warm |  |  |  |  |  |
| Embedding | Warm |  |  |  |  |  |
| Retrieval | Warm |  |  |  |  |  |
| Rerank | Warm |  |  |  |  |  |

짧은 해석 메모:
- 가장 큰 병목:
- interactive support 가능 여부:

---

## Table E3-2. Indexing latency benchmark

캡션:

`Table E3-2. Indexing latency benchmark.`

| Operation | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| Total indexing |  |  |  |  |  |
| OCR / text extraction |  |  |  |  |  |
| Caption / metadata generation |  |  |  |  |  |
| Image embedding |  |  |  |  |  |
| Text embedding |  |  |  |  |  |
| Milvus insert |  |  |  |  |  |

---

## Table E3-3. Resource usage summary

캡션:

`Table E3-3. Resource usage summary.`

| Environment | GPU / CPU | Peak memory | Notes |
|---|---|---:|---|
| Retrieval experiment |  |  |  |
| Indexing experiment |  |  |  |

---

## Table E4-1. Pilot usability study summary

캡션:

`Table E4-1. Pilot usability study summary.`

| Metric | Result | Notes |
|---|---:|---|
| Participants |  |  |
| Mean task completion time |  |  |
| Mean manual edit count |  |  |
| External search usage rate |  |  |
| Mean usability score |  |  |
| Mean trust score |  |  |

짧은 해석 메모:
- 사용자가 가장 유용하게 느낀 부분:
- 가장 혼란스러웠던 부분:

---

## Table E4-2. Usability questionnaire item summary

캡션:

`Table E4-2. Mean response scores for usability questionnaire items.`

| Questionnaire item | Mean score | Notes |
|---|---:|---|
| I could understand how to use the interface without help. |  |  |
| The search results were useful for narrowing down candidate parts. |  |  |
| The evidence shown with the results helped me judge whether the result was trustworthy. |  |  |
| The metadata preview reduced the effort needed to prepare listing information. |  |  |
| I would prefer using this prototype over completely manual search for similar tasks. |  |  |
| I felt confident making a decision from the shortlist provided by the system. |  |  |

---

## Table E5-1. Engineering reliability evidence

캡션:

`Table E5-1. Engineering reliability evidence and current validation status.`

| Validation item | Result | Notes |
|---|---|---|
| API pytest |  |  |
| Model pytest |  |  |
| Web build |  |  |
| Dataset manifest generation |  |  |
| Evaluation input generation |  |  |

---

## Table E5-2. Objective status summary

캡션:

`Table E5-2. Objective-by-objective status summary.`

| Objective | Current status | Evidence | Remaining gap |
|---|---|---|---|
| O1 Working MVP workflow |  |  |  |
| O2 Retrieval effectiveness |  |  |  |
| O3 OCR robustness |  |  |  |
| O4 Interactive feasibility |  |  |  |
| O5 User usefulness and trust |  |  |  |

---

## 빠르게 채우는 순서

1. 먼저 `Table E1-1`, `Table E3-1`, `Table E4-1`부터 채운다.
2. 그 다음 `Table E2-1`과 `Table E5-1`을 채운다.
3. 시간이 부족하면 domain breakdown과 detailed error table은 뒤로 미룬다.
4. 최종 보고서에는 최소한 다음 4개는 반드시 들어가게 한다.
   - `Table E1-1`
   - `Table E2-1`
   - `Table E3-1`
   - `Table E4-1` 또는 protocol pending 문단

