# 실험 결과표 입력 템플릿

이 문서는 실험 결과 숫자를 바로 입력할 수 있도록 만든 **표 전용 템플릿**이다.  
현재 문서는 2026-03-21 기준으로 확보된 수치를 먼저 채워 넣은 작업본이다.  
미완료 항목은 `미완료` 또는 `N/A`로 남겨두었다.

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
| C1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9667 | sampled holdout, reranker on, quality gain 없음, warm latency 급증 |
| C2 | 미완료 | 미완료 | 미완료 | 미완료 | 미완료 | OCR-on retrieval ablation 미완료 |
| C3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9667 | sampled holdout baseline, OCR off, reranker off |
| C4 | 미완료 | 미완료 | 미완료 | 미완료 | 미완료 | text-light ablation 미실행 |

짧은 해석 메모:
- Best config: 정확도 기준으로는 `C1`과 `C3`가 동일, 운영성 기준으로는 `C3`
- Worst config: 완료된 비교군 중에서는 `C1`이 latency 측면에서 가장 불리
- 핵심 원인: reranker가 sampled holdout에서 순위 품질을 더 올리지 못했고, local Apple Silicon 환경에서는 비용만 크게 증가함

---

## Table E1-2. Retrieval effectiveness by domain

캡션:

`Table E1-2. Retrieval effectiveness by domain.`

| Configuration | Domain | Accuracy@1 | Accuracy@5 | Recall@5 | MRR | Notes |
|---|---|---:|---:|---:|---:|---|
| C1 | auto_part | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 24 queries, latency 매우 높음 |
| C1 | semiconductor_equipment_part | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6 queries, latency 매우 높음 |
| C2 | auto_part | 미완료 | 미완료 | 미완료 | 미완료 | OCR-on retrieval ablation 미완료 |
| C2 | semiconductor_equipment_part | 미완료 | 미완료 | 미완료 | 미완료 | OCR-on retrieval ablation 미완료 |
| C3 | auto_part | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 24 queries, baseline |
| C3 | semiconductor_equipment_part | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6 queries, baseline |
| C4 | auto_part | 미완료 | 미완료 | 미완료 | 미완료 | text-light ablation 미실행 |
| C4 | semiconductor_equipment_part | 미완료 | 미완료 | 미완료 | 미완료 | text-light ablation 미실행 |

짧은 해석 메모:
- 자동차 부품 쪽 특징: 정확도는 높고 warm latency도 상대적으로 낮음
- 반도체 장비 부품 쪽 특징: 정확도는 높지만 preprocessing 비용이 더 큼

---

## Table E1-3. Error analysis summary

캡션:

`Table E1-3. Summary of common retrieval failure patterns.`

| Failure type | Count | Representative cause | Most affected configuration | Notes |
|---|---:|---|---|---|
| Visually similar variant confusion | 1 | 동일 `group_key` 내 유사 부품(`auto_gparts_cat50_000231 -> 000229`) | C1, C3 | exact item 기준에서만 오답으로 집계 |
| OCR text mismatch | 0 | OCR off baseline에서는 직접 원인 아님 | C1, C3 | retrieval ablation에서는 OCR 비활성 |
| Missing metadata evidence | 0 | sampled holdout baseline에서는 관찰 안 됨 | C1, C3 | 현 샘플에서는 metadata로 shortlist 보조 가능 |
| Label too small / blurred | 0 | retrieval sampled holdout에서는 직접 오답 원인으로 집계 안 됨 | C1, C3 | OCR pilot에서는 큰 문제로 관찰됨 |
| Wrong top-1 but correct in top-5 | 1 | exact item mismatch이지만 동일 part group 회수 | C1, C3 | group-level metric은 맞지만 item-level은 오답 |

---

## Table E2-1. OCR robustness results

캡션:

`Table E2-1. OCR robustness results on the identifier-visible subset.`

| Method | CER | Exact full-string match | Part number recall | Maker recall | Notes |
|---|---:|---:|---:|---:|---|
| PaddleOCR | 1.9625 | 0.1000 | 0.1000 | 0.1000 | `10` samples, OCR-only pilot, `523.45s` |
| Qwen extracted text | 미완료 | 미완료 | 미완료 | 미완료 | full OCR+Qwen pilot은 local time budget 내 미완료 |
| OCR + Qwen merged evidence | 미완료 | 미완료 | 미완료 | 미완료 | full OCR+Qwen pilot은 local time budget 내 미완료 |

짧은 해석 메모:
- 가장 강한 방법: 완료된 항목 중에서는 `PaddleOCR` 단독이 유일한 완료 baseline
- 가장 취약한 조건: blur / low-text / image-block markdown 위주 장면

---

## Table E2-2. OCR robustness by condition

캡션:

`Table E2-2. OCR robustness by image condition.`

| Condition | CER | Exact match | Part number recall | Maker recall | Notes |
|---|---:|---:|---:|---:|---|
| Clean single-object / OCR visible | 0.4815 | 0.3333 | 0.3333 | 0.3333 | `3` samples, 산업용 라벨이 선명한 경우 일부 성공 |
| Blur / low-text / image-block dominant | 2.8510 | 0.0000 | 0.0000 | 0.0000 | `7` samples, 자동차 부품 샘플 다수에서 실패 |
| Cluttered background | N/A | N/A | N/A | N/A | 현재 `10`-sample pilot에서 별도 분리 집계 안 함 |
| Small label / far distance | N/A | N/A | N/A | N/A | 현재 `10`-sample pilot에서 별도 분리 집계 안 함 |
| Partial occlusion | N/A | N/A | N/A | N/A | 현재 `10`-sample pilot에서 별도 분리 집계 안 함 |

---

## Table E3-1. Query latency benchmark

캡션:

`Table E3-1. Query latency benchmark.`

| Operation | Condition | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---|---:|---:|---:|---:|---|
| Query total | Cold (`C3`) | 699.78 | 647.29 | 663.39 | 715.23 | sampled holdout baseline |
| Query total | Warm (`C3`) | 731.13 | 642.29 | 653.70 | 751.85 | interactive baseline |
| Preprocessing | Warm (`C3`) | 698.96 | 611.04 | 619.84 | 715.53 | 대부분의 비용이 이 구간에 집중 |
| Embedding | Warm | N/A | N/A | N/A | N/A | current runner는 embedding을 preprocessing 내부로 집계 |
| Retrieval | Warm (`C3` image+text search`) | 28.01 | N/A | N/A | N/A | image search 3.72 + text search 20.39 + fetch 3.90 |
| Rerank | Warm (`C1` finalize-dominant) | 65199.13 | 64242.96 | 71168.84 | 73226.32 | reranker-on local CPU fallback, 운영상 비실용적 |

짧은 해석 메모:
- 가장 큰 병목: `C3`에서는 preprocessing, `C1`에서는 reranker-dominated finalize
- interactive support 가능 여부: `C3`는 가능, `C1`은 현재 local Apple Silicon에서는 비현실적

---

## Table E3-2. Indexing latency benchmark

캡션:

`Table E3-2. Indexing latency benchmark.`

| Operation | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| Total indexing | 미수집 | 미수집 | 미수집 | 미수집 | 이번 sampled evaluation 범위에서는 정식 재측정 안 함 |
| OCR / text extraction | 미수집 | 미수집 | 미수집 | 미수집 | OCR pilot은 retrieval/OCR robustness 쪽으로 별도 측정 |
| Caption / metadata generation | 미수집 | 미수집 | 미수집 | 미수집 | 이번 표에서는 정식 수집 안 함 |
| Image embedding | 미수집 | 미수집 | 미수집 | 미수집 | 기존 운영 로그는 있으나 현재 sampled 표에는 미반영 |
| Text embedding | 미수집 | 미수집 | 미수집 | 미수집 | 현재 sampled 표에는 미반영 |
| Milvus insert | 미수집 | 미수집 | 미수집 | 미수집 | 별도 정식 인덱싱 benchmark 필요 |

---

## Table E3-3. Resource usage summary

캡션:

`Table E3-3. Resource usage summary.`

| Environment | GPU / CPU | Peak memory | Notes |
|---|---|---:|---|
| Retrieval experiment (`C3`) | Apple Silicon `mps` + CPU mixed | 미수집 | baseline path는 practical |
| Retrieval experiment (`C1`) | reranker `cpu` fallback | 미수집 | `mps`에서 reranker scoring 불안정, latency 급증 |
| OCR pilot | Paddle runtime + local CPU-heavy path | 미수집 | `10` samples에 `523.45s`, routine local OCR로는 부담 큼 |

---

## Table E4-1. Pilot usability study summary

캡션:

`Table E4-1. Pilot usability study summary.`

| Metric | Result | Notes |
|---|---:|---|
| Participants | 미완료 | pilot user study 미실시 |
| Mean task completion time | 미완료 |  |
| Mean manual edit count | 미완료 |  |
| External search usage rate | 미완료 |  |
| Mean usability score | 미완료 |  |
| Mean trust score | 미완료 |  |

짧은 해석 메모:
- 사용자가 가장 유용하게 느낀 부분:
- 가장 혼란스러웠던 부분:

---

## Table E4-2. Usability questionnaire item summary

캡션:

`Table E4-2. Mean response scores for usability questionnaire items.`

| Questionnaire item | Mean score | Notes |
|---|---:|---|
| I could understand how to use the interface without help. | 미완료 | pilot questionnaire 미수집 |
| The search results were useful for narrowing down candidate parts. | 미완료 |  |
| The evidence shown with the results helped me judge whether the result was trustworthy. | 미완료 |  |
| The metadata preview reduced the effort needed to prepare listing information. | 미완료 |  |
| I would prefer using this prototype over completely manual search for similar tasks. | 미완료 |  |
| I felt confident making a decision from the shortlist provided by the system. | 미완료 |  |

---

## Table E5-1. Engineering reliability evidence

캡션:

`Table E5-1. Engineering reliability evidence and current validation status.`

| Validation item | Result | Notes |
|---|---|---|
| API pytest | passed | `21 passed` |
| Model pytest | passed | `11 passed` |
| Web build | passed | production build succeeded |
| Dataset manifest generation | passed | sampled/query manifest runners produced outputs |
| Evaluation input generation | passed | `index 1192 / query 299` |

---

## Table E5-2. Objective status summary

캡션:

`Table E5-2. Objective-by-objective status summary.`

| Objective | Current status | Evidence | Remaining gap |
|---|---|---|---|
| O1 Working MVP workflow | 구현 | `E0 8/8`, API/UI/runtime 동작 | 없음 |
| O2 Retrieval effectiveness | 부분구현 | sampled holdout `C1/C3`, current-index sanity benchmark | `C2`, `C4`, larger controlled ablation 미완료 |
| O3 OCR robustness | 부분구현 | `10`-sample OCR-only pilot 완료 | full OCR+Qwen benchmark 및 OCR-on retrieval benchmark 미완료 |
| O4 Interactive feasibility | 부분구현 | `C3` warm mean `731.13 ms`, current-index warm mean `653.65 ms` | indexing benchmark/resource peak 미수집 |
| O5 User usefulness and trust | 미완료 | UI/agent/catalog 기능 준비 완료 | pilot usability study 미실시 |

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
