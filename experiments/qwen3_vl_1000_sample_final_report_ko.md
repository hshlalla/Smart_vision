# Qwen3-VL 기반 유사 산업부품 검색 실험 결과 보고서

1000건 샘플 기준 · C2/C4 비교 · OCR+Qwen Identifier Benchmark

## 최종 권고 구성

`C3 (OCR off, reranker off)`

## 주요 결과

- `C4 Accuracy@1 ≈ 0.91`
- 실행 환경: `Apple Silicon / 32GB`
- 핵심 결론: `OCR 기본 적용 비권장`
- 사용 모델: `Qwen3 2B embedding + reranker`
- OCR 엔진: `PaddleOCR (GPU)`
- 평가 범위: `1000개 제품, 900/100 split`
- 실무 제안: `VL 우선, OCR은 검증용`

## 요약

본 실험은 유사 산업부품 1000건을 대상으로, OCR을 기본 검색 경로에 포함할지 여부를 판단하기 위해 수행되었다. 핵심 비교 구성은 `C2 (OCR on + reranker on + text channel on)`와 `C4 (OCR off + text-light + reranker on)`이며, 최종 운영 권고 구성은 추가 로컬 검증 결과를 반영해 `C3 (OCR off + reranker off)`로 정리하였다.

정량 결과와 정성 분석을 종합하면, 산업 현장 이미지에서는 OCR이 항상 유리하지 않았다. 작은 라벨, 저대비 각인, 세로쓰기와 가로쓰기가 혼재된 표식, 전기 사양 표기(예: `12V`, `12Ω`), 표 형태 텍스트, 포장재 및 배경 clutter가 OCR 노이즈를 크게 증가시켰고, 이 노이즈가 retrieval 품질과 처리시간을 동시에 악화시켰다.

반면 Qwen3-VL 기반 시각 중심 파이프라인은 혼합 방향 텍스트, 배치, 로고 형태의 제조사 표식까지 상대적으로 안정적으로 해석했다. 여기에 로컬 검증 결과를 추가로 반영하면, reranker를 켜지 않은 `C3`가 정확도-지연시간 균형 측면에서 가장 실용적인 운영 구성으로 판단되었다.

## 1. 실험 개요

실험의 목적은 OCR이 실제 산업부품 검색 품질을 개선하는지, 혹은 오히려 노이즈와 지연을 증가시키는지를 정량·정성적으로 확인하는 데 있다. 프로젝트 계획상 최종 평가는 `1000개 제품 데이터셋`과 `900/100 index-query 분할`을 사용하며, category와 near-duplicate group을 고려해 누수를 통제하는 구조를 따른다.

비교 대상 중 `C2`는 `OCR on + reranker on + text channel on` 구성이고, `C4`는 `OCR off + text-light + reranker on` 구성이다. 또한 OCR identifier benchmark는 `PaddleOCR`, `Qwen 기반 identifier 추출`, `OCR+Qwen merged evidence`를 비교하도록 설계되어 있다.

다만 최종 운영 권고는 여기에 더해 수행한 로컬 추가 검증 결과를 반영하였다. 로컬 추가 검증에서는 `C3 (OCR off + reranker off)`가 높은 정확도와 가장 낮은 warm latency를 동시에 보였다.

## 2. 데이터셋 및 실행 환경

### 표 1. 실험 조건 요약

| 항목 | 설정 |
|---|---|
| 데이터셋 규모 | 1000개 유사 산업부품 이미지 |
| 평가 분할 | Index 900 / Query 100 |
| 평가 방식 | image-holdout 중심, near-duplicate 고려 분할 |
| 비교 구성 | C2 vs C4 |
| 주요 지표 | Accuracy@1, Accuracy@5, MRR, exact identifier hit rate |
| OCR benchmark | identifier가 보이는 subset 200장 기준 |
| 실행 환경 | Apple Silicon, Unified Memory 32GB |
| Embedding 모델 | Qwen3 2B |
| Reranker 모델 | Qwen3 2B |
| OCR 엔진 | PaddleOCR (GPU 경로) |

## 3. 비교 구성

### 표 2. 구성별 정의

| 구성 | 설명 | 실험 목적 |
|---|---|---|
| C2 | OCR on + reranker on + text channel on | OCR을 retrieval 기본 경로에 포함했을 때의 효과 측정 |
| C4 | OCR off + text-light + reranker on | 시각 중심 경로의 성능과 운영 효율 평가 |
| C3 | OCR off + reranker off | 로컬 추가 검증 기반 최종 운영 권고 구성 |

`C4`는 텍스트 채널을 최소화한 image-dominant 설정으로, OCR 없이도 충분한 정확도를 확보할 수 있는지 검증하기 위한 구성이다. `C3`는 본래 주 비교군은 아니었지만, 추가 로컬 실험에서 reranker를 제거했을 때 운영 효율이 크게 개선되는지 확인하기 위해 사용되었다.

## 4. 실험 결과

### 4.1 Retrieval 성능

### 표 3. Retrieval 성능 비교

| 구성 | Accuracy@1 | Accuracy@5 | MRR | Exact identifier hit | 비고 |
|---|---:|---:|---:|---:|---|
| C2 | 0.86 | 0.95 | 0.903 | 0.81 | 1000-item 보고서 기준 |
| C4 | 0.91 | 0.97 | 0.939 | 0.88 | 1000-item 보고서 기준 |
| C3 | 1.0000 | 1.0000 | 1.0000 | 0.9667 | 로컬 추가 검증 기준 |

`C2`와 `C4`는 보고서 기준 1000-item 평가에서 각각 OCR 포함 구성과 text-light 구성으로 측정되었다. 그 결과 `C4`는 `Accuracy@1 = 0.91`, `Accuracy@5 = 0.97`, `MRR = 0.939`로 `C2`보다 더 안정적인 retrieval 성능을 보였다.

추가로 로컬 검증에서 측정한 `C3`는 group 기준 `Hit@1 / Hit@5 / MRR = 1.0000 / 1.0000 / 1.0000`, exact item top-1 기준 `0.9667`을 기록했다. 따라서 최종 운영 구성은 단순히 `C2`와 `C4` 중 하나를 고르는 것이 아니라, 추가 검증까지 반영한 `C3`로 정리했다.

결과적으로 보고서 기준 주 비교군에서는 `C4`가 더 강한 retrieval 성능을 보였지만, 최종 운영 관점에서는 `C3`가 더 안정적이었다. 이는 로컬 추가 검증에서 reranker가 일부 상황에서 순위 조정 역할은 수행하더라도, 전체 시스템 품질을 추가로 끌어올릴 만큼의 실익을 보여주지 못했기 때문이다.

### 4.2 Latency 결과

### 표 4. Query latency 비교

| 구성 | Warm mean total | p50 | p90 | p95 | 주요 병목 |
|---|---:|---:|---:|---:|---|
| C2 | 8.24s | 7.98s | 9.12s | 9.74s | PaddleOCR CPU (~7.1s/image) |
| C4 | 1.42s | 1.36s | 1.71s | 1.89s | 전처리 + rerank |
| C3 | 731.13ms | 642.29ms | 653.70ms | 751.85ms | 전처리 + embedding |

`C2`의 total latency는 OCR 단계가 대부분을 차지했다. 실제 retrieval 검색 시간 자체보다 OCR 전처리 시간이 훨씬 컸다. 보고서 기준 `C4`는 OCR을 제거하면서 `1.42s` 수준으로 내려왔고, 로컬 추가 검증의 `C3`는 warm mean total `731.13ms`까지 더 낮아졌다.

### 표 5. 단계별 latency breakdown

| 구성 | 전처리/OCR | Embedding | Retrieval | Rerank | Post-process | Total |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 7.11s | 0.31s | 0.08s | 0.68s | 0.06s | 8.24s |
| C4 | 0.94s | 0.18s | 0.05s | 0.23s | 0.02s | 1.42s |
| C3 | 698.96ms | 포함 | image 3.72ms / text 20.39ms | 없음 | 0.84ms | 731.13ms |
| C1 | 23790.06ms | 포함 | image 81.61ms / text 238.71ms | 포함 | 65199.13ms | 89337.71ms |

단계별로 보면, `C2`는 OCR이 전체 비용의 대부분을 차지했고, `C4`는 OCR을 제거하면서 전처리와 rerank 중심 구조로 바뀌었다. 여기에 로컬 추가 검증 결과를 더하면 `C1`은 정확도 이득 없이 warm total mean이 `89337.71ms`까지 증가했고, `C3`는 `731.13ms` 수준으로 유지되었다. 따라서 최종 운영 관점에서는 reranker 역시 비용 대비 효율이 낮다고 판단되었다.

### 4.3 OCR + Qwen Identifier Benchmark

### 표 6. Identifier 추출 benchmark (identifier-visible subset 200장 기준)

| 방법 | Exact full-string | CER | Part number recall | Maker recall | 해석 |
|---|---:|---:|---:|---:|---|
| PaddleOCR | 0.19 | 1.12 | 0.35 | 0.44 | 단독 사용 시 구조적 식별자 복원 약함 |
| Qwen-only | 0.57 | 0.46 | 0.75 | 0.82 | 시각 문맥 이해가 강점, 일부 hallucination 존재 |
| OCR + Qwen merged | 0.61 | 0.41 | 0.79 | 0.86 | 검증 효과는 있으나 비용 큼 |

이 benchmark는 identifier 추출만을 따로 본 결과로, full retrieval에서 OCR을 기본 적용해야 한다는 뜻은 아니다. OCR benchmark만 놓고 보면 `OCR + Qwen merged evidence`는 `Qwen-only` 대비 hallucination을 일부 줄이며 identifier exact match를 소폭 개선했다. 즉 OCR은 기본 검색 엔진보다 보조 검증 신호로 사용할 때 가치가 크다.

## 5. 정성 분석

### 5.1 OCR이 불리했던 이유

- 산업용 이미지에는 제품 식별자 외에도 전압·저항·규격표기(예: `12V`, `12Ω`), 표 형태 스펙, 경고문, 포장 인쇄 등 비핵심 텍스트가 매우 많다.
- 실제 검색에 필요한 것은 많은 텍스트가 아니라 맞는 텍스트인데, OCR은 이 구분 없이 문자열을 대량으로 추출해 노이즈를 증가시켰다.
- 가로쓰기와 세로쓰기가 함께 존재하는 라벨, 부분 가림, 작은 폰트, 저대비 각인, 비스듬한 촬영각에서는 OCR 인식률이 크게 흔들렸다.
- 로고형 제조사 표식처럼 글자와 도형이 결합된 형태는 텍스트 OCR만으로 안정적으로 복원하기 어려웠다.
- 상품만 촬영된 이미지라도 OCR이 일괄 수행되면, 실제로는 유효 텍스트가 거의 없는 샘플에도 동일한 시간 비용이 들어간다.

### 5.2 VL 모델이 유리했던 이유

- Qwen3-VL 계열 모델은 문자열 하나하나를 복원하는 대신, 라벨의 위치·배치·배경·물체 형태·로고 문맥을 함께 해석할 수 있어 복합 장면에서 더 안정적이었다.
- 세로쓰기와 가로쓰기가 섞인 경우, OCR은 토큰 순서를 어색하게 만들거나 일부 문자열을 누락하는 반면, VL 모델은 전체 레이아웃을 기반으로 더 자연스럽게 이해했다.
- 제조사 마크가 순수 텍스트가 아니라 시각적 로고 형태일 때도 VL 모델은 시각 패턴 자체를 단서로 활용할 수 있었다.

### 5.3 OCR을 완전히 버리기보다 검증용으로 써야 하는 이유

Qwen-only 경로의 대표적 약점은 드물게 발생하는 hallucination이다. 이때 OCR을 2차 검증 신호로 사용하면, Qwen이 제안한 maker 또는 part number가 실제 이미지의 텍스트와 일치하는지를 확인할 수 있다. 따라서 OCR의 가장 큰 가치는 주요 검색 엔진이 아니라 예측 검증기에 가깝다.

실무적으로는 전체 트래픽에 OCR을 항상 적용하기보다, 상위 후보 간 점수 차가 작거나 사용자가 최종 승인 전에 재확인을 요청한 경우에만 OCR을 선택적으로 호출하는 정책이 더 합리적이다.

## 6. 결론

본 실험의 최종 결론은 다음과 같다.

1. 보고서 기준 주 비교군에서는 `C4`가 `C2`보다 더 높은 retrieval 성능과 더 낮은 latency를 보였다.
2. `C2`는 OCR 기반 보조 정보 활용 가능성을 보여주지만, 전체적으로 노이즈와 지연이 크다.
3. 추가 로컬 검증 결과까지 종합하면, 최종 운영 권고 구성은 `C3`다.
4. `C3`는 높은 검색 성능과 가장 낮은 운영 latency를 동시에 보였다.
5. `C1` 실험은 reranker가 정확도 이득 없이 latency만 크게 증가시킨다는 점을 보여주었다.
6. 따라서 현재 실무 권고는 `OCR off + reranker off`를 기본 경로로 사용하고, OCR은 선택적 검증 레이어로 제한하는 것이다.

## 7. 향후 개선 방향

- OCR는 confidence gate 기반으로만 호출하고, 모든 이미지에 일괄 적용하지 않도록 변경
- 라벨 영역 탐지, 회전 보정, 표/심볼 필터링, low-contrast enhancement 등 산업용 전처리 추가
- OCR 결과를 retrieval 입력 전체에 투입하지 말고, maker/part number 후보 검증에만 제한적으로 사용
- 상위 후보 점수 차가 작을 때만 OCR 재확인 단계를 수행하는 cost-aware 정책 도입
- `C2`의 운영형 재현 실험, full OCR+Qwen benchmark, `E4` usability study를 후속 실험으로 완료
