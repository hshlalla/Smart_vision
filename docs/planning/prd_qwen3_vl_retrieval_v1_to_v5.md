# PRD: Qwen3-VL-Centred Retrieval Pipeline

## 0. Document Purpose

이 문서는 Smart Vision 검색/인덱싱 파이프라인을 `Qwen3-VL` 중심으로 재설계하기 위한 제품 요구사항 문서다.
핵심 목적은 다음 세 가지다.

1. 현재 파이프라인의 병목과 설계 한계를 정리한다.
2. `Qwen3-VL Embedding / Reranker / Instruct` 중심의 새 파이프라인 요구사항을 정의한다.
3. 버전별로 왜 요구사항이 바뀌었는지 추적 가능하게 문서화한다.

이 문서는 즉시 코드 구현을 강제하는 문서가 아니라, 이후 구현과 실험을 일관되게 진행하기 위한 기준 문서다.

---

## 1. Problem Statement

현재 시스템은 멀티모달 하이브리드 검색 구조를 갖고 있지만, 실제 운영상 다음 문제가 발생했다.

- `BGE-VL-large` 기반 이미지 임베딩은 CPU 환경에서 너무 느리다.
- OCR은 exact string 확보에는 유리하지만, 사용자 업로드 배경이 복잡하거나 여러 제품이 섞이면 쉽게 오염된다.
- 기존 구조에서는 reranker 인터페이스는 있었지만 실제 learned reranker가 연결되지 않아, retrieval 이후의 정교한 재정렬이 부족했다.
- 캡셔닝은 `Qwen3-VL-2B-Instruct`를 이미 사용하고 있었지만, 전체 검색 설계와의 일관성은 부족했다.

이 문제를 해결하기 위해 기본 파이프라인을 `Qwen3-VL` 중심으로 재구성하고, OCR은 기본 경로에서 제거하거나 optional mode로 제한하는 방향을 검토한다.

---

## 2. Target Product Direction

기본 방향은 다음과 같다.

- 기본 검색/인덱싱 경로는 `Qwen3-VL` 중심으로 단순화한다.
- OCR은 기본 요구사항이 아니라 실험적/보조 기능으로 취급한다.
- 최종적으로는 사용자가 "단일 제품 사진을 올리면 빠르게 후보를 찾고, 설명과 근거를 제시하며, 확정 시 지식베이스에 반영"하는 흐름을 만든다.

---

## 3. Target User Flow

### 3.1 Default Flow

사용자가 별도 옵션을 켜지 않았을 때 기본 경로는 다음과 같다.

1. 사용자가 제품 이미지를 업로드한다.
2. 이미지 임베딩을 `Qwen3-VL-Embedding-2B`로 생성한다.
3. 제품 정보 추출을 `Qwen3-VL-2B-Instruct`로 수행한다.
4. 제품 캡션 추출을 `Qwen3-VL-2B-Instruct`로 수행한다.
5. 3, 4 단계에서 나온 텍스트를 `BGE-M3`에 넣어 텍스트 임베딩을 생성한다.
6. Milvus에서 이미지/텍스트 후보를 검색한다.
7. `Qwen3-VL-Reranker-2B`로 top-N 후보를 재정렬한다.
8. 최종 추론 결과를 사용자에게 보여준다.
9. 내부 인덱스에서 확신이 낮거나 결과가 부족하면 웹 검색을 수행한다.
10. 사용자가 결과를 확인하고 승인하면 저장한다.
11. 저장된 정보와 캐시는 이후 검색 시 재사용한다.

### 3.2 Optional Expert Flow

고급 사용자가 더 엄격한 식별을 원할 경우에만 별도 옵션을 고려한다.

- OCR 보조 모드를 켠다.
- OCR이 먼저 텍스트를 뽑아 exact string evidence를 추가한다.
- 단, 이 모드는 이미지 품질/배경/겹침에 따라 오염 가능성이 높으므로 기본값으로 두지 않는다.

현재 판단으로는 이 expert flow는 설계 후보로만 유지하고, 기본 요구사항에는 포함하지 않는다.

---

## 4. Functional Requirements

### FR1. Image Embedding

- 시스템은 제품 이미지를 `Qwen3-VL-Embedding-2B`로 임베딩해야 한다.
- 이 벡터는 Milvus의 이미지 컬렉션에 저장 가능해야 한다.
- 검색 시 동일 모델로 질의 이미지 임베딩을 생성해야 한다.

### FR2. Product Description Extraction

- 시스템은 `Qwen3-VL-2B-Instruct`로 제품 정보 추출을 수행해야 한다.
- 최소 추출 필드는 다음과 같다.
  - maker
  - candidate product name
  - candidate part/model string
  - inferred category
  - brief evidence

### FR3. Caption Generation

- 시스템은 `Qwen3-VL-2B-Instruct`로 설명형 캡션을 생성해야 한다.
- 캡션은 검색용 텍스트 evidence로 저장 가능해야 한다.

### FR4. Text Embedding

- 시스템은 extracted product info + caption text를 `BGE-M3`로 임베딩해야 한다.
- OCR을 기본 경로에서 제거하더라도 metadata/caption 기반 검색이 가능해야 한다.

### FR5. Retrieval

- 시스템은 이미지 임베딩 검색과 텍스트 임베딩 검색을 병렬적으로 수행해야 한다.
- 검색 후보는 `model_id` 기준으로 합쳐야 한다.
- score decomposition을 유지해야 한다.

### FR6. Re-ranking

- 시스템은 `Qwen3-VL-Reranker-2B`로 top-N 후보를 재정렬해야 한다.
- reranker는 query image, query text, candidate image, candidate text를 함께 볼 수 있어야 한다.

### FR7. Fallback Web Search

- 내부 검색이 충분히 확신되지 않을 때 웹 검색을 추가로 수행할 수 있어야 한다.
- 외부 검색 결과는 answer enrichment 용도로 사용하고, 자동 저장은 금지한다.

### FR8. Human Confirmation and Writeback

- 최종 후보는 사용자 승인 후에만 저장한다.
- 자동 writeback은 기본 비활성 상태여야 한다.
- 승인된 결과만 캐시/인덱스 재사용 대상으로 본다.

### FR9. Cache Reuse

- 승인된 제품 정보, 요약, 웹 보강 정보는 이후 검색/채팅에서 재사용 가능해야 한다.
- 캐시는 검색 latency 개선과 사용자 경험 향상을 목적으로 한다.

---

## 5. Non-Functional Requirements

- GPU 환경에서 실사용 가능한 latency를 목표로 한다.
- 기본 경로는 OCR 없이도 동작해야 한다.
- OCR을 제거하더라도 exact identifier matching 성능을 측정할 수 있어야 한다.
- 실험 전후 결과를 재현 가능하게 로그와 평가 스크립트를 남겨야 한다.
- 기존 사용자 UI를 완전히 깨지 않도록 점진적으로 마이그레이션해야 한다.

---

## 6. Version History (V1-V5)

### V1. Initial Hybrid Baseline

구성:

- `BGE-VL-large` image embedding
- `BGE-M3` text embedding
- PaddleOCR
- optional captioning
- similarity + lexical/spec score fusion

왜 이렇게 시작했는가:

- retrieval-first MVP를 빠르게 만들기 위해서
- 이미지/텍스트/OCR를 모두 결합한 전형적인 hybrid retrieval 구조가 필요했기 때문

문제:

- image encoder가 느림
- reranker가 실질적으로 없음
- OCR 품질이 장면 복잡도에 크게 흔들림

### V2. Ranking Reliability Upgrade

변경:

- 한글 lexical match 보강
- exact field boost 추가
- low-score candidate cutoff 추가

왜 바뀌었는가:

- 사용자 기대와 실제 검색 순위가 맞지 않았음
- exact string이 들어간 결과가 1위가 되지 않는 문제가 있었음

### V3. Qwen3-VL Adoption Candidate

변경:

- 이미지 표현을 `Qwen3-VL-Embedding-2B`로 교체 검토
- reranker를 `Qwen3-VL-Reranker-2B`로 연결
- caption/instruct를 `Qwen3-VL-2B-Instruct` 중심으로 정리

왜 바뀌었는가:

- `Qwen3-VL` 모델군이 retrieval, reranking, instruction을 같은 계열로 제공
- image-side stack 일관성 강화
- 실제 multimodal reranking 도입 필요

### V4. OCR Optional Mode Proposal

변경 제안:

- OCR을 기본 경로에서 제거
- 사용자가 필요할 때만 OCR 보조 모드 활성화

왜 제안되었는가:

- 복잡한 배경, 다중 물체, 라벨 오염에서 OCR false evidence 위험이 큼
- Qwen3-VL이 이미지 이해와 OCR-like understanding을 상당 부분 흡수할 가능성이 있음

남은 의문:

- exact part number가 중요한 경우 OCR 제거가 손해일 수 있음
- 따라서 이 버전은 설계 제안 단계로 유지

### V5. Experiment-First Product Direction

최종 현재 방향:

- 기본 경로는 OCR 없는 `Qwen3-VL image + instruct + BGE-M3 text + reranker` 구조
- OCR은 기본 요구사항에서 제외
- OCR 유지 여부는 비교 실험으로 판단

왜 이 방향이 되었는가:

- 현재 가장 큰 목표는 pipeline simplification과 search quality 검증
- 복잡한 OCR 분기까지 한 번에 안고 가면 실험 해석이 어려워짐
- 먼저 기본 경로를 단순하게 정의하고, 그 다음 OCR이 실제로 이득인지 측정하는 것이 합리적임

---

## 7. Explicit Out of Scope

다음 항목은 현재 PRD의 즉시 범위에 넣지 않는다.

- YOLO 기반 object detection 필수 적용
- OCR을 기본 mandatory step으로 두는 설계
- 자동 writeback 허용
- full autonomous product identification claim

이 항목들은 추후 실험 결과가 정리된 뒤 별도 PRD로 다룬다.

---

## 8. Risks

- `Qwen3-VL`이 제품 설명은 잘하지만 exact identifier string을 충분히 보존하지 못할 수 있다.
- OCR 제거 시 short text matching 성능이 떨어질 수 있다.
- image embedding, reranking, text embedding이 서로 다른 모델에 의해 분리될 때 score fusion 최적화가 필요하다.
- 실험 전에 컬렉션 재구성 및 재인덱싱이 필요하다.

---

## 9. Decision Log

현재 문서 기준 의사결정은 다음과 같이 기록한다.

- 기본 경로에서 OCR 제거 방향을 우선 검토한다.
- 텍스트 인코더는 당장 제거하지 않고 `BGE-M3`를 유지한다.
- `Qwen3-VL-Reranker-2B`는 반드시 실제 검색 경로에 연결한다.
- 최종 채택 여부는 1000개 유사 상품 실험으로 검증한다.

---

## 10. Next Documents

이 PRD 다음 단계 문서는 아래를 참조한다.

- `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- 구현 backlog는 필요 시 `docs/planning/to_do_list.md`에 연결한다.
