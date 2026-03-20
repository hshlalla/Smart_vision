Smart Vision Model Release Notes
================================

## 2.4.0

### Added
- 경량 text-only retrieval runtime을 추가했습니다.
- request 단위 reranker on/off override를 지원하도록 검색 경로를 확장했습니다.
- Apple Silicon `mps` 디바이스 선택을 PyTorch 기반 런타임에 반영했습니다.
- 실험용 별도 Milvus 컬렉션을 사용할 수 있도록 unified JSONL 인입 스크립트에 collection override 옵션을 추가했습니다.

### Changed
- `model_id` 카운터 저장소를 Milvus 컬렉션에서 SQLite로 전환했습니다.
- 인덱싱 전처리에서 OCR 입력과 임베딩 입력을 분리했습니다.
- 저장용/임베딩용 이미지 사본은 축소본을 사용하도록 조정했습니다.
- 메타 preview / caption 전략이 `LOCAL_MODE`와 상위 API 설정 기준에 맞춰 정리되었습니다.
- 대량 인입 시 기존 벡터 인덱스를 매번 다시 생성하지 않도록 조정해 재인덱싱 startup 정체를 줄였습니다.
- unified JSONL 인입은 동일 부품이 다른 `model_id`로 다시 들어와도 기존 모델에 merge 할 수 있도록 `maker + part_number` / `part_number` 기반 dedup 경로를 사용합니다.
- 중복으로 판단된 후속 입력은 skip만 하지 않고, 더 풍부한 `description`/`metadata_text`를 기존 모델 텍스트에 병합하도록 정리했습니다.
- `Qwen3-VL-Reranker-2B` 로딩 경로를 공식 `Qwen3VLForConditionalGeneration` 기반으로 정리하고, 현재 Apple Silicon 환경에서 필요 시 `RERANKER_DEVICE=cpu` fallback을 사용할 수 있게 했습니다.

### Fixed
- text-only 검색 경로가 attrs 이미지 정보를 함께 반환하지 않던 문제를 수정했습니다.
- part number exact match에서 공백/하이픈 차이에 덜 민감하도록 정규화 매칭을 보강했습니다.
- 기존 Milvus 인덱스가 이미 존재하는 상황에서 `create_index()` 재호출 대기로 인입이 멈추던 문제를 수정했습니다.
- 기존 reranker 초기화가 잘못된 아키텍처 로더를 사용하던 문제를 수정했습니다.

## 2.3.0

### Added
- `Qwen3-VL-Embedding-2B` 기반 이미지 임베더를 추가했습니다.
- `Qwen3-VL-Reranker-2B` 기반 multimodal reranker를 추가했습니다.
- 검색 랭킹 규칙 회귀 방지를 위한 테스트를 추가했습니다.
- `packages/model/tests/test_search_ranking.py`
- `packages/model/tests/test_ocr_pipeline.py`

### Changed
- 이미지 검색 스택을 `BGE-VL-large`에서 `Qwen3-VL-Embedding-2B`로 전환했습니다.
- 텍스트 검색 스택은 `BGE-M3`를 유지하고, image/text 혼합 구조 위에 `Qwen3-VL-Reranker-2B` top-N 재정렬을 적용하도록 검색 경로를 개편했습니다.
- 한글 lexical tokenization, exact field boost, low-score cutoff를 추가해 exact 문자열 질의에서 순위 안정성을 높였습니다.

### Fixed
- `PaddleOCRVL` 초기화 실패 또는 `fused_rms_norm_ext` 부재 시 표준 `PaddleOCR`로 안전하게 폴백하도록 보강했습니다.
- PaddleOCR 3.x의 `ocr()`/`predict()` 호출 차이와 반환 포맷 차이로 인해 OCR 경로가 실패하던 문제를 완화했습니다.
- 이전에는 실질적으로 비활성 상태였던 rerank 인터페이스를 실제 learned reranker 경로로 연결했습니다.

## 2.2.0

### Added
- CI에서 모델 테스트 자동 실행을 추가했습니다.
- `.github/workflows/tests.yml`의 `model-tests` job이 `packages/model/tests`를 검증합니다.

### Changed
- 모델 런타임 로직 변경은 없고, 테스트 자동화 운영 경로를 보강했습니다.

## 2.1.0

### Added
- 모델 패키지 테스트를 추가했습니다.
- `packages/model/tests/test_metadata_normalizer.py`
- `packages/model/tests/test_tracker_dataset.py`
- 커버 범위: 메타 정규화 규칙, `None` 필드 처리, tracker CSV 파싱/필수 컬럼 검증/조회 동작

### Changed
- 모델 런타임 동작 변경은 없고, 회귀 방지용 테스트 체계를 보강했습니다.

## 2.0.0

### Added
- 캡셔너 백엔드 선택 로직을 확장했습니다.
- `CAPTIONER_BACKEND` 및 `ENABLE_CAPTIONER` 환경변수로 GPT/Qwen/비활성 모드를 제어할 수 있습니다.
- 기본 선택 정책을 추가했습니다: `OPENAI_API_KEY` 있으면 GPT, 없고 CUDA 있으면 Qwen, 둘 다 아니면 캡셔닝 비활성.
- OCR 파이프라인 fallback을 추가했습니다.
- `PaddleOCRVL` 미설치 시 `PaddleOCR`로 자동 전환
- 둘 다 없으면 빈 OCR 결과를 반환하고 파이프라인은 중단되지 않음
- 하이브리드 검색 스코어링을 강화했습니다.
- dense(image/ocr/caption) + lexical + spec 매칭 점수 혼합
- `lexical_hit`, `lexical_score`, `spec_match_score`, `caption_score`, `text_query_score`를 결과에 포함

### Changed
- `HybridSearchOrchestrator.search()`가 텍스트 질의의 스펙 토큰(예: 16V, 3A)과 part number 증거를 반영하도록 변경되었습니다.
- 모델별 이미지 후보 집계 및 정렬 로직을 보강해 다중 이미지 인덱싱 시 상위 후보 안정성을 높였습니다.
- 컬렉션/인덱스 운영 경로가 `packages/model` 기준으로 변경되었습니다.

### Fixed
- OCR 의존성이 없는 환경에서도 모듈 import 단계에서 즉시 실패하지 않도록 안전한 조건부 import로 정리했습니다.
- 스캔 문서/이미지 중심 입력에서 OCR 엔진 부재로 전체 파이프라인이 중단되는 문제를 완화했습니다.

### Breaking
- 모델 패키지 루트가 `smart-vision-model`에서 `packages/model`로 물리 이동되었습니다.
- 설치 경로가 `pip install -e smart-vision-model`에서 `pip install -e packages/model`로 변경되었습니다.

## 1.2.0

### Added
- `data_collection/tracker_dataset.py`를 도입해 `Category_Code`, `STD_MAKER_NAME`, `NON_STD_MODEL_NAME`, `STD_MODEL_NAME`를 정규화하고 결합한 `MODEL_NAME`을 제공, `index_tracker_model()`과 증분 스크립트가 공통 메타 소스를 이용하도록 했습니다.
- `HybridSearchOrchestrator.index_tracker_model()`을 추가하여 `data/images/<MODEL_ID>/` 디렉터리와 트래커 메타만으로 이미지/메타 데이터를 자동 인덱싱할 수 있게 했습니다.
- `scripts/index_tracker_incremental.py` 유틸리티를 제공해 트래커 CSV와 이미지 루트를 스캔하면서 새 모델 또는 새 이미지가 있을 때만 SHA256 해시 비교 후 증분 인덱싱을 수행합니다.
- 각 서브모듈(`hybrid_search_pipeline`, `data_collection`, `preprocessing`, `retrieval`, `search`, `scripts`)에 README를 추가해 구성과 사용 방법을 문서화했습니다.
- Qwen3-VL-8B-Instruct 기반 `Qwen3VLCaptioner`를 도입해 이미지당 의미 캡션을 생성하고 BGE-M3로 임베딩, `caption_parts` 컬렉션에 저장해 의미 기반 검색 채널을 추가했습니다.

### Changed
- `attrs_parts` 컬렉션 스키마에 `image_path` 필드를 포함시키고, 기존 이미지 사본과의 해시 비교를 통해 중복 업로드를 방지했습니다.
- 오케스트레이터가 모델 단위의 이미지 인덱스를 캐싱해 `model_id::img_###` 일련번호를 자동 증가시키고, 중복 PK 감지 시 새 번호를 할당하도록 변경했습니다.
- Tracker 기반 인덱싱 경로에서도 메타데이터 정규화가 일관되도록 `index_model_metadata()` 호출 시 description/model_name을 병행 저장합니다.
- 검색 시 이미지·OCR·캡션 세 채널을 `α·image + β·ocr + γ·caption`으로 융합하고, API/데모 응답에 `ocr_score`, `caption_score`, `text_query_score`, `caption_text` 등을 포함하도록 파이프라인을 전면 개편했습니다.
- `attrs_parts`/`model_texts`의 `ocr_text` 길이를 2048자로 확장하고 캡션 텍스트를 함께 저장해 장문 라벨과 설명 검색 시 손실이 없도록 했습니다.

### Fixed
- 기존 attrs 컬럼 수와 삽입 데이터 수가 다른 경우 발생하던 `DataNotMatchException`을 방지하기 위해 스키마 정의와 삽입 페이로드를 정비했습니다.
- 이미지 사본 해시 계산 시 파일 존재 여부와 예외를 로깅하여 손상된 경로나 삭제된 파일이 있어도 증분 작업이 중단되지 않도록 했습니다.

## 1.1.0

### Added
- Introduced the `model_texts` Milvus collection for model-level dense embeddings, allowing metadata-only items to participate in search.
- Aggregated OCR management now merges all image-level tokens per `model_id` before recomputing the shared embedding.

### Changed
- Image ingestion upserts the associated model record prior to storing image vectors, keeping metadata and OCR text in sync.
- Milvus collection schemas accept optional extra fields via `FieldSpec`, simplifying future schema extensions.
- Stored metadata text is preserved alongside aggregated OCR strings for display as well as embedding.

### Fixed
- Prevented duplicate OCR strings by normalising and deduplicating tokens while aggregating.

## 1.0.0

### Added
- Derived `text_corpus` that blends OCR output with structured metadata so dense text search can match maker, part number, and category cues in addition to raw OCR text.
- Utility helpers on `HybridMilvusIndex` to report collection statistics and drop collections, enabling downstream tooling (demo status tab) without duplicating Milvus logic.

### Changed
- Switched Milvus primary keys for `image_parts`, `text_parts`, and `attrs_parts` to VARCHAR and drive them from caller-provided `pk` values, allowing multiple images per `model_id`.
- Normalizer now preserves both `model_id` and `pk` values while standardising maker/part/category fields.
- Stored `ocr_text` now uses the combined corpus (metadata phrases + OCR text) for consistency with the new embedding.

### Fixed
- Prevented OCR token join failures by ensuring tokens are converted to strings before concatenation.
- Ensured Milvus vector dimensions are inferred from the underlying BGE-VL projection (768 dim) to avoid 1024-dimension schema mismatches.
