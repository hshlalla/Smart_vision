Smart Vision API Release Notes
===============================

## 2.0.0

### Added
- `Catalog RAG` 기능을 API에 추가했습니다.
- 새 엔드포인트: `POST /api/v1/catalog/index_pdf`, `POST /api/v1/catalog/search`
- 카탈로그 스키마 추가: `CatalogIndexResponse`, `CatalogSearchRequest`, `CatalogSearchResponse`
- 카탈로그 서비스 추가:
- PDF 페이지 텍스트 추출 + 이미지 OCR fallback(스캔형 PDF 대응)
- 청크 인덱싱(`catalog_chunks`), `model_id`/`part_number`/`maker` 메타 저장
- 검색 시 dense+lexical+spec 혼합 스코어(`score`, `lexical_score`, `spec_match_score`)
- 검색/임베딩 TTL 캐시 도입으로 반복 질의 성능 개선
- Agent tool에 `catalog_search`를 추가해 내부 카탈로그 기반 질의응답이 가능해졌습니다.
- Agent 응답 후처리에서 카탈로그 근거(`source/page`)가 누락되면 자동으로 Evidence 블록을 붙이도록 강화했습니다.
- API 의존성 보강:
- `uvicorn[standard]`
- `numpy<2`(PyTorch/NumPy ABI 충돌 회피)
- `Pillow`, `paddleocr`, `paddlepaddle`, `pypdf`

### Changed
- `main.py` 라우터에 `catalog`를 정식 등록했습니다.
- `run_dev.sh`를 안정화했습니다.
- `PYTHONPATH` unbound 에러 방지(`PYTHONPATH:-`)
- 모델 루트(`packages/model`) 자동 추가
- `uvicorn` 실행 방식을 `python -m uvicorn`으로 변경
- 패키지 설치를 `python -m pip`로 통일
- Hybrid 서비스에 쿼리 캐시(60s) 및 이미지 base64 해시 기반 캐시 키를 적용했습니다.
- 레포 구조가 `apps/api` 기준으로 변경되었습니다.

### Fixed
- 개발 실행 시 `PYTHONPATH: unbound variable` 오류를 해결했습니다.
- 개발 실행 시 `uvicorn: command not found` 상황을 `python -m uvicorn`으로 우회했습니다.
- `smart_match`, `PIL`, `paddleocr`, `pypdf` 미설치로 인한 런타임 실패 경로를 의존성/경로 측면에서 정리했습니다.

### Breaking
- API 루트 경로가 `smart-vision-api`에서 `apps/api`로 물리 이동되었습니다.
- 모델 패키지 참조 경로가 `smart-vision-model`에서 `packages/model`로 변경되었습니다.

## 1.3.0

### Added
- 인덱싱 경로에 단계별 타이밍 로그를 추가해 API 호출 시 OCR/임베딩/flush 구간별 소요 시간을 확인할 수 있습니다.
- 기본 로깅 설정을 제공해 별도 설정이 없는 환경에서도 INFO 로그가 바로 출력되도록 개선했습니다.

### Changed
- 반복 업서트 시 모델 단위 컬렉션은 덮어쓰고 이미지 단위 컬렉션은 새 PK를 배정해 중복 저장을 방지하는 현재 동작을 명시했습니다.

## 1.2.0

### Added
- 백엔드 오케스트레이터 업데이트를 반영해 API 경로도 `index_tracker_model` 및 증분 인덱싱 스크립트에서 사용되는 자동 일련번호·메타데이터 병합 로직을 그대로 활용할 수 있게 되었습니다.
- `attrs_parts` 컬렉션에 저장되는 `image_path` 및 해시 기반 중복 검사를 통해 API 업로드 시에도 동일 이미지를 반복 저장하지 않습니다.
- 질의 이미지에서 Qwen3-VL-8B-Instruct로 생성한 캡션을 BGE-M3 임베딩으로 저장/검색해 의미 기반 리콜을 향상시켰습니다. API 응답에는 `ocr_score`, `caption_score`, `text_query_score`, `caption_text` 필드가 추가됩니다.

### Changed
- 이미지/텍스트/속성 컬렉션 스키마가 확장되었으므로 기존 Milvus 컬렉션을 드롭 후 재생성해야 API 업로드가 성공합니다.
- 메타데이터 업서트 시 description/model_name 필드를 병합 저장해 모델 단위 검색 결과가 더 풍부한 텍스트를 반환합니다.

## 1.1.0

### Added
- Exposed `index_model_metadata` service method so clients can register model records without uploading images.

### Changed
- Image ingestion path now upserts the model metadata first, keeping model-level embeddings in sync with image-level OCR updates.

## 1.0.0

### Added
- Server-side enforcement of `model_id` in `HybridSearchService.index_asset`; requests lacking the field now return a validation error.
- Automatic PK generation (`model_id::api_<uuid>`) so API uploads follow the new Milvus schema while keeping `model_id` available for grouping.

### Changed
- Metadata forwarded to the orchestrator now includes both `model_id` (business identifier) and `pk` (Milvus primary key), aligning API ingestion with demo/cli workflows.

### Fixed
- Prevented silent ingestion when clients omitted `model_id`, ensuring downstream collections never receive anonymous records.
