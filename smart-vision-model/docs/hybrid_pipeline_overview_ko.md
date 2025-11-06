# Smart Vision 하이브리드 검색 파이프라인 정의서

## 1. 프로젝트 전반 개요
- **목표**: 반도체 중고 장비 사진과 텍스트 메타데이터를 결합한 멀티모달 검색/인덱싱 파이프라인을 제공.
- **핵심 컴포넌트**
  - `smart-vision-model`: 파이프라인 로직과 학습/전처리 모듈을 패키지화.
  - `smart-vision-api`: FastAPI 기반 REST 서비스로 오케스트레이터를 노출.
  - `smart-vision-demo`: Gradio UI로 검색 데모 제공.
- **데이터 저장소**: Milvus 2.4+ (벡터 + 속성 하이브리드 컬렉션), 이미지 사본은 로컬 `media/`.

## 2. 입력 파이프라인
### 2.1 데이터 소스
- **Tracker Dataset (`data/tracker_subset.csv`)**
  - 필수 필드: `Category_Code`, `STD_MAKER_NAME`, `MODEL_ID`, `NON_STD_MODEL_NAME`, `STD_MODEL_NAME`.
  - `TrackerDataset.from_csv()`가 레코드를 `TrackerRecord`로 정규화.
- **모바일 촬영 자산 (`data/images/<MODEL_ID>/*.jpg`)**
  - `mobile_capture_pipeline.py` 예시대로 QC Step → MinIO/S3 업로드 프로세스 적용 가능.

### 2.2 주요 입력 인터페이스
| 사용 시나리오 | 함수/엔드포인트 | 입력 형식 | 비고 |
|---------------|----------------|-----------|------|
| 단건 이미지 인덱싱 | `HybridSearchOrchestrator.preprocess_and_index` | 이미지 경로, `{"model_id", "maker", ...}` | 모델 ID 필수 |
| 트래커 기반 일괄 인덱싱 | `HybridSearchOrchestrator.index_tracker_model` | `model_id`, 이미지 루트 경로 | CSV에서 메타 자동 조인 |
| 커스텀 일괄 적재 | `HybridSearchOrchestrator.bulk_index` | `{"metadata": {...}, "images": [...]}` 배열 | `halt_on_error`로 실패 전략 제어 |
| REST 색인 API | `POST /api/v1/hybrid/index` | 멀티파트: 이미지 + 메타 필드 | 서비스 외부 연동 |

## 3. 전처리 및 임베딩 단계
### 3.1 메타데이터 정규화
- `MetadataNormalizer`가 `maker`, `part_number`, `category`, `description` 등을 표준화.
- 모델별 텍스트 설명은 `Maker`, `Part Number`, `Category`, `Description`을 조합해 생성.

### 3.2 이미지 & 텍스트 처리
- `PreprocessingPipeline`
  1. **OCR (`PaddleOCRVLPipeline`)**: 이미지에서 텍스트 추출 → 중복 라인 제거 후 누적 저장.
  2. **이미지 임베딩 (`BGEVLImageEncoder`)**: BGE-VL, FP16 + GPU 우선.
  3. **이미지 캡셔닝 (`Qwen3VLCaptioner`)**: Qwen3-VL-8B-Instruct로 의미 설명 캡션 생성.
  4. **텍스트 임베딩 (`BGEM3TextEncoder`)**: OCR 텍스트와 캡션을 각각 BGE-M3로 인코딩.
- 출력: `image_vector`, `ocr_vector`, `caption_vector`, `ocr_tokens`, `ocr_text`, `caption_text`, 정규화된 `metadata`.

### 3.3 중복 관리 및 기본 키
- 이미지 PK 패턴: `MODEL_ID::img_###`.
- 기존 PK 충돌 시 자동 증가 index로 재할당.
- 이미지 사본은 `MEDIA_ROOT`(기본 `media/`) 아래 `PK.png`로 저장.

## 4. Milvus 하이브리드 인덱스 구조
| 컬렉션 | 주요 필드 | 설명 |
|--------|-----------|------|
| `image_parts` | `pk`, `vector` | 이미지 임베딩 (HNSW + COSINE) |
| `text_parts` | `pk`, `vector` | OCR 라벨 임베딩 (BGE-M3) |
| `caption_parts` | `pk`, `vector` | Qwen3 캡션 임베딩 (BGE-M3) |
| `attrs_parts` | `pk`, `model_id`, `maker`, `part_number`, `category`, `ocr_text(≤2048)`, `caption_text(≤2048)`, `image_path` | 속성/메타 데이터 |
| `model_texts` | `model_id`, `vector`, `metadata_text`, `ocr_text`, `caption_text`, `maker`, `part_number`, `category`, `description` | 모델 단위 텍스트 + 캡션 표현 |
- `HybridMilvusIndex`가 컬렉션 생성 → 인덱스 생성 → 로드까지 자동 실행.
- `upsert_model()`로 모델 텍스트 벡터 최신화, `fetch_attributes()`/`fetch_models()`로 메타 조회.

## 5. 검색/결과 파이프라인
### 5.1 검색 질의 흐름
1. **이미지 질의**: BGE-VL 임베딩 → `image_parts`에서 ANN 검색 → 모델별 이미지 유사도 집계.
2. **OCR 유사도**: 질의 이미지에서 추출한 OCR 텍스트 → `text_parts` 검색 → 라벨 유사도 누적.
3. **캡션 의미 유사도**: Qwen3-VL 캡션 → `caption_parts` 검색 → 의미 기반 유사도 계산.
4. **사용자 텍스트 질의(선택)**: 입력 텍스트를 `model_texts`와 매칭해 OCR 가중치에 합산.
5. **퓨전 스코어**: `FusionWeights(alpha, beta, gamma)` 기준 `α·image + β·ocr + γ·caption`.
6. **Lexical Boost**: 질의 문자열이 메타/캡션/OCR에 포함되면 +0.25 보정.
7. **Part Number 필터**: 요청에 `part_number`가 있으면 일치 모델만 반환하며 `verified=True`.
8. **Cross-Encoder**: 현재 `noop`(0점) 자리, 향후 확장 가능.

### 5.2 API/데모 출력 포맷
```json
{
  "model_id": "SG-12345",
  "score": 0.87,
  "image_score": 0.82,
  "ocr_score": 0.71,
  "caption_score": 0.24,
  "text_query_score": 0.00,
  "maker": "SurplusGLOBAL",
  "part_number": "PN-001",
  "category": "ETCH",
  "description": "...",
  "metadata_text": "...",
  "ocr_text": "...",
  "caption_text": "...",
  "images": [
    {
      "image_id": "SG-12345::img_001",
      "similarity": 0.84,
      "image_path": "/app/media/SG-12345::img_001.png",
      "caption_text": "Front panel of the SG-12345 etch chamber with control labels."
    }
  ],
  "verified": true
}
```

### 5.3 REST 검색 인터페이스
| 엔드포인트 | 입력 | 주요 옵션 | 응답 |
|------------|------|-----------|------|
| `POST /api/v1/hybrid/search` | JSON `{ "query_text", "image_base64", "part_number", "top_k" }` | 텍스트/이미지 동시 질의, PN 필터 | 상위 `top_k` 후보 목록 |
| `POST /api/v1/hybrid/index` | 멀티파트 이미지 + 메타 | GPU OCR/임베딩 자동 실행 | `{"status": "indexed"}` |

## 6. 운영 및 확장 고려 사항
- **리소스**: 최초 실행 시 BGE/PaddleOCR 모델 가중치 다운로드 → GPU 메모리 확보 필요.
- **캡셔닝 모델**: Qwen3-VL-8B-Instruct 로딩 시 GPU 20GB+ 권장, 프롬프트/토큰 수는 환경에 맞게 조정.
- **텍스트 길이 관리**: `attrs_parts.ocr_text`/`caption_text`는 2048자까지 저장하며, 초과 시 안전하게 잘라냅니다.
- **Milvus 스키마 변경**: 변경 시 기존 컬렉션 드롭 후 오케스트레이터 재생성.
- **증분 인덱싱**: 기존 모델에 새로운 이미지 추가 시 OCR 텍스트는 중복 없이 병합.
- **로그/모니터링**: `smart-vision-api/logs/`에 서비스 로그 저장, 로거 레벨은 `LOG_LEVEL` 환경변수로 제어.
- **배포 경로**: `scripts/run_docker.sh`(Milvus 포함), `run_dev.sh`, `run_prod.sh` 스크립트 제공.

## 7. PT 활용 포인트
- 첫 장: 전체 아키텍처 다이어그램(데이터 수집 → 전처리 → Milvus → 검색/API)을 한 슬라이드에 정리.
- 중간 장: 입력(Tracker/모바일)과 전처리(임베딩/정규화) 관계 강조.
- 후반 장: 검색 흐름과 출력 예시를 JSON+UI 스크린샷으로 전달.
- 마무리: 운영상 주의사항(모델 다운로드, GPU, Milvus 관리)과 향후 확장(크로스 인코더, 재랭킹) 소개.
