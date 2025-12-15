### preprocessing 폴더 개요

하이브리드 검색 인덱싱 직전에 실행되는 멀티모달 전처리 계층을 구성한다. 입력은 **이미지 경로 + 메타데이터 딕셔너리**, 출력은 `NormalizedRecord`로 캡슐화된 이미지/텍스트 임베딩과 정규화 메타데이터이다.

#### `pipeline.py`
- `PreprocessingPipeline`이 전체 스테이지를 오케스트레이션한다.
  1. `PaddleOCRVLPipeline.extract(image)`으로 OCR 토큰/합성 텍스트 수집.
  2. `BGEVLImageEncoder.encode(image)`로 이미지 벡터 생성.
  3. `MetadataNormalizer.normalize(metadata)`로 필드 단위 스펙 정규화.
  4. (옵션) `Qwen3VLCaptioner.generate(image)`로 캡션 텍스트 추가.
  5. `BGEM3TextEncoder.encode_document()`로 OCR 텍스트와 캡션을 각각 벡터화.
- 결과는 `NormalizedRecord`에 묶여 `image_vector`, `ocr_vector`, `caption_vector`, `metadata`, `ocr_tokens`, `ocr_text`, `caption_text`, `text_corpus`를 포함한다. 이후 인덱서가 그대로 Milvus 컬렉션(`image_collection`, `text_collection`, `model_ID`) 입력으로 사용한다.

#### `metadata_normalizer.py` 이부분을 정형화라고 말하는것이 옳다.
- `maker`: `strip().title()`로 공백 제거 + 각 단어 첫 글자를 대문자로 통일.
- `part_number`: 대문자 변환 후 `[^0-9A-Za-z]` 제거해 순수 알파넘 값만 유지.
- `category`: upper-case + trim.
- `model_id`, `pk`, `description`은 문자열화 후 공백 제거해 그대로 전달.
- 결과 딕셔너리는 downstream에서 필수/옵션 필드를 명확히 구분하도록 도와준다.

#### `ocr/OCR.py`
- `PaddleOCRVLPipeline`이 PaddleOCR-VL을 우선 사용하며, import 실패 시 자동으로 표준 `PaddleOCR` 인스턴스로 폴백한다.
- `score_threshold` 이하 토큰은 버리고, 페이지/라인 단위로 정제된 `OCRToken(text, score, box)` 리스트와 `combined_text`를 만든다.
- 경고를 주는 방법들을 생각해보는것으로 (마키나락스에서 진행하는것을 확인하고서 결정)
- VL 모델이 반환하는 markdown 페이지와 이미지도 조합해 `markdown_text`, `markdown_images`로 노출한다.
- `visualize()` 헬퍼로 박스/텍스트를 원본 이미지 위에 그려 디버깅 가능.

#### `captioning/qwen3_captioner.py`
- Qwen3-VL 계열 모델을 로드해 이미지 캡션을 생성한다.
- 기본 모델은 `Qwen/Qwen3-VL-4B-Instruct`, 실패 시 `Qwen/Qwen2-VL-7B-Instruct`로 폴백.
- GPU 사용 시 `dtype=float16` + `device_map=auto`로 VRAM을 절약하고, CPU에서는 float32로 자동 전환.
- 프롬프트, 샘플링 옵션, 최대 토큰 길이를 파라미터로 제어해 산업 장비 특화 설명을 생산한다.

#### `embedding/bge_vl_encoder.py`
- BAAI BGE-VL 모델을 이용해 이미지 임베딩을 생성한다.
- Torch 디바이스는 CUDA 우선이며, 입력 이미지를 전처리(`Resize`, `ToTensor`, 정규화) 후 FP16/FP32 텐서로 인코딩해 `torch.Tensor`를 반환한다.
- 출력 벡터는 L2 일반화되어 Milvus `image_collection`에 바로 삽입 가능.

#### `embedding/bge_m3_encoder.py`
- BGE-M3 텍스트 모델을 래핑해 문장/문단을 벡터화한다.
- `encode_document(text)`는 CLS 임베딩을, `encode_query(text)`는 검색 질의 최적화 임베딩을 제공한다.
- 길이 4096 토큰까지 처리하며, CUDA 가용 시 FP16으로 추론해 처리량을 높인다.

#### 기타 하위 모듈
- `image_classification/`, `object_detection/` 디렉터리는 확장 포인트로, 품목 식별이나 결함 감지 같은 추가 전처리 단계를 플러그인 형태로 붙일 수 있다.

### 전처리 특징 요약
- 이미지 하나당 **OCR → 메타데이터 정규화 → 캡션 → 텍스트/이미지 임베딩**이 순차 실행되고, 모든 산출물은 `NormalizedRecord`로 묶여 후속 인덱싱/검색 파이프라인에 전달된다.
- OCR 토큰 결합 시 중복을 제거하고, 캡션·메타데이터 문구를 합쳐 `text_corpus`를 만들기 때문에 검색 단계에서 다양한 텍스트 신호를 활용할 수 있다.
- 각 모듈은 예외 발생 시 로거나 안전한 폴백 경로를 갖고 있어, 부분 실패가 전체 파이프라인 중단으로 이어지지 않도록 설계되어 있다.
