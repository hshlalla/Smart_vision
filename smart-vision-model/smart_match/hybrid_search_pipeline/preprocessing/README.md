### preprocessing 폴더

- **`pipeline.py`**
  - 이미지 → OCR → 텍스트 임베딩까지 연결하는 전처리 파이프라인(`PreprocessingPipeline`)을 정의합니다.
  - 메타데이터를 표준화(`MetadataNormalizer`)한 뒤 이미지/텍스트 벡터, OCR 토큰, 텍스트 말뭉치를 반환합니다.

- **`metadata_normalizer.py`**
  - `maker`, `part_number`, `category` 등을 정규화하여 일관된 메타데이터를 제공합니다.

- **`ocr/OCR.py`**
  - `PaddleOCRVLPipeline`이 PaddleOCR-VL을 래핑합니다.
  - GPU 사용 여부(`use_gpu`), 각종 전처리 옵션을 그대로 전달할 수 있으며, VL 기능이 없으면 자동으로 기본 PaddleOCR로 폴백합니다.
  - Markdown 페이지, 마크다운 이미지, 텍스트 토큰까지 결과를 구성합니다.

- **`embedding/bge_vl_encoder.py` & `embedding/bge_m3_encoder.py`**
  - BAAI BGE-VL (이미지)과 BGE-M3 (텍스트) 모델을 로딩하고, fp16 + GPU 우선 모드로 임베딩을 생성합니다.
  - Torch 디바이스는 `cuda` 사용 가능 여부에 따라 자동 결정되며, 필요 시 생성자에서 직접 지정할 수 있습니다.

- **기타 하위 모듈**
  - `image_classification/` 및 `object_detection/`은 추가적인 비전 전처리(예: YOLOv8 탐지)를 위한 확장 지점입니다.

#### 특징
- 이미지 하나당 OCR → 텍스트 임베딩 → 이미지 임베딩이 순차적으로 실행되며, 결과는 CPU 텐서로 반환되어 Milvus에 삽입됩니다.
- OCR 토큰 병합 시 중복 라인이 제거되어 모델별 OCR 텍스트가 누적 관리됩니다.
