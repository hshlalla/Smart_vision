### 하이브리드 파이프라인 개요

- `hybrid_pipeline_runner.py`는 전체 파이프라인의 오케스트레이터입니다.  
  - BGE-VL 이미지 임베더, BGE-M3 텍스트 임베더, PaddleOCR-VL OCR 엔진을 초기화합니다.  
  - Milvus 컬렉션을 생성하고 로딩하며, 자동으로 `model_id::img_###` 형식의 PK를 할당해 중복 없이 이미지를 저장합니다.  
  - `index_model_metadata`, `preprocess_and_index`, `index_tracker_model`, `bulk_index` 등을 통해 메타데이터·OCR·이미지 임베딩을 결합합니다.
- `data_collection/`, `preprocessing/`, `retrieval/`, `search/` 하위폴더는 각각 입력 데이터, 전처리, Milvus 인덱스, 하이브리드 점수 결합 로직을 담당합니다(각 폴더 README 참조).
- `resources/`는 파이프라인에서 참조하는 보조 JSON 데이터를 저장합니다.

#### 주요 제공 기능
1. **트래커 CSV 연동**  
   - `data_collection/tracker_dataset.py`를 통해 `Category_Code`, `STD_MAKER_NAME`, `NON_STD_MODEL_NAME`, `STD_MODEL_NAME` 등을 로딩하고, 모델명은 비표준/표준명을 합친 문자열로 정규화합니다.  
   - `index_tracker_model()`은 `data/images/<MODEL_ID>/` 폴더에서 이미지들을 찾아 메타데이터와 함께 자동 인덱싱합니다.

2. **증분 이미지 처리**  
   - 이미지 PK 충돌을 검사하고, 기존에 저장된 이미지 해시와 비교해서 새로운 파일만 추가합니다.  
   - OCR 결과는 중복 텍스트 라인을 제거한 뒤 기존 결과에 병합합니다.

3. **Milvus 컬렉션 관리**  
   - `retrieval/milvus_hybrid_index.py`는 이미지/텍스트/속성/모델 컬렉션을 관리하며, `image_path` 컬럼까지 포함한 attrs 스키마를 사용합니다.  
   - `query_attributes_by_model()`로 특정 모델의 이미지 속성을 빠르게 조회할 수 있습니다.

4. **검색 파이프라인**  
   - 이미지·텍스트 질의를 모두 지원하며, 가중치(`FusionWeights`)로 최종 점수를 계산합니다.  
   - 필요 시 part number 필터링과 검증 플래그를 제공합니다.

#### 사용 시 참고
- Milvus 스키마가 변경되면 기존 컬렉션을 드롭하고 오케스트레이터를 재생성해 새 스키마를 적용하세요.  
- `TRACKER_DATASET_PATH`, `MEDIA_ROOT`, `MILVUS_URI` 등의 환경변수로 경로와 접속 정보를 조정할 수 있습니다.  
- GPU 사용 여부는 PaddleOCR 및 Torch 설정에 따라 자동/수동으로 조절할 수 있습니다 (`PaddleOCRVLPipeline(use_gpu=True/False)` 등).
