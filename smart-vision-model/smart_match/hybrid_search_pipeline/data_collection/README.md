### data_collection 폴더

- **`tracker_dataset.py`**
  - `TrackerDataset.from_csv()`로 `data/tracker_subset.csv` 형태의 메타데이터를 로딩합니다.
  - 필수 컬럼: `Category_Code`, `STD_MAKER_NAME`, `MODEL_ID`, `NON_STD_MODEL_NAME`, `STD_MODEL_NAME`.
  - 비표준/표준 모델명을 결합해 `MODEL_NAME`을 생성하고, `TrackerRecord`로 정규화된 메타를 제공합니다.
  - 오케스트레이터의 `index_tracker_model()` 및 증분 스크립트에서 사용됩니다.

- **`mobile_capture_pipeline.py`**
  - 모바일 기기에서 수집한 자산을 QC 후 S3/MinIO로 업로드하는 예시 파이프라인입니다.
  - `QCStep` 프로토콜로 품질 검증 스텝을 정의하고, `BaseClient.upload_file()`을 통해 객체 스토리지에 저장합니다.

#### 활용 패턴
1. `TrackerDataset`으로 모델 메타를 로딩 → `HybridSearchOrchestrator.index_tracker_model()` 또는 `scripts/index_tracker_incremental.py`에서 사용.
2. 모바일 캡처 파이프라인은 실제 서비스에서 사전 수집/검수 용도로 활용 가능하며, 저장소는 MinIO(S3 호환) 기반입니다.
