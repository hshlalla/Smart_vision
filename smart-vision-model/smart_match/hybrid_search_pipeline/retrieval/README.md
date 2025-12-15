### retrieval 폴더

- **`milvus_hybrid_index.py`**
  - 이미지(`image_parts`), 텍스트(`text_parts`), 속성(`attrs_parts`), 모델(`model_texts`) 컬렉션을 관리합니다.
  - **HNSW + COSINE** 인덱스를 기본으로 생성하며, `create_indexes()` → `load()` 순으로 사용됩니다.
  - `attrs_parts` 스키마에는 `model_id`, `maker`, `part_number`, `category`, `ocr_text`, `image_path`가 포함되어 있어 이미지 사본 위치를 추적할 수 있습니다.
  - `query_attributes_by_model()`로 특정 모델에 속한 모든 이미지 속성을 조회하고, `_fetch_attrs_for_hits()`는 검색 결과에서 빠른 메타 조회를 지원합니다.
  - `upsert_model()`은 모델 단위 텍스트 벡터를 갱신하기 전에 기존 행을 삭제하여 최신 정보를 유지합니다.

#### 인덱스 방식: HNSW + COSINE
- **HNSW (Hierarchical Navigable Small World)**
  - 그래프 기반 근사 최근접 탐색(ANN) 구조로, 벡터를 여러 레벨의 small-world 네트워크에 배치한다.
  - 삽입 시 `M`, `efConstruction` 파라미터로 그래프 연결 정도/정확도를 조절하며, 검색 시 `ef` 값을 키워 정확도를 높일 수 있다.
  - 수십만 이상 벡터에서도 탐색 복잡도가 로그 수준으로 유지되어 real-time 검색에 적합하다.
- **COSINE Distance**
  - BGE-VL/BGE-M3 임베딩이 L2 normalize되어 있으므로 방향성(코사인 유사도)이 가장 중요한 신호다.
  - Milvus에서는 `metric_type="COSINE"`을 지정하여 HNSW 그래프가 코사인 거리 기반으로 이웃을 정렬하도록 한다.
- 이 조합 덕분에 이미지·OCR·캡션 벡터를 동일한 설정으로 관리할 수 있으며, 파라미터만 조정해 속도/정확도 균형을 쉽게 맞춘다.

#### 사용 시 주의
- 스키마가 변경되면 기존 컬렉션을 드롭한 뒤 오케스트레이터를 다시 생성해야 충돌 없이 데이터가 삽입됩니다.
- `insert()`는 각 컬렉션에 동일한 PK 배열을 사용하므로, PK 개수와 벡터/속성 행 개수가 정확히 일치해야 합니다.
