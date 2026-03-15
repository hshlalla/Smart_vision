# smart-match-model

`packages/model` 디렉터리는 **Smart Match** 프로젝트에서 사용하는 학습/실험 코드와
하이브리드 검색 파이프라인을 제공합니다.  
데이터 수집 → 전처리 → 임베딩 → Milvus 색인 → 검색·재랭킹까지 한 번에 구성할 수 있도록
모듈화되어 있으며, API/데모 서비스에서 동일한 패키지를 재사용합니다.

---

## 🔍 아키텍처 개요

```
Data Collection Layer
  📱 촬영(모바일) → QC 스크립트 → 업로드(S3/MinIO)

Preprocessing & Embedding Layer
  🔹 Vision Encoder (Qwen3-VL-Embedding-2B)
  🔹 OCR Engine (PaddleOCR-VL)
  🔹 Text Encoder (BGE-M3)
  🔹 Reranker (Qwen3-VL-Reranker-2B)
  🔹 Captioner (Qwen3-VL-2B-Instruct)
  🔹 Metadata Normalizer (Maker, PartNo, Category)

Milvus Hybrid Index Layer
  🧠 image_parts : Dense image vectors
  🧠 text_parts  : Dense OCR/text vectors
  🧠 attrs_parts : Structured metadata
  🧩 Fusion Retriever (vector + sparse + filter hybrid)

Search & Re-Ranking Layer
  1️⃣ Query Image → Qwen3-VL-Embedding-2B → image search top-K
  2️⃣ OCR Text / metadata / caption → BGE-M3 → text search top-K
  3️⃣ Fusion Score = α·cos(img) + β·cos(txt)
  4️⃣ Qwen3-VL-Reranker-2B top-N re-ranking
  5️⃣ Result Verification (OCR 일치도 + PN Match)
```

각 레이어는 `smart_match/` 패키지에서 모듈화되어 있으며,
`HybridSearchOrchestrator` 클래스로 전체 파이프라인을 실행할 수 있습니다.

추가로, 단일 API 인스턴스 운영을 전제로 “카테고리 prefix 기반 연번 model_id”를 위해
Milvus에 카운터 컬렉션(`sv_counters`)을 사용합니다.

---

## 📁 디렉터리 구조

```
packages/model/
├── smart_match/
│   ├── __init__.py
│   ├── data_collection/mobile_capture_pipeline.py
│   ├── preprocessing/
│   │   ├── embedding/
│   │   │   └── qwen3_vl_embedding.py
│   │   ├── metadata_normalizer.py
│   │   ├── ocr/OCR.py
│   │   └── pipeline.py
│   ├── retrieval/milvus_hybrid_index.py
│   ├── search/fusion_retriever.py
│   ├── search/qwen3_vl_reranker.py
│   └── hybrid_pipeline_runner.py
├── docs/releasenote.txt                 # 구조 및 변경 이력
├── pyproject.toml / requirements.txt    # 패키징 & 의존성
└── README.md
```

---

## ⚙️ 설치

```bash
# Python 3.12 이상 권장
python -m pip install --upgrade pip
pip install -e .
```

필요 의존성은 `requirements.txt`와 `pyproject.toml`에 정의되어 있습니다.  
PaddleOCR-VL, Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B, Qwen3-VL-2B-Instruct, BGE-M3 모델은 최초 실행 시 자동으로 가중치를 다운로드합니다.

---

## 🚀 사용 방법

### 1. 하이브리드 파이프라인 테스트

```python
from smart_match import HybridSearchOrchestrator
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import FusionWeights, MilvusConnectionConfig

orchestrator = HybridSearchOrchestrator(
    milvus=MilvusConnectionConfig(uri="tcp://localhost:19530"),
    fusion_weights=FusionWeights(alpha=0.6, beta=0.4),
)

model_id = orchestrator.allocate_model_id(category="ETCH")  # e000001 형태
metadata = {"model_id": model_id, "maker": "SG", "part_number": "PN-001", "category": "ETCH"}
orchestrator.preprocess_and_index("sample.jpg", metadata)

results = orchestrator.search(query_text="etch chamber", top_k=5)
print(results)
```

> **주의**  
> - Milvus 2.4+ 인스턴스가 실행 중이어야 하며, 기본 URI는 `tcp://localhost:19530` 입니다.  
> - 검색 전에 최소 하나 이상의 자산을 `preprocess_and_index`로 등록해주세요.

### OCR 실험 스위치

- `ENABLE_OCR=0`
  - 인덱싱/검색 양쪽 OCR을 모두 끕니다.
- `ENABLE_OCR_INDEXING=0`
  - 인덱싱 시 OCR만 끕니다.
- `ENABLE_OCR_QUERY=0`
  - 검색 시 query-time OCR만 끕니다.
- `HYBRID_IMAGE_COLLECTION`, `HYBRID_TEXT_COLLECTION`, `HYBRID_ATTRS_COLLECTION`, `HYBRID_MODEL_COLLECTION`, `HYBRID_CAPTION_COLLECTION`
  - 실험별로 Milvus 컬렉션을 분리할 때 사용합니다.

실험용 `C1`/`C2` 비교를 할 때는 OCR on/off 상태별로 별도 Milvus 컬렉션에 다시 인덱싱하는 것을 권장합니다.

### 2. API / 데모와 연동

`smart-match-model` 패키지는 API(`apps/api`)와 데모(`apps/demo`)에서 그대로 import하여 사용합니다.

```bash
# API / 데모 실행 전에 패키지를 설치하세요.
pip install -e ../../packages/model
```

자세한 정보는 각 디렉터리의 README를 참고하세요.

---

## 📚 추가 문서

- `docs/releasenote.txt`: 릴리스 노트 및 후속 작업 제안
- `smart_match/hybrid_pipeline_runner.py`: 엔드-투-엔드 오케스트레이션 예제
- `smart_match/data_collection/mobile_capture_pipeline.py`: 데이터 수집 계층 샘플 구조

---

## 🛡️ 라이선스

Proprietary — suhun.hong
