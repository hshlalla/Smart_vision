# Smart Vision Architecture (Mermaid)

아래 다이어그램은 `흐름`이 보이도록 입력 → 처리 → 저장/응답 순서로 다시 정리했습니다.

## 1) 전체 Pipeline (End-to-End)

```mermaid
flowchart TD
  U["1. User"] --> W["2. apps/web"]
  W -->|"JWT + JSON/FormData"| API["3. apps/api FastAPI"]

  API --> R1["3-1. Hybrid API"]
  API --> R2["3-2. Catalog API"]
  API --> R3["3-3. Agent API"]

  R1 --> M["4. packages model smart_match"]
  R3 --> M
  R2 --> C["4-2. Catalog Service"]

  subgraph MODEL["Model Core"]
    M --> P["Preprocessing"]
    P --> O["OCR"]
    P --> VI["Image Embedding: Qwen3-VL"]
    P --> VT["Text Embedding: BGE-M3"]
    P --> CAP["Captioning Optional"]
    M --> F["Fusion and lexical/spec scoring"]
  end

  F --> DB[("5. Milvus Collections")]
  C --> DB
  DB --> API
  API --> W
  W --> U

  R3 --> EXT1["OpenAI Optional"]
  R3 --> EXT2["Web or gparts Optional"]
```

## 2) Frontend 상세 (`apps/web`)

```mermaid
flowchart LR
  B["Browser"] --> APP["src/App.tsx"]
  APP --> AUTH["auth state"]
  AUTH --> LAYOUT["AppShellLayout"]

  LAYOUT --> IDX["IndexPage"]
  LAYOUT --> SRCH["SearchPage"]
  LAYOUT --> CHAT["AgentChatPage"]
  LAYOUT --> CAT["CatalogPage"]

  IDX --> U["src/utils/api.ts"]
  SRCH --> U
  CHAT --> U
  CAT --> U

  U --> H1["/api/v1/hybrid/index/preview"]
  U --> H1B["/api/v1/hybrid/index/confirm"]
  U --> H1C["/api/v1/hybrid/index/tasks/<task_id>"]
  U --> H2["/api/v1/hybrid/search"]
  U --> H3["/api/v1/agent/chat"]
  U --> H4["/api/v1/catalog/index_pdf"]
  U --> H5["/api/v1/catalog/search"]

  H1 --> V1["Metadata preview and duplicate candidate"]
  H1B --> V1B["Queued indexing task"]
  H1C --> V1C["Task status polling"]
  H2 --> V2["Top-K results and scores"]
  H3 --> V3["Answer, sources, and identified item"]
  H5 --> V4["Catalog chunk list and source-page data"]
```

## 3) Model 상세 (`packages/model/smart_match`)

```mermaid
flowchart TD
  INI["Index request: image and metadata"] --> ORCH["HybridSearchOrchestrator"]
  INS["Search request: image or text"] --> ORCH

  ORCH --> NORM["MetadataNormalizer"]
  ORCH --> OCR["OCR Pipeline"]
  ORCH --> IMG["Qwen3VLImageEncoder"]
  ORCH --> TXT["BGEM3TextEncoder"]
  ORCH --> CAP["Captioner Optional"]

  NORM --> IDX["HybridMilvusIndex"]
  OCR --> IDX
  IMG --> IDX
  TXT --> IDX
  CAP --> IDX

  IDX --> COL[("image, text, attrs, model_texts, caption collections")]
  ORCH --> CNT["MilvusCounterStore"]
  CNT --> SQL[("Local SQLite counter namespace")]

  COL --> SCORE["Dense score from image, OCR, caption, and text"]
  ORCH --> LEX["Lexical score"]
  ORCH --> SPEC["Spec match score"]

  SCORE --> FINAL["Final score blend"]
  LEX --> FINAL
  SPEC --> FINAL
  FINAL --> OUT["Top-K models, metadata, and image list"]
```

## 4) Sequence 상세 (`/api/v1/hybrid/index/preview` + `/api/v1/hybrid/index/confirm`)

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant Web as "apps/web IndexPage"
  participant API as "apps/api hybrid index endpoints"
  participant HS as "services/hybrid.py"
  participant Orch as "HybridSearchOrchestrator"
  participant Pre as "Preprocessing Pipeline"
  participant Milvus as "Milvus"

  User->>Web: 이미지 업로드 후 preview 실행
  Web->>API: POST index preview with images and optional metadata
  API->>HS: preview_index_asset images and draft
  HS-->>API: metadata draft and optional duplicate candidate
  API-->>Web: preview response
  User->>Web: 필드 수정 후 confirm 실행
  Web->>API: POST index confirm with images and confirmed metadata
  API->>HS: confirm_index_asset images and metadata
  HS->>Orch: preprocess_and_index temp image and metadata
  Orch->>Pre: OCR and image text caption embedding
  Pre-->>Orch: vectors and normalized metadata
  Orch->>Milvus: upsert image text attrs model text and caption collections
  Milvus-->>Orch: insert/flush OK
  Orch-->>HS: task created and indexing executed in background
  HS-->>API: queued task response
  API-->>Web: 200 OK
  Web-->>User: 작업 상태 polling 시작
```

## 5) Catalog + Agent Orchestration Path

```mermaid
flowchart TD
  U["User question or search intent"] --> WEB["apps/web Agent or Catalog UI"]
  WEB --> API["apps/api"]

  API --> AG["Agent API"]
  API --> CAT["Catalog API"]
  API --> HY["Hybrid Search API"]

  AG --> DECIDE{"Need more evidence?"}

  DECIDE -->|"Inventory lookup"| HY
  DECIDE -->|"Internal document lookup"| CAT
  DECIDE -->|"Open-world or price context"| EXT["External web search optional"]

  HY --> MILVUS[("Milvus model and attrs collections")]
  CAT --> PDFIDX["Catalog PDF index and chunk store"]

  MILVUS --> HYRES["Identified item metadata and images"]
  PDFIDX --> CATRES["Catalog chunks with source and page"]
  EXT --> WEBRES["External links and summaries"]

  HYRES --> AGG["Agent response composer"]
  CATRES --> AGG
  WEBRES --> AGG

  AGG --> APIRESP["Unified response body"]
  APIRESP --> WEB
  WEB --> OUT["Answer text, item card, catalog evidence, external sources"]
```
