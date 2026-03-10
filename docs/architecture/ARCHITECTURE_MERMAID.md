# Smart Vision Architecture (Mermaid)

아래 다이어그램은 `흐름`이 보이도록 입력 → 처리 → 저장/응답 순서로 다시 정리했습니다.

## 1) 전체 Pipeline (End-to-End)

```mermaid
flowchart TD
  U[1. User] --> W[2. apps/web]
  W -->|JWT + JSON/FormData| API[3. apps/api FastAPI]

  API --> R1[3-1. Hybrid API]
  API --> R2[3-2. Catalog API]
  API --> R3[3-3. Agent API]

  R1 --> M[4. packages/model smart_match]
  R3 --> M
  R2 --> C[4-2. Catalog Service]

  subgraph MODEL["Model Core"]
    M --> P[Preprocessing]
    P --> O[OCR]
    P --> VI[Image Embedding BGE-VL]
    P --> VT[Text Embedding BGE-M3]
    P --> CAP[Captioning Optional]
    M --> F[Fusion + lexical/spec scoring]
  end

  F --> DB[(5. Milvus Collections)]
  C --> DB
  DB --> API
  API --> W
  W --> U

  R3 --> EXT1[OpenAI Optional]
  R3 --> EXT2[Web/gparts Optional]
```

## 2) Frontend 상세 (`apps/web`)

```mermaid
flowchart LR
  B[Browser] --> APP[src/App.tsx]
  APP --> AUTH[auth state]
  AUTH --> LAYOUT[AppShellLayout]

  LAYOUT --> IDX[IndexPage]
  LAYOUT --> SRCH[SearchPage]
  LAYOUT --> CHAT[AgentChatPage]
  LAYOUT --> CAT[CatalogPage]

  IDX --> U[src/utils/api.ts]
  SRCH --> U
  CHAT --> U
  CAT --> U

  U --> H1[/api/v1/hybrid/index]
  U --> H2[/api/v1/hybrid/search]
  U --> H3[/api/v1/agent/chat]
  U --> H4[/api/v1/catalog/index_pdf]
  U --> H5[/api/v1/catalog/search]

  H1 --> V1[Index success/fail]
  H2 --> V2[Top-K results + scores]
  H3 --> V3[Answer + sources + identified]
  H5 --> V4[Catalog chunk list + source/page]
```

## 3) Model 상세 (`packages/model/smart_match`)

```mermaid
flowchart TD
  INI[Index request image+metadata] --> ORCH[HybridSearchOrchestrator]
  INS[Search request image/text] --> ORCH

  ORCH --> NORM[MetadataNormalizer]
  ORCH --> OCR[OCR Pipeline]
  ORCH --> IMG[BGEVLImageEncoder]
  ORCH --> TXT[BGEM3TextEncoder]
  ORCH --> CAP[Captioner Optional]

  NORM --> IDX[HybridMilvusIndex]
  OCR --> IDX
  IMG --> IDX
  TXT --> IDX
  CAP --> IDX

  IDX --> COL[(image_parts/text_parts/attrs_parts/model_texts/caption_parts)]
  ORCH --> CNT[MilvusCounterStore]
  CNT --> COL

  COL --> SCORE[Dense score image/ocr/caption/text]
  ORCH --> LEX[Lexical score]
  ORCH --> SPEC[Spec match score]

  SCORE --> FINAL[Final score blend]
  LEX --> FINAL
  SPEC --> FINAL
  FINAL --> OUT[Top-K models + metadata + image list]
```

## 4) Sequence 상세 (`/api/v1/hybrid/index`)

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant Web as apps/web IndexPage
  participant API as apps/api /hybrid/index
  participant HS as services/hybrid.py
  participant Orch as HybridSearchOrchestrator
  participant Pre as Preprocessing Pipeline
  participant Milvus as Milvus

  User->>Web: 이미지 + model_id 입력 후 인덱싱 클릭
  Web->>API: POST multipart/form-data (image, model_id, maker...)
  API->>HS: index_asset(image, metadata)
  HS->>Orch: preprocess_and_index(tmp_image, metadata+pk)
  Orch->>Pre: OCR + image/text/caption embedding
  Pre-->>Orch: vectors + normalized metadata
  Orch->>Milvus: upsert image_parts/text_parts/attrs_parts/model_texts/caption_parts
  Milvus-->>Orch: insert/flush OK
  Orch-->>HS: success
  HS-->>API: {"status":"indexed"}
  API-->>Web: 200 OK
  Web-->>User: 인덱싱 완료 표시
```
