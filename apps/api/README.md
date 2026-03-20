# Smart Vision API

`apps/api` provides the FastAPI backend for Smart Vision.
It serves hybrid search, async indexing, authentication, agent chat, catalog search, and media files.

## Main Capabilities

- Auth
  - `GET /api/v1/auth/status`
  - `POST /api/v1/auth/login`
  - `GET /api/v1/auth/me`
- Hybrid indexing and search
  - `POST /api/v1/hybrid/index/preview`
  - `POST /api/v1/hybrid/index/confirm`
  - `GET /api/v1/hybrid/index/tasks/{task_id}`
  - `POST /api/v1/hybrid/search`
- Agent bot
  - `POST /api/v1/agent/chat`
- Catalog RAG
  - `POST /api/v1/catalog/index_pdf`
  - `POST /api/v1/catalog/search`
- Media serving
  - `/media/{filename}`

## Current Behavior

### Indexing Flow

1. `preview`
   - Builds a metadata draft from uploaded images.
   - Backend can be GPT or local Qwen depending on configuration and UI override.
   - Optional label-only OCR inputs can be uploaded from the UI and supplied as stronger evidence.
   - Label OCR is only used as preview-time evidence for metadata generation.
   - Preview also checks whether an existing indexed part already matches the draft `part_number` (and maker when available).
   - If a likely duplicate is found, the API returns a duplicate candidate so the UI can ask whether the new upload should be appended to the existing model.
2. User review
   - User can edit maker, part number, category, description, product info, and price.
   - User can explicitly choose:
     - keep the upload as a new model, or
     - append it to an existing indexed model when the system finds a likely duplicate.
3. `confirm`
   - Returns immediately with `task_id`.
   - Actual indexing runs in a background worker.
   - Frontend polls task status until `completed` or `failed`.
   - Task states are `queued`, `running`, `completed`, and `failed`.

### Search Flow

- Text-only search
  - Uses the lightweight `BGE-M3 + model collection` path.
  - Returns representative images from attrs records.
- Image search
  - Uses the multimodal orchestrator.
- Reranker
  - Can be toggled from the Search UI.
  - Still requires backend `ENABLE_RERANKER=1` to actually initialize the reranker.

### Agent Chat

- First tries internal search for direct inventory lookup.
- Part numbers are normalized so queries like `91200 4F310`, `91200-4F310`, and `912004F310` resolve to the same candidate.
- If an internal match is present, the response includes identified item metadata and image data for the frontend.
- If needed, the agent can still use external web search or catalog search.

### Catalog RAG

- Catalog PDF indexing supports two practical modes from the UI:
  - PDF text mode
    - Extracts text directly from the PDF and embeds it.
    - Best for text-native PDFs.
  - PaddleOCR-VL mode
    - Renders each page as an image and uses PaddleOCR-VL to recover markdown-like text with better table/layout preservation.
    - Better for scanned PDFs, image-heavy pages, and table-centric catalogs.
- In PaddleOCR-VL mode, extracted markdown-style table text is stored as chunk text and can later be cited by search or agent answers.

### Agent Response Composition

When the agent uses multiple tools in one turn, the final answer is composed from multiple evidence channels:

- Internal hybrid search
  - Used for registered inventory lookup.
  - Produces the matched product candidate with fields like `model_id`, `maker`, `part_number`, `category`, `description`, and representative `images`.
  - The API exposes the best accepted internal match separately as `identified` in the response body so the frontend can show product metadata and image cards.
- Catalog RAG
  - Used for internal PDF manuals, catalogs, and spec documents.
  - Produces chunk-level evidence with `source` and `page`.
  - The natural-language answer may summarize this evidence inline.
  - If the answer text omits document citations, the API appends a `Catalog Evidence` section with source/page references.
- External web search
  - Used only when the agent needs open-world or price information beyond the internal inventory/catalog data.
  - Returned as `sources` with title and URL.

In practice, this means a single chat response can contain:

1. A natural-language summary answer.
2. An internal matched item card via `identified`.
3. Internal PDF evidence via catalog citations.
4. External links in `sources` when web search was used.

Typical combination pattern:

- `hybrid_search`
  - identifies which registered part best matches the question
- `catalog_search`
  - explains what the indexed document says about that part
- final answer
  - merges both into one response, e.g. "this appears to match part X, and the internal catalog/manual describes it as Y on page Z"

### Duplicate Review and Merge Rationale

The duplicate review step was added for a practical field scenario:

- the same physical part can be uploaded more than once,
- a later upload may have better label visibility, cleaner images, or richer seller metadata,
- and blindly creating a new `model_id` would fragment search results and inventory history.

Because of that, the current design prefers human-reviewed merge behavior for interactive indexing:

- `preview` warns when an existing indexed part with the same normalized `part_number` is found,
- the user can decide whether to append the new images/metadata to the existing model,
- and `confirm` uses that choice by keeping the existing `model_id` when the user accepts the merge.

For offline dataset ingestion, the batch script uses a stronger automatic policy:

- first match on normalized `maker + part_number`,
- then fall back to normalized `part_number` when that value is unique,
- skip exact duplicate image paths,
- append genuinely new images to the existing model,
- and merge richer text fields instead of discarding them.

## Run

### Recommended

```bash
cd apps/api
source ../../.venv/bin/activate
./scripts/run_dev.sh
```

Notes:

- `run_dev.sh` loads `apps/api/.env`.
- Default API port is `8001`.
- It also starts the Milvus docker compose stack if available.

### Direct uvicorn

```bash
uvicorn smart_vision_api.main:app --host 0.0.0.0 --port 8001 --env-file .env
```

## Important Environment Variables

### Core

- `MILVUS_URI`
- `MEDIA_ROOT`
- `MAX_IMAGE_BASE64_LENGTH`
- `AUTH_ENABLED`
- `AUTH_USERNAME`
- `AUTH_PASSWORD`

### OCR

- `ENABLE_OCR=0`
  - Disable OCR for both indexing and query-time search.
- `ENABLE_OCR_INDEXING=0`
  - Disable OCR only during indexing.
- `ENABLE_OCR_QUERY=0`
  - Disable OCR only during query-time search.

### Metadata Preview / Caption

- `LOCAL_MODE`
  - `0`: hosted mode preferred
  - `1`: local mode preferred
- `METADATA_PREVIEW_BACKEND`
  - `auto`
  - `openai`
  - `qwen`
- `CAPTIONER_BACKEND`
  - `auto`
  - `gpt`
  - `qwen`
  - `none`

Resolution rule:

- `LOCAL_MODE=0` and `OPENAI_API_KEY` exists
  - metadata preview: GPT
  - captioning: GPT when captioning is enabled
- `LOCAL_MODE=1`
  - metadata preview: Qwen
  - captioning: Qwen when captioning is enabled

Request-scoped UI override:

- The Index UI can explicitly request:
  - `Auto`
  - `GPT`
  - `Local`
- `Local` maps to the Qwen preview path.

Recommended caption policy:

- Current hosted/default mode:
  - `CAPTIONER_BACKEND=gpt`
- Later local-only experiments:
  - `CAPTIONER_BACKEND=qwen`
- If `CAPTIONER_BACKEND=auto`, the runner resolves:
  - `LOCAL_MODE=1` -> `qwen`
  - `LOCAL_MODE=0` and `OPENAI_API_KEY` present -> `gpt`

### Search Quality / Performance

- `ENABLE_RERANKER`
- `RERANKER_DEVICE`
- `RERANKER_MAX_LENGTH`
- `WARMUP_QWEN_PREVIEW_ON_STARTUP`
- collection names:
  - `HYBRID_IMAGE_COLLECTION`
  - `HYBRID_TEXT_COLLECTION`
  - `HYBRID_ATTRS_COLLECTION`
  - `HYBRID_MODEL_COLLECTION`
  - `HYBRID_CAPTION_COLLECTION`

## Apple Silicon Notes

- PyTorch paths use `cuda -> mps -> cpu`.
- MPS works, but heavy multimodal local inference is still much slower than a CUDA workstation.
- For local experiments on Apple Silicon, the current practical setup is often:
  - OCR off
  - text-only fast path on
  - reranker off by default
  - GPT metadata preview for quality, or Qwen for local-only comparison
- The multimodal reranker is now wired to the official Qwen loading path, but this machine can still require `RERANKER_DEVICE=cpu` because `mps` may crash during real reranker scoring.
- If reranker is forced onto `cpu`, expect query latency to increase sharply.
- If OCR env flags are disabled, label OCR uploads from the UI are also bypassed at runtime.

## Stored Metadata

User-facing draft fields accepted by confirm:

- `model_id`
- `maker`
- `part_number`
- `category`
- `description`
- `product_info`
- `price_value`

Structured fields persisted in search/index collections are currently centered on:

- `maker`
- `part_number`
- `category`
- `description`
- `metadata_text`
- `ocr_text`
- `caption_text`
- `image_path`

`product_info` and `price_value` currently act more like enrichment inputs than fully expanded first-class retrieval columns.

When duplicate items are merged into an existing model, the current ingestion/indexing path tries to preserve richer later data:

- `description` is merged instead of blindly overwritten,
- `metadata_text` is merged and re-embedded,
- `caption_text` is accumulated across images,
- and newly uploaded images are added under the existing `model_id`.

## API Examples

### Login

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### Preview

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/hybrid/index/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64_list": ["..."],
    "metadata_mode": "auto"
  }'
```

### Confirm

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/hybrid/index/confirm" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64_list": ["..."],
    "metadata": {
      "maker": "Hyundai",
      "part_number": "91200-4F310",
      "category": "ACTUATOR",
      "description": "example"
    }
  }'
```

### Search

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/hybrid/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "91200 4F310",
    "top_k": 10,
    "use_reranker": false
  }'
```

### Agent Chat

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "91200 4F310 찾아줘",
    "update_milvus": false
  }'
```
