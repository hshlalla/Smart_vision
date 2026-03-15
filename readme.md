# Smart Vision Project

Smart Vision is a local-first hybrid retrieval system for indexed industrial parts.
It combines a FastAPI backend, a web UI, and a reusable model package for image/text retrieval,
catalog search, async indexing, and agent-assisted lookup.

## Project Structure

```text
Smart_vision/
├── apps/
│   ├── api                              # FastAPI service
│   ├── web                              # Main web UI
│   └── demo                             # Optional Gradio demo
├── packages/
│   └── model                            # Hybrid retrieval / indexing package
├── docs/                                # Architecture, plans, reports, release notes
├── submission/                          # Submission artifacts and evidence
└── data/                                # Local data and experiments
```

## Current Runtime Model

- Search
  - Text-only queries use a lightweight `BGE-M3 + Milvus model collection` path.
  - Image queries use the heavier multimodal path.
- Indexing
  - `preview` creates metadata drafts.
  - `confirm` runs async background indexing and returns a `task_id`.
  - Task state is polled from `/api/v1/hybrid/index/tasks/{task_id}`.
- Agent chat
  - Uses internal hybrid search first.
  - Can still call web/catalog tools when needed.
  - If an internal match is available, the UI now shows the matched item image.
- Local vs hosted metadata/caption generation
  - `LOCAL_MODE=0` and `OPENAI_API_KEY` set: GPT-backed metadata preview and captioning.
  - `LOCAL_MODE=1`: Qwen-backed local preview/captioning.
- OCR
  - Controlled by env flags.
  - Current local experiments often keep OCR disabled because Apple Silicon runs it slowly and inconsistently for this workload.

## Recommended Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- Node.js / npm
- Apple Silicon is supported via MPS, but this is not CUDA-equivalent performance

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e packages/model
pip install -e apps/api
```

### Run Order

1. Milvus
```bash
cd apps/api
export PATH="$HOME/.local/bin:$PATH"
docker compose up -d
```

2. API
```bash
cd apps/api
source ../../.venv/bin/activate
./scripts/run_dev.sh
```

3. Web
```bash
cd apps/web
export PATH="$HOME/.local/bin:$PATH"
npm install
npm run dev -- --host 0.0.0.0
```

## Important Defaults

- API default port: `8001`
- Web dev port: `5173`
- Milvus URI for local host: `tcp://localhost:19530`
- Media files are served from `/media/...`
- The frontend should point to the machine IP, not `127.0.0.1`, when accessed from another device

## Key Environment Variables

See [apps/api/README.md](/Users/studio/Downloads/project/Smart_vision/apps/api/README.md) for the full API matrix.

- `AUTH_ENABLED`
- `AUTH_USERNAME`
- `AUTH_PASSWORD`
- `MILVUS_URI`
- `ENABLE_OCR`
- `ENABLE_OCR_INDEXING`
- `ENABLE_OCR_QUERY`
- `LOCAL_MODE`
- `METADATA_PREVIEW_BACKEND`
- `CAPTIONER_BACKEND`
- `ENABLE_RERANKER`
- `WARMUP_QWEN_PREVIEW_ON_STARTUP`

## Notes

- Apple Silicon support is implemented through `cuda -> mps -> cpu` device selection for PyTorch paths.
- The project now uses SQLite for `model_id` counters instead of a Milvus counter collection.
- Text-only search was separated from the heavy multimodal runtime to reduce latency and memory pressure.
