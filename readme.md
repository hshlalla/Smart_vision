# Smart Vision Project

A machine learning based intelligence platform for Smart Vision.

## Project Structure

```
Smart_vision/
├── apps/
│   ├── web                              # Mobile-friendly web UI (primary UI)
│   ├── api                              # FastAPI service (hybrid search + agent + catalog RAG)
│   └── demo                             # (Optional) Gradio UI for quick debugging
├── packages/
│   └── model                            # ML/search pipeline package
├── data/
│   └── raw                              # Dataset root
└── docs/
    └── PROJECT_STRUCTURE.md            # Structure and navigation guide
```

Use canonical paths only (`apps/*`, `packages/*`, `data/*`).

## Features

- **Equipment Categorization**: Automated categorization of semiconductor equipment
- **RESTful API**: FastAPI-based service for hybrid search + agent bot
- **Web Front**: Mobile-friendly UI with login + chat
- **Production Ready**: Docker support with CUDA acceleration

## Getting Started

### Prerequisites

- Python 3.12.3 or higher
- CUDA 12.6.3 (for GPU support)
- Docker 24.0.0 or higher
- transformers == 4.51.x 

### Installation

1. Create virtual environment + install packages:
```bash
python -m venv .venv
source .venv/bin/activate

pip install -e packages/model
pip install -e apps/api
```

### Running the Services

Recommended run order:
1) Milvus (docker) → 2) API → 3) Front

#### 1) Milvus (docker)

```bash
cd apps/api
./scripts/run_docker.sh
```

#### 2) API Service

```bash
cd apps/api
export MILVUS_URI=tcp://localhost:19530
export AUTH_ENABLED=true
export AUTH_USERNAME=admin
export AUTH_PASSWORD=admin123
export OPENAI_API_KEY=...   # required for /api/v1/agent/chat

uvicorn smart_vision_api.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3) Front (mobile-friendly)

```bash
cd apps/web
npm install
echo 'VITE_API_BASE_URL=http://localhost:8000' > .env
npm run dev -- --host
```

Optional: Gradio demo (debug only)
```bash
cd apps/demo
./run_demo.sh
```

## Docker Deployment

```bash
cd apps/api
docker build -t smart-vision-api:latest .
docker run -d \
    --name smart-vision-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    smart-vision-api:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `MODEL_DIR` | Directory for model artifacts | models |
| `LOG_DIR` | Directory for log files | logs |
| `MILVUS_URI` | Milvus URI | `tcp://standalone:19530` |
| `MEDIA_ROOT` | Stored image copies (attrs image_path) | `media` |
| `AUTH_ENABLED` | Enable login/auth | `false` |
| `OPENAI_API_KEY` | LLM key for agent bot | (required for agent) |
| `COUNTERS_COLLECTION` | Milvus counters collection | `sv_counters` |

## License

Proprietary - suhun.hong

## Contact

For questions and support, please contact suhun.hong.
