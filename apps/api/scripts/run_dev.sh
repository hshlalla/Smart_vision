#!/bin/bash

# Enable strict error handling for development environment
set -euo pipefail

# Resolve script and environment file paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$PROJECT_ROOT")")"
MODEL_ROOT="$REPO_ROOT/packages/model"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables from .env file
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    export MILVUS_URI="tcp://localhost:19530"
else
    echo "Warning: $ENV_FILE not found. Using default development configuration."
fi

# Install Python dependencies
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing Python dependencies from $REQ_FILE"
    python -m pip install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found. Skipping pip install."
fi

echo "Starting Milvus stack via docker compose..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d etcd minio standalone

# Configure development logging levels for debugging
echo "Configuring debug logging settings..."
export LOG_LEVEL="DEBUG"

# Set up Python environment
# Add project root to PYTHONPATH and change to project root directory
cd "$PROJECT_ROOT"
if [[ ":${PYTHONPATH:-}:" != *":$PROJECT_ROOT"* ]]; then
    echo "Adding project root to PYTHONPATH: $PROJECT_ROOT"
    echo "PYTHONPATH before: ${PYTHONPATH:-}"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"
fi

if [[ -d "$MODEL_ROOT" && ":${PYTHONPATH:-}:" != *":$MODEL_ROOT"* ]]; then
    echo "Adding model root to PYTHONPATH: $MODEL_ROOT"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$MODEL_ROOT"
fi

# Launch FastAPI development server with hot reload
echo "Starting FastAPI development server..."
python -m uvicorn smart_vision_api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --log-level debug
