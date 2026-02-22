#!/bin/bash

# Enable strict error handling for production environment
set -euo pipefail

# Resolve script and environment file paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables from .env file
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    export MILVUS_URI="${MILVUS_URI:-tcp://localhost:19530}"
else
    echo "Warning: $ENV_FILE not found. Using default production configuration."
fi

# Install Python dependencies
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing Python dependencies from $REQ_FILE"
    pip install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found. Skipping pip install."
fi

echo "Starting Milvus stack via docker compose..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d etcd minio standalone

# Configure production logging levels
echo "Setting up production logging configuration..."
export LOG_LEVEL="INFO"

# Configure Python environment and change to project root
cd "$PROJECT_ROOT"
if [[ ":${PYTHONPATH:-}:" != *":$PROJECT_ROOT"* ]]; then
    echo "Adding project root to PYTHONPATH: $PROJECT_ROOT"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"
fi

# Ensure log directory exists
LOG_DIR="$PROJECT_ROOT/${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
echo "Ensured log directory exists at: $LOG_DIR"

# Launch FastAPI production server
echo "Starting FastAPI production server..."
uvicorn smart_vision_api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info \
    --no-access-log
