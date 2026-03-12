#!/bin/bash

# Enable strict error handling mode
set -euo pipefail

# Get the absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$PROJECT_ROOT")")"
ENV_FILE="$PROJECT_ROOT/.env"

ensure_milvus_network() {
    if ! docker network inspect milvus >/dev/null 2>&1; then
        echo "Creating docker network: milvus"
        docker network create milvus >/dev/null
    fi
}

# Load environment variables from .env file
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "Warning: $ENV_FILE not found. Using default configuration."
fi

# Ensure required persistent storage directories exist
mkdir -p "$PROJECT_ROOT/${MODEL_DIR:-models}" "$PROJECT_ROOT/${LOG_DIR:-logs}"

if [[ "${SKIP_MILVUS:-0}" != "1" ]]; then
    ensure_milvus_network
    echo "Starting Milvus stack via docker compose..."
    docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d etcd minio standalone
else
    echo "SKIP_MILVUS=1, skipping Milvus startup."
fi

IMAGE_NAME="${IMAGE_NAME:-smart-vision-api}"
CONTAINER_NAME="${CONTAINER_NAME:-smart-vision-api}"

echo "Building smart-vision-api image from repo root: $REPO_ROOT"
docker build -f "$PROJECT_ROOT/Dockerfile" -t "$IMAGE_NAME" "$REPO_ROOT"

GPU_ARGS=()
if [[ "${ENABLE_GPU:-auto}" != "0" ]] && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "NVIDIA runtime detected. Running container with GPU support."
    GPU_ARGS=(--gpus all)
else
    echo "NVIDIA runtime not detected. Running container without GPU."
fi

ENV_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
    ENV_ARGS=(--env-file "$ENV_FILE")
fi

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --name "$CONTAINER_NAME" \
  "${ENV_ARGS[@]}" \
  -p "${HOST_PORT:-8000}:${CONTAINER_PORT:-8000}" \
  -v "$PROJECT_ROOT/models:/app/apps/api/models" \
  -v "$PROJECT_ROOT/logs:/app/apps/api/logs" \
  --network milvus \
  "$IMAGE_NAME"



echo "Container $CONTAINER_NAME successfully initialized and running"
