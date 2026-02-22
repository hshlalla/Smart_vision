#!/bin/bash

# Enable strict error handling mode
set -euo pipefail

# Get the absolute paths
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
else
    echo "Warning: $ENV_FILE not found. Using default configuration."
fi

# Ensure required persistent storage directories exist
mkdir -p "$PROJECT_ROOT/${MODEL_DIR:-models}" "$PROJECT_ROOT/${LOG_DIR:-logs}"

echo "Starting Milvus stack via docker compose..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d etcd minio standalone

IMAGE_NAME="${IMAGE_NAME:-smart-vision-api}"
CONTAINER_NAME="${CONTAINER_NAME:-smart-vision-api}"

echo "Building smart-vision-api image (with GPU support)..."
docker build -t "$IMAGE_NAME" "$PROJECT_ROOT"

echo "Running $CONTAINER_NAME container with GPU enabled..."

docker run --rm -it \
  --gpus all \
  --name "$CONTAINER_NAME" \
  --env-file "$ENV_FILE" \
  -p "${HOST_PORT:-8000}:${CONTAINER_PORT:-8000}" \
  -v "$PROJECT_ROOT/models:/app/models" \
  -v "$PROJECT_ROOT/logs:/app/logs" \
  --network milvus \
  "$IMAGE_NAME"



echo "Container $CONTAINER_NAME successfully initialized and running"
