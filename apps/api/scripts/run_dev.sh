#!/bin/bash

# Enable strict error handling for development environment
set -euo pipefail

# Resolve script and environment file paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$PROJECT_ROOT")")"
MODEL_ROOT="$REPO_ROOT/packages/model"
ENV_FILE="$PROJECT_ROOT/.env"
DEFAULT_VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

resolve_python_bin() {
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        printf '%s\n' "$VIRTUAL_ENV/bin/python"
        return
    fi

    if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
        printf '%s\n' "$DEFAULT_VENV_PYTHON"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi

    command -v python3
}

ensure_milvus_network() {
    if ! docker network inspect milvus >/dev/null 2>&1; then
        echo "Creating docker network: milvus"
        docker network create milvus >/dev/null
    fi
}

install_python_packages() {
    local python_bin="$1"

    echo "Installing editable packages for API and model using $python_bin"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$python_bin" -e "$MODEL_ROOT" -e "$PROJECT_ROOT"
    else
        "$python_bin" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$python_bin" -m pip install -e "$MODEL_ROOT" -e "$PROJECT_ROOT"
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
    echo "Warning: $ENV_FILE not found. Using default development configuration."
fi

export MILVUS_URI="${MILVUS_URI:-tcp://localhost:19530}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
PYTHON_BIN="$(resolve_python_bin)"

echo "Using Python interpreter: $PYTHON_BIN"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    install_python_packages "$PYTHON_BIN"
else
    echo "SKIP_INSTALL=1, skipping editable package installation."
fi

if [[ "${SKIP_MILVUS:-0}" != "1" ]]; then
    ensure_milvus_network
    echo "Starting Milvus stack via docker compose..."
    docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d etcd minio standalone
else
    echo "SKIP_MILVUS=1, skipping Milvus startup."
fi

# Configure development logging levels for debugging
echo "Configuring debug logging settings..."
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

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

if [[ "${SKIP_SERVER:-0}" == "1" ]]; then
    echo "SKIP_SERVER=1, skipping FastAPI startup."
    exit 0
fi

# Launch FastAPI development server with hot reload
echo "Starting FastAPI development server..."
UVICORN_ARGS=(
    smart_vision_api.main:app
    --host 0.0.0.0
    --port 8001
    --log-level debug
)

if [[ "${UVICORN_RELOAD:-0}" == "1" ]]; then
    UVICORN_ARGS+=(--reload)
else
    echo "UVICORN_RELOAD=1 not set, starting without --reload to avoid PaddleX reinitialization issues."
fi

exec "$PYTHON_BIN" -m uvicorn "${UVICORN_ARGS[@]}"
