#!/bin/bash

# Set up strict error handling
set -euo pipefail

# Load environment variables from .env file
ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    if [[ -f ".env.example" ]]; then
        echo "Warning: $ENV_FILE not found. Copy .env.example -> .env if you need custom settings."
    else
        echo "Warning: $ENV_FILE not found. Using existing environment variables."
    fi
fi

# Install Python dependencies
REQ_FILE="requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing Python dependencies from $REQ_FILE"
    pip install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found. Skipping pip install."
fi

# Configure Python environment
# Ensure the project root is in PYTHONPATH for module imports
PROJECT_ROOT="$(pwd)"
if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT"* ]]; then
    echo "Adding project root to PYTHONPATH: $PROJECT_ROOT"
    export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT"
fi

# Default Milvus URI for local demo runs.
export MILVUS_URI="${MILVUS_URI:-tcp://localhost:19530}"

# Launch the Gradio demo application
echo "Starting Gradio demo application..."
python3 app.py
