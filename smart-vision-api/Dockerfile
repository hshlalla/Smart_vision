# Base image with CUDA support for ML model inference
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install Python 3.12.3 and required system dependencies
# Uses non-interactive frontend to prevent prompts during build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for application
WORKDIR /app

# Configure Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install production ASGI server
# Note: Installing separately to utilize layer caching
RUN pip install --no-cache-dir "uvicorn[standard]>=0.34.2"

# Install project dependencies from vendor directory
# Custom wheels are used to ensure reproducible builds
COPY vendor/wheels wheels/
RUN pip install --no-cache-dir wheels/*.whl

# Copy application source code
# Note: Ordered by change frequency to optimize layer caching
COPY pyproject.toml README.md ./
COPY smart_vision_api smart_vision_api/

# Install API package in development mode
RUN pip install --no-cache-dir --no-build-isolation .

# Create persistent storage directories
RUN mkdir -p models logs

# Configure application environment
ENV MODEL_DIR=/app/models \
    LOG_DIR=/app/logs \
    LOG_LEVEL=INFO \
    EQUIPMENT_CATEGORIZATION_DEVICE=cuda \
    EQUIPMENT_CATEGORIZATION_MODEL_PATH=equipment_categorization \
    RETRIEVER_DEVICE=cuda \
    PYTHONUNBUFFERED=1

# Configure container
EXPOSE 8000
VOLUME ["models", "logs"]

# Launch production server with optimized settings
CMD ["uvicorn", "smart_vision_api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--no-access-log"]
