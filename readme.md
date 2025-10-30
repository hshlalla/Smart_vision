# Smart Vision Project

A machine learning based intelligence platform for SurplusGLOBAL Smart Vision.

## Project Structure

```
smart-vision/
├── smart-vision-api/     # FastAPI service for model deployment
├── smart-vision-demo/    # Gradio web interface for model demonstration
└── smart-vision-model/   # ML model implementation and training
```

## Features

- **Equipment Categorization**: Automated categorization of semiconductor equipment
- **RESTful API**: FastAPI-based service for model inference
- **Interactive Demo**: Gradio web interface for model testing
- **Production Ready**: Docker support with CUDA acceleration

## Getting Started

### Prerequisites

- Python 3.12.3 or higher
- CUDA 12.6.3 (for GPU support)
- Docker 24.0.0 or higher
- transformers == 4.51.x 

### Installation

1. Clone the repository:
```bash
git clone -b PROD https://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/smart-vision
cd smart-vision
```

2. Install dependencies:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -e smart-vision-model
pip install -e smart-vision-api
```

### Running the Services
 **Note:** To run the Milvus container, you must execute `run_docker.sh` first.

#### API Service

```bash
cd smart-vision-api
./scripts/run_docker.sh  # docker mode
./scripts/run_dev.sh  # Development mode
./scripts/run_prod.sh # Production mode
```

#### Demo Interface

```bash
cd smart-vision-demo
./run_demo.sh
```

## Docker Deployment

```bash
cd smart-vision-api
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

## License

Proprietary - SurplusGLOBAL AI Team

## Contact

For questions and support, please contact the SurplusGLOBAL AI Team.
