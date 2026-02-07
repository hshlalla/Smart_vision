"""Smart Vision API Service

FastAPI application providing ML-powered analytics for semiconductor equipment.
Implements scalable inference services with monitoring capabilities.

Technical Stack:
- Framework: FastAPI (async)
- Runtime: Python 3.12+
- ML Backend: PyTorch
- API Docs: OpenAPI 3.0

Core Services:
- Equipment Classification
  * Category prediction
  * Batch processing
  * Confidence scoring

System Features:
- Async request handling
- GPU acceleration
- Health monitoring
- Error tracking
"""

from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.v1 import agent, auth, catalog, hybrid
from .core.config import settings
from .core.logger import get_logger

# Configure logging
logger = get_logger()

# Initialize FastAPI application
app = FastAPI(
    title="Smart Vision API",
    description="ML-powered intelligent analytics for Smart Vision",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS for the web front-end
origins = settings.cors_origins_list
allow_credentials = False if origins == ["*"] else True
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers
app.include_router(
    hybrid.router,
    prefix=settings.API_PREFIX,
)
app.include_router(
    auth.router,
    prefix=settings.API_PREFIX,
)
app.include_router(
    agent.router,
    prefix=settings.API_PREFIX,
)
app.include_router(
    catalog.router,
    prefix=settings.API_PREFIX,
)


@app.get(
    "/",
    response_model=Dict[str, str],
    summary="Service Health Check",
    description="Verify system status and component health",
)
async def health_check() -> Dict[str, str]:
    """Check system operational status.

    Verifies:
    - API availability
    - Model readiness
    - Resource status
    - Version info

    Returns:
        Dict[str, str]: Health check results
    """
    return {
        "service": "Smart Vision API",
        "version": settings.VERSION,
        "status": "operational",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions.

    Provides consistent error responses for:
    - API errors
    - Model failures
    - Resource issues
    - System errors

    Args:
        request: The incoming request
        exc: The unhandled exception

    Returns:
        JSONResponse: Formatted error response
    """
    return JSONResponse(
        status_code=500, content={"error": "Internal Server Error", "detail": str(exc)}
    )
