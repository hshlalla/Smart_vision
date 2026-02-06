"""
Hybrid Search API

Endpoints:
    - POST /hybrid/index : register new assets into Milvus using the preprocessing pipeline
    - POST /hybrid/search: perform multimodal search with optional query text and image
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...core.auth import require_user
from ...services.hybrid import hybrid_service
from ...schemas.payload import HybridIndexResponse, HybridSearchRequest, HybridSearchResponse

router = APIRouter(prefix="/hybrid", tags=["Hybrid Search"])


@router.post(
    "/index",
    summary="Index asset",
    description="Runs preprocessing (OCR + embeddings) and stores the result in Milvus.",
)
async def index_asset(
    _username: str = Depends(require_user),
    image: UploadFile = File(..., description="Equipment photo"),
    model_id: str = Form(..., description="Model ID"),
    maker: str = Form("", description="Maker metadata"),
    part_number: str = Form("", description="Part number metadata"),
    category: str = Form("", description="Category metadata"),
    description: str = Form("", description="Optional description"),
) -> HybridIndexResponse:
    try:
        metadata = {
            "model_id": model_id,
            "maker": maker,
            "part_number": part_number,
            "category": category,
            "description": description,
        }
        result = hybrid_service.index_asset(image, metadata)
        return HybridIndexResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/search",
    response_model=HybridSearchResponse,
    summary="Hybrid multimodal search",
)
async def search(
    request: HybridSearchRequest,
    _username: str = Depends(require_user),
) -> HybridSearchResponse:
    try:
        results = hybrid_service.search(
            query_text=request.query_text,
            image_b64=request.image_base64,
            top_k=request.top_k,
            part_number=request.part_number,
        )
        return HybridSearchResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
