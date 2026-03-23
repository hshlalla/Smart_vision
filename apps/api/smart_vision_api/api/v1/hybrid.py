"""
Hybrid Search API

Endpoints:
    - POST /hybrid/index : register new assets into Milvus using the preprocessing pipeline
    - POST /hybrid/search: perform multimodal search with optional query text and image
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...core.auth import require_user
from ...core.config import settings
from ...core.logger import get_logger
from ...services.hybrid import hybrid_service
from ...schemas.payload import (
    HybridIndexConfirmRequest,
    HybridIndexPreviewRequest,
    HybridIndexPreviewResponse,
    HybridIndexResponse,
    HybridIndexTaskResponse,
    HybridSearchRequest,
    HybridSearchResponse,
)

router = APIRouter(prefix="/hybrid", tags=["Hybrid Search"])
logger = get_logger("api.hybrid")


@router.post(
    "/index/preview",
    response_model=HybridIndexPreviewResponse,
    summary="Preview GPT metadata for an uploaded image",
)
async def preview_index_asset(
    request: HybridIndexPreviewRequest,
    _username: str = Depends(require_user),
) -> HybridIndexPreviewResponse:
    try:
        image_b64_list = [img for img in ([request.image_base64] if request.image_base64 else []) + list(request.image_base64_list or []) if img]
        if not image_b64_list:
            raise HTTPException(status_code=422, detail="at least one image is required.")
        if any(len(img or "") > settings.MAX_IMAGE_BASE64_LENGTH for img in image_b64_list):
            raise HTTPException(
                status_code=413,
                detail=f"image_base64 is too large (max {settings.MAX_IMAGE_BASE64_LENGTH} chars per image).",
            )
        logger.info(
            "POST /hybrid/index/preview received: image_count=%d image_base64_len=%d",
            len(image_b64_list),
            len(image_b64_list[0] or ""),
        )
        label_image_b64_list = [img for img in list(request.label_image_base64_list or []) if img]
        if any(len(img or "") > settings.MAX_IMAGE_BASE64_LENGTH for img in label_image_b64_list):
            raise HTTPException(
                status_code=413,
                detail=f"label_image_base64 is too large (max {settings.MAX_IMAGE_BASE64_LENGTH} chars per image).",
            )
        result = hybrid_service.preview_index_asset(
            image_b64_list=image_b64_list,
            metadata_mode=request.metadata_mode,
            label_image_b64_list=label_image_b64_list,
        )
        return HybridIndexPreviewResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /hybrid/index/preview failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/index/confirm",
    response_model=HybridIndexResponse,
    summary="Confirm metadata and store asset in Milvus",
)
async def confirm_index_asset(
    request: HybridIndexConfirmRequest,
    _username: str = Depends(require_user),
) -> HybridIndexResponse:
    try:
        image_b64_list = [img for img in ([request.image_base64] if request.image_base64 else []) + list(request.image_base64_list or []) if img]
        if not image_b64_list:
            raise HTTPException(status_code=422, detail="at least one image is required.")
        if any(len(img or "") > settings.MAX_IMAGE_BASE64_LENGTH for img in image_b64_list):
            raise HTTPException(
                status_code=413,
                detail=f"image_base64 is too large (max {settings.MAX_IMAGE_BASE64_LENGTH} chars per image).",
            )
        metadata = request.model_dump(exclude={"image_base64", "image_base64_list"})
        logger.info(
            "POST /hybrid/index/confirm received: image_count=%d model_id=%s maker=%s part_number=%s category=%s",
            len(image_b64_list),
            metadata.get("model_id", ""),
            metadata.get("maker", ""),
            metadata.get("part_number", ""),
            metadata.get("category", ""),
        )
        result = hybrid_service.confirm_index_asset(image_b64_list=image_b64_list, metadata=metadata)
        logger.info(
            "POST /hybrid/index/confirm queued: model_id=%s task_id=%s",
            result.get("model_id", ""),
            result.get("task_id", ""),
        )
        return HybridIndexResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /hybrid/index/confirm failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/index/tasks/{task_id}",
    response_model=HybridIndexTaskResponse,
    summary="Get async indexing task status",
)
async def get_index_task(
    task_id: str,
    _username: str = Depends(require_user),
) -> HybridIndexTaskResponse:
    try:
        result = hybrid_service.get_index_task(task_id)
        return HybridIndexTaskResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Indexing task not found.") from exc
    except Exception as exc:
        logger.exception("GET /hybrid/index/tasks/%s failed: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
        logger.info(
            "POST /hybrid/index received: filename=%s model_id=%s maker=%s part_number=%s category=%s",
            image.filename,
            model_id,
            maker,
            part_number,
            category,
        )
        metadata = {
            "model_id": model_id,
            "maker": maker,
            "part_number": part_number,
            "category": category,
            "description": description,
        }
        result = hybrid_service.index_asset(image, metadata)
        logger.info("POST /hybrid/index completed: model_id=%s status=%s", model_id, result.get("status"))
        return HybridIndexResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /hybrid/index failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/index/bulk_zip",
    response_model=HybridIndexResponse,
    summary="Bulk index an items ZIP archive",
)
async def index_bulk_zip(
    archive: UploadFile = File(..., description="ZIP archive containing items/<domain>/<item_id>/meta.json and images/*"),
    _username: str = Depends(require_user),
) -> HybridIndexResponse:
    try:
        logger.info(
            "POST /hybrid/index/bulk_zip received: filename=%s content_type=%s",
            archive.filename,
            archive.content_type,
        )
        result = hybrid_service.index_bulk_zip_archive(archive)
        logger.info(
            "POST /hybrid/index/bulk_zip queued: task_id=%s job_type=%s",
            result.get("task_id", ""),
            result.get("job_type", ""),
        )
        return HybridIndexResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /hybrid/index/bulk_zip failed: %s", exc)
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
        if request.image_base64 and len(request.image_base64) > settings.MAX_IMAGE_BASE64_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"image_base64 is too large (max {settings.MAX_IMAGE_BASE64_LENGTH} chars).",
            )
        logger.info(
            "POST /hybrid/search received: has_image=%s image_base64_len=%d query_len=%d part_number=%s top_k=%d",
            bool(request.image_base64),
            len(request.image_base64 or ""),
            len(request.query_text or ""),
            request.part_number or "",
            request.top_k,
        )
        results = hybrid_service.search(
            query_text=request.query_text,
            image_b64=request.image_base64,
            top_k=request.top_k,
            part_number=request.part_number,
            use_reranker=request.use_reranker,
        )
        logger.info("POST /hybrid/search completed: result_count=%d", len(results or []))
        return HybridSearchResponse(results=results)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /hybrid/search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
