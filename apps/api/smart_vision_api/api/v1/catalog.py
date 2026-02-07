"""Catalog RAG API.

Endpoints:
    - POST /catalog/index_pdf : index internal PDF into vector store
    - POST /catalog/search    : semantic chunk retrieval from indexed catalogs
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...core.auth import require_user
from ...schemas.catalog import CatalogIndexResponse, CatalogSearchRequest, CatalogSearchResponse
from ...services.catalog import catalog_service

router = APIRouter(prefix="/catalog", tags=["Catalog RAG"])


@router.post("/index_pdf", response_model=CatalogIndexResponse, summary="Index catalog PDF")
async def index_pdf(
    _username: str = Depends(require_user),
    pdf: UploadFile = File(..., description="Catalog PDF file"),
    source: str = Form("", description="Optional source name"),
    model_id: str = Form("", description="Optional model_id metadata"),
    part_number: str = Form("", description="Optional part_number metadata"),
    maker: str = Form("", description="Optional maker metadata"),
) -> CatalogIndexResponse:
    try:
        result = catalog_service.index_pdf(
            pdf=pdf,
            source=source or None,
            model_id=model_id,
            part_number=part_number,
            maker=maker,
        )
        return CatalogIndexResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search", response_model=CatalogSearchResponse, summary="Search indexed catalog chunks")
async def search_catalog(
    request: CatalogSearchRequest,
    _username: str = Depends(require_user),
) -> CatalogSearchResponse:
    try:
        results = catalog_service.search(
            query_text=request.query_text,
            top_k=request.top_k,
            model_id=request.model_id,
            part_number=request.part_number,
        )
        return CatalogSearchResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

