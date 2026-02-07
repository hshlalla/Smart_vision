"""Pydantic schemas for internal catalog RAG APIs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CatalogIndexResponse(BaseModel):
    status: str = Field(..., description="Index operation status")
    document_id: str = Field(..., description="Assigned catalog document identifier")
    source: str = Field(..., description="Source name (usually PDF filename)")
    pages_indexed: int = Field(..., ge=0)
    chunks_indexed: int = Field(..., ge=0)


class CatalogSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Question or lookup text")
    top_k: int = Field(10, ge=1, le=50, description="Number of chunks to return")
    model_id: Optional[str] = Field(None, description="Optional model_id filter")
    part_number: Optional[str] = Field(None, description="Optional part_number filter")


class CatalogChunkResult(BaseModel):
    score: float
    lexical_score: float = 0.0
    spec_match_score: float = 0.0
    document_id: str
    source: str
    page: int
    chunk_id: int
    model_id: str
    part_number: str
    maker: str
    text: str


class CatalogSearchResponse(BaseModel):
    results: list[CatalogChunkResult]
