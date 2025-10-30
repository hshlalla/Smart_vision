"""Pydantic schemas for hybrid search API."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HybridIndexResponse(BaseModel):
    status: str = Field(..., description="Index operation status message")


class HybridSearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Text query string")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    part_number: Optional[str] = Field(None, description="Optional part number filter")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class HybridSearchResponse(BaseModel):
    results: list[Dict[str, object]]
