"""Pydantic schemas for hybrid search API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HybridIndexResponse(BaseModel):
    status: str = Field(..., description="Index operation status message")
    model_id: Optional[str] = Field(None, description="Stored model identifier")


class HybridMetadataDraft(BaseModel):
    model_id: str = Field("", description="Optional model ID. Empty means auto-allocate on confirm.")
    maker: str = Field("", description="Suggested maker metadata")
    part_number: str = Field("", description="Suggested part number metadata")
    category: str = Field("", description="Suggested category metadata")
    description: str = Field("", description="Suggested natural-language description")
    product_info: str = Field("", description="Suggested product type")
    price_value: Optional[int] = Field(None, description="Approximate price in KRW if inferred")
    source: str = Field("openai", description="Metadata generation source")


class HybridIndexPreviewRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_base64_list: List[str] = Field(default_factory=list, description="Optional list of base64 encoded images")


class HybridIndexPreviewResponse(BaseModel):
    status: str = Field(..., description="Preview generation status")
    draft: HybridMetadataDraft


class HybridIndexConfirmRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded representative image")
    image_base64_list: List[str] = Field(default_factory=list, description="Optional list of base64 encoded images")
    model_id: str = Field("", description="Optional model ID. Empty means auto-allocate.")
    maker: str = Field("", description="Maker metadata")
    part_number: str = Field("", description="Part number metadata")
    category: str = Field("", description="Category metadata")
    description: str = Field("", description="Description metadata")
    product_info: str = Field("", description="Optional product type")
    price_value: Optional[int] = Field(None, description="Optional estimated price")


class HybridSearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Text query string")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    part_number: Optional[str] = Field(None, description="Optional part number filter")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class HybridSearchResponse(BaseModel):
    results: list[Dict[str, object]]
