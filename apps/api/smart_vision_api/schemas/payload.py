"""Pydantic schemas for hybrid search API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HybridIndexResponse(BaseModel):
    status: str = Field(..., description="Index operation status message")
    model_id: Optional[str] = Field(None, description="Stored model identifier")
    task_id: Optional[str] = Field(None, description="Async indexing task identifier")
    job_type: str = Field("single", description="Queued indexing job type")


class HybridIndexTaskSummary(BaseModel):
    total_items: int = Field(0, ge=0, description="Total discovered item directories in a bulk ZIP job")
    processed_items: int = Field(0, ge=0, description="Number of items already processed")
    indexed_items: int = Field(0, ge=0, description="Number of items indexed successfully")
    failed_items: int = Field(0, ge=0, description="Number of items that failed during indexing")
    recent_indexed_model_ids: List[str] = Field(
        default_factory=list,
        description="Recent successfully indexed model identifiers",
    )
    recent_errors: List[str] = Field(
        default_factory=list,
        description="Recent bulk ZIP indexing errors",
    )


class HybridIndexTaskResponse(BaseModel):
    task_id: str = Field(..., description="Async indexing task identifier")
    status: str = Field(..., description="Task status")
    model_id: Optional[str] = Field(None, description="Resolved/stored model identifier")
    detail: str = Field("", description="Human-readable task detail")
    job_type: str = Field("single", description="Queued indexing job type")
    summary: Optional[HybridIndexTaskSummary] = Field(None, description="Optional bulk indexing progress summary")


class HybridMetadataDraft(BaseModel):
    model_id: str = Field("", description="Optional model ID. Empty means auto-allocate on confirm.")
    maker: str = Field("", description="Suggested maker metadata")
    part_number: str = Field("", description="Suggested part number metadata")
    category: str = Field("", description="Suggested category metadata")
    description: str = Field("", description="Suggested natural-language description")
    product_info: str = Field("", description="Suggested product type")
    price_value: Optional[int] = Field(None, description="Approximate price in KRW if inferred")
    source: str = Field("openai", description="Metadata generation source")


class HybridDuplicateCandidate(BaseModel):
    model_id: str = Field(..., description="Existing model identifier")
    maker: str = Field("", description="Existing maker metadata")
    part_number: str = Field("", description="Existing part number metadata")
    category: str = Field("", description="Existing category metadata")
    description: str = Field("", description="Existing description metadata")
    image_path: str = Field("", description="Representative stored image path")
    reason: str = Field("", description="Human-readable reason for duplicate detection")


class HybridIndexPreviewRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_base64_list: List[str] = Field(default_factory=list, description="Optional list of base64 encoded images")
    metadata_mode: str = Field("auto", description="Metadata generation mode: auto, gpt, local")
    label_image_base64_list: List[str] = Field(default_factory=list, description="Optional label-only images for OCR assistance")


class HybridIndexPreviewResponse(BaseModel):
    status: str = Field(..., description="Preview generation status")
    draft: HybridMetadataDraft
    ocr_image_indices: List[int] = Field(default_factory=list, description="Image indices recommended for OCR at confirm time")
    label_ocr_text: str = Field("", description="OCR text extracted from uploaded label images")
    duplicate_candidate: Optional[HybridDuplicateCandidate] = Field(
        None,
        description="Existing indexed model that appears to match the draft metadata",
    )


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
    ocr_image_indices: List[int] = Field(default_factory=list, description="Optional subset of image indices to OCR")


class HybridSearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Text query string")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    part_number: Optional[str] = Field(None, description="Optional part number filter")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    use_reranker: Optional[bool] = Field(None, description="Override reranker usage for this request")


class HybridSearchResponse(BaseModel):
    results: list[Dict[str, object]]
