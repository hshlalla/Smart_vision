from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message/question")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded image (no data: prefix)")
    max_tool_results: int = Field(5, ge=1, le=20, description="Max results returned by each tool")
    update_milvus: bool = Field(
        False,
        description="When true, persist enriched info back into Milvus after human review",
    )


class AgentChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Web sources used (title/url)")
    identified: Dict[str, Any] = Field(default_factory=dict, description="Best-effort identified product info")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Internal hybrid search candidate results")
    debug: Dict[str, Any] = Field(default_factory=dict)
