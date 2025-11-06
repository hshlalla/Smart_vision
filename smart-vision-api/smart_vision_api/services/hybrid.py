"""
Hybrid Search Service Layer

Provides a singleton wrapper around the HybridSearchOrchestrator defined in the
`smart_match` package so the FastAPI application can reuse model
resources across requests.
"""

from __future__ import annotations

import base64
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import UploadFile

from smart_match import HybridSearchOrchestrator
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
    FusionWeights,
    MilvusConnectionConfig,
)

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger("hybrid_service")


class HybridSearchService:
    """Lazy-initialized orchestrator wrapper for API usage."""

    def __init__(self) -> None:
        logger.info("Initializing HybridSearchOrchestrator for API service...")
        self._orchestrator = HybridSearchOrchestrator(
            milvus=MilvusConnectionConfig(uri=settings.MILVUS_URI),
            fusion_weights=FusionWeights(alpha=0.5, beta=0.3, gamma=0.2),
        )
        logger.info("HybridSearchOrchestrator ready.")

    @property
    def orchestrator(self) -> HybridSearchOrchestrator:
        return self._orchestrator

    def index_asset(self, image: UploadFile, metadata: Dict[str, str]) -> Dict[str, str]:
        """Persist uploaded asset by running the preprocessing pipeline."""
        with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix or ".jpg", delete=False) as tmp:
            tmp.write(image.file.read())
            tmp_path = Path(tmp.name)

        model_id = str(metadata.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("metadata must include 'model_id'.")
        enriched_metadata = dict(metadata)
        enriched_metadata["model_id"] = model_id
        enriched_metadata["pk"] = f"{model_id}::api_{uuid.uuid4().hex[:8]}"

        try:
            self._orchestrator.preprocess_and_index(tmp_path, enriched_metadata)
        finally:
            tmp_path.unlink(missing_ok=True)

        return {"status": "indexed"}

    def index_model_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        model_id = str(metadata.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("metadata must include 'model_id'.")
        self._orchestrator.index_model_metadata(model_id, metadata)
        return {"status": "registered"}

    def search(
        self,
        *,
        query_text: Optional[str],
        image_b64: Optional[str],
        top_k: int,
        part_number: Optional[str],
    ):
        """Execute hybrid search optionally using both modalities."""
        image_path = None
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                image_path = Path(tmp.name)
        try:
            results = self._orchestrator.search(
                query_image=image_path,
                query_text=query_text,
                top_k=top_k,
                part_number=part_number,
            )
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
        return results


hybrid_service = HybridSearchService()
