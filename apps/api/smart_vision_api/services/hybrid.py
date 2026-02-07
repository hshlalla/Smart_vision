"""
Hybrid Search Service Layer

Provides a singleton wrapper around the HybridSearchOrchestrator defined in the
`smart_match` package so the FastAPI application can reuse model
resources across requests.
"""

from __future__ import annotations

import base64
import hashlib
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import UploadFile

from smart_match import HybridSearchOrchestrator
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
    FusionWeights,
    MilvusConnectionConfig,
)

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger("hybrid_service")


class _TTLCache:
    def __init__(self, *, ttl_seconds: int, max_items: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_items
        self._items: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def get(self, key: str):
        now = time.time()
        item = self._items.get(key)
        if item is None:
            return None
        exp, value = item
        if now >= exp:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.time() + self._ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)


class HybridSearchService:
    """Lazy-initialized orchestrator wrapper for API usage."""

    def __init__(self) -> None:
        self._orchestrator: HybridSearchOrchestrator | None = None
        self._milvus_config = MilvusConnectionConfig(uri=settings.MILVUS_URI)
        self._fusion_weights = FusionWeights(alpha=0.5, beta=0.3, gamma=0.2)
        self._query_cache = _TTLCache(ttl_seconds=60, max_items=256)

    @property
    def orchestrator(self) -> HybridSearchOrchestrator:
        if self._orchestrator is None:
            logger.info("Initializing HybridSearchOrchestrator for API service...")
            self._orchestrator = HybridSearchOrchestrator(
                milvus=self._milvus_config,
                fusion_weights=self._fusion_weights,
            )
            logger.info("HybridSearchOrchestrator ready.")
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
            self.orchestrator.preprocess_and_index(tmp_path, enriched_metadata)
        finally:
            tmp_path.unlink(missing_ok=True)

        return {"status": "indexed"}

    def index_model_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        model_id = str(metadata.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("metadata must include 'model_id'.")
        self.orchestrator.index_model_metadata(model_id, metadata)
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
        image_key = hashlib.sha1((image_b64 or "").encode("utf-8")).hexdigest()[:16] if image_b64 else ""
        cache_key = hashlib.sha1(
            f"{query_text or ''}|{image_key}|{top_k}|{part_number or ''}".encode("utf-8")
        ).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        image_path = None
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                image_path = Path(tmp.name)
        try:
            results = self.orchestrator.search(
                query_image=image_path,
                query_text=query_text,
                top_k=top_k,
                part_number=part_number,
            )
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
        self._query_cache.set(cache_key, results)
        return results


hybrid_service = HybridSearchService()
