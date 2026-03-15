"""
Hybrid Search Service Layer

Provides a singleton wrapper around the HybridSearchOrchestrator defined in the
`smart_match` package so the FastAPI application can reuse model
resources across requests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from fastapi import UploadFile
from PIL import Image

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger("hybrid_service")

if TYPE_CHECKING:
    from smart_match import HybridSearchOrchestrator
    from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
        FusionWeights,
        MilvusConnectionConfig,
    )


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
        self._milvus_config = None
        self._fusion_weights = None
        self._query_cache = _TTLCache(ttl_seconds=60, max_items=256)

    @property
    def orchestrator(self) -> HybridSearchOrchestrator:
        if self._orchestrator is None:
            from smart_match import HybridSearchOrchestrator
            from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
                FusionWeights,
                MilvusConnectionConfig,
            )

            logger.info("Initializing HybridSearchOrchestrator for API service...")
            if self._milvus_config is None:
                self._milvus_config = MilvusConnectionConfig(uri=settings.MILVUS_URI)
            if self._fusion_weights is None:
                self._fusion_weights = FusionWeights(alpha=0.5, beta=0.3, gamma=0.2)
            self._orchestrator = HybridSearchOrchestrator(
                milvus=self._milvus_config,
                image_collection=settings.HYBRID_IMAGE_COLLECTION,
                text_collection=settings.HYBRID_TEXT_COLLECTION,
                attrs_collection=settings.HYBRID_ATTRS_COLLECTION,
                model_collection=settings.HYBRID_MODEL_COLLECTION,
                caption_collection=settings.HYBRID_CAPTION_COLLECTION,
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

        return {"status": "indexed", "model_id": model_id}

    def preview_index_asset(self, *, image_b64_list: list[str]) -> Dict[str, Any]:
        image_paths: list[Path] = []
        try:
            image_paths = self._write_temp_images_from_b64_list(image_b64_list)
            draft = self._suggest_metadata_from_images(image_paths)
            return {"status": "preview_ready", "draft": draft}
        finally:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)

    def confirm_index_asset(self, *, image_b64_list: list[str], metadata: Dict[str, Any]) -> Dict[str, str]:
        image_paths: list[Path] = []
        try:
            image_paths = self._write_temp_images_from_b64_list(image_b64_list)
            cleaned = self._normalize_confirm_metadata(metadata)
            model_id = cleaned.get("model_id", "").strip()
            if not model_id:
                model_id = self.orchestrator.allocate_model_id(category=cleaned.get("category") or None)
                cleaned["model_id"] = model_id
            for image_path in image_paths:
                per_image_metadata = dict(cleaned)
                per_image_metadata.pop("pk", None)
                self.orchestrator.preprocess_and_index(image_path, per_image_metadata)
            return {"status": "indexed", "model_id": model_id}
        finally:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)

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

    @staticmethod
    def _write_temp_image_from_b64(image_b64: str) -> Path:
        image_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            return Path(tmp.name)

    @classmethod
    def _write_temp_images_from_b64_list(cls, image_b64_list: list[str]) -> list[Path]:
        if not image_b64_list:
            raise ValueError("at least one image is required.")
        return [cls._write_temp_image_from_b64(image_b64) for image_b64 in image_b64_list]

    @staticmethod
    def _normalize_confirm_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        product_info = str(metadata.get("product_info") or "").strip()
        description = str(metadata.get("description") or "").strip()
        if product_info and not description:
            description = product_info
        cleaned: Dict[str, str] = {
            "model_id": str(metadata.get("model_id") or "").strip(),
            "maker": str(metadata.get("maker") or "").strip(),
            "part_number": str(metadata.get("part_number") or "").strip(),
            "category": str(metadata.get("category") or "").strip(),
            "description": description,
        }
        price_value = metadata.get("price_value")
        if price_value not in (None, ""):
            cleaned["price_text"] = str(price_value)
        if product_info:
            cleaned["product_info"] = product_info
        return cleaned

    def _suggest_metadata_from_images(self, image_paths: list[Path]) -> Dict[str, Any]:
        client = self._build_openai_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured for metadata preview.")

        selected_paths = image_paths[:4]
        payloads = [self._encode_image_for_openai(image_path) for image_path in selected_paths]
        prompt = (
            "You are generating metadata for one industrial component shown in one or more photos that will later be indexed.\n"
            "Return exactly one JSON object with keys: maker, part_number, category, description, product_info, price_value.\n"
            "Rules:\n"
            "- Use empty string for unknown text fields.\n"
            "- category should be a short stable label suitable for indexing.\n"
            "- description should be concise but retrieval-friendly.\n"
            "- part_number should prefer visible text on labels.\n"
            "- product_info should be a short product type.\n"
            "- price_value should be an integer KRW estimate if you can infer it, otherwise null.\n"
            "- Treat all images as the same product from different angles or distances.\n"
            "- Prefer details that are consistent across multiple photos.\n"
            "- Do not include markdown."
        )
        content = [{"type": "input_text", "text": prompt}]
        for payload in payloads:
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{payload}", "detail": "low"})
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        raw = getattr(response, "output_text", "") or ""
        parsed = self._extract_json_object(str(raw))
        if not parsed:
            raise RuntimeError("OpenAI metadata preview returned an unreadable response.")

        price_value = parsed.get("price_value")
        if isinstance(price_value, str):
            digits = "".join(ch for ch in price_value if ch.isdigit())
            price_value = int(digits) if digits else None
        elif not isinstance(price_value, int):
            price_value = None

        return {
            "model_id": "",
            "maker": str(parsed.get("maker") or "").strip(),
            "part_number": str(parsed.get("part_number") or "").strip(),
            "category": str(parsed.get("category") or "").strip(),
            "description": str(parsed.get("description") or "").strip(),
            "product_info": str(parsed.get("product_info") or "").strip(),
            "price_value": price_value,
            "source": "openai",
        }

    @staticmethod
    def _build_openai_client():
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package is required for metadata preview.") from exc
        return OpenAI()

    @staticmethod
    def _encode_image_for_openai(image_path: Path, *, max_side: int = 1024) -> str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)
            if longest > max_side:
                scale = max_side / float(longest)
                img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any] | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


hybrid_service = HybridSearchService()
