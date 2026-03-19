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
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from fastapi import UploadFile
from PIL import Image
from pymilvus import Collection, connections, utility
from smart_match.hybrid_search_pipeline.search.ranking_utils import (
    compute_exact_field_boost,
    compute_lexical_score,
    passes_min_score,
)
from smart_match.hybrid_search_pipeline.preprocessing.metadata_normalizer import MetadataNormalizer

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


@dataclass
class _IndexTask:
    task_id: str
    status: str
    model_id: str
    detail: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


class _TextOnlySearchRuntime:
    """Lightweight text-only retrieval path for chat and search requests."""

    def __init__(self) -> None:
        self._text_encoder = None
        self._model_collection: Collection | None = None
        self._attrs_collection: Collection | None = None
        self._collection_lock = Lock()
        self._result_cache = _TTLCache(ttl_seconds=60 * 10, max_items=512)

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            from smart_match.hybrid_search_pipeline.preprocessing.embedding.bge_m3_encoder import BGEM3TextEncoder

            logger.info("Initializing lightweight text encoder for text-only search...")
            self._text_encoder = BGEM3TextEncoder()
        return self._text_encoder

    @property
    def model_collection(self) -> Collection:
        with self._collection_lock:
            if self._model_collection is None:
                logger.info("Connecting lightweight text-only search to Milvus model collection...")
                connections.connect(alias="default", uri=settings.MILVUS_URI)
                if not utility.has_collection(settings.HYBRID_MODEL_COLLECTION):
                    raise RuntimeError(f"Milvus model collection '{settings.HYBRID_MODEL_COLLECTION}' is not initialized yet.")
                collection = Collection(name=settings.HYBRID_MODEL_COLLECTION)
                collection.load()
                self._model_collection = collection
            return self._model_collection

    @property
    def attrs_collection(self) -> Collection | None:
        with self._collection_lock:
            if self._attrs_collection is None:
                connections.connect(alias="default", uri=settings.MILVUS_URI)
                if not utility.has_collection(settings.HYBRID_ATTRS_COLLECTION):
                    return None
                collection = Collection(name=settings.HYBRID_ATTRS_COLLECTION)
                collection.load()
                self._attrs_collection = collection
            return self._attrs_collection

    def _load_images_for_models(self, model_ids: list[str]) -> dict[str, list[Dict[str, Any]]]:
        collection = self.attrs_collection
        if collection is None or not model_ids:
            return {}
        safe_ids = [model_id.replace("\\", "\\\\").replace('"', '\\"') for model_id in model_ids if model_id]
        if not safe_ids:
            return {}
        quoted_ids = ",".join([f'"{model_id}"' for model_id in safe_ids])
        expr = f"model_id in [{quoted_ids}]"
        rows = collection.query(expr=expr, output_fields=["pk", "model_id", "image_path", "maker", "part_number", "category", "caption_text"])
        grouped: dict[str, list[Dict[str, Any]]] = {}
        for row in rows:
            model_id = str(row.get("model_id") or "")
            if not model_id:
                continue
            grouped.setdefault(model_id, []).append(
                {
                    "image_id": str(row.get("pk") or ""),
                    "image_path": str(row.get("image_path") or ""),
                    "maker": str(row.get("maker") or ""),
                    "part_number": str(row.get("part_number") or ""),
                    "category": str(row.get("category") or ""),
                    "caption_text": str(row.get("caption_text") or ""),
                }
            )
        for images in grouped.values():
            images.sort(key=lambda item: item.get("image_id", ""))
        return grouped

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        # Milvus COSINE search returns a similarity-like score where larger is better.
        # Our model/text collections all use COSINE, so inverting with (1 - score)
        # incorrectly makes identical matches score near zero.
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_part_number(value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"[^0-9A-Za-z]", "", str(value).upper())

    @staticmethod
    def _normalize_category_key(value: str | None) -> str:
        if not value:
            return ""
        normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().upper())
        return normalized.strip("_")

    @staticmethod
    def _normalize_text_match(value: str | None) -> str:
        if not value:
            return ""
        normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _query_exact_category_matches(self, normalized_category_query: str, *, limit: int) -> dict[str, Dict[str, Any]]:
        if not normalized_category_query:
            return {}
        safe = normalized_category_query.replace("\\", "\\\\").replace('"', '\\"')
        rows = self.model_collection.query(
            expr=f'category == "{safe}"',
            output_fields=[
                "pk",
                "metadata_text",
                "ocr_text",
                "caption_text",
                "maker",
                "part_number",
                "category",
                "description",
            ],
            limit=limit,
        )
        out: dict[str, Dict[str, Any]] = {}
        for row in rows:
            model_id = str(row.get("pk") or "")
            if model_id:
                out[model_id] = row
        return out

    def search(self, *, query_text: str, top_k: int, part_number: str | None = None) -> list[Dict[str, Any]]:
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return []
        normalized_part_number = self._normalize_part_number(part_number)
        normalized_category_query = self._normalize_category_key(cleaned_query)
        normalized_query_text = self._normalize_text_match(cleaned_query)
        cache_key = hashlib.sha1(f"{cleaned_query}|{top_k}|{normalized_part_number}".encode("utf-8")).hexdigest()
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached

        t_start = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        t_embed = time.perf_counter()
        query_vector = self.text_encoder.encode_query(cleaned_query).tolist()
        timings_ms["text_embedding"] = round((time.perf_counter() - t_embed) * 1000, 2)

        search_k = max(top_k, min(100, top_k * 3))
        t_search = time.perf_counter()
        hits = self.model_collection.search(
            data=[list(query_vector)],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=search_k,
            output_fields=[
                "metadata_text",
                "ocr_text",
                "caption_text",
                "maker",
                "part_number",
                "category",
                "description",
            ],
        )[0]
        timings_ms["model_search"] = round((time.perf_counter() - t_search) * 1000, 2)

        exact_category_rows = self._query_exact_category_matches(
            normalized_category_query,
            limit=max(top_k, min(200, top_k * 10)),
        )

        candidate_model_ids = {str(getattr(hit, "id", "") or "") for hit in hits if str(getattr(hit, "id", "") or "")}
        candidate_model_ids.update(exact_category_rows.keys())
        image_map = self._load_images_for_models(sorted(candidate_model_ids))

        results: list[Dict[str, Any]] = []
        t_finalize = time.perf_counter()
        query_lower = cleaned_query.lower()
        seen_model_ids: set[str] = set()
        query_part_number = self._normalize_part_number(cleaned_query)
        for hit in hits:
            entity = hit.entity if hasattr(hit, "entity") and hit.entity is not None else {}
            getter = entity.get if hasattr(entity, "get") else lambda key, default="": default
            model_id = str(getattr(hit, "id", "") or getter("pk", ""))
            metadata_text = str(getter("metadata_text", "") or "")
            ocr_text = str(getter("ocr_text", "") or "")
            caption_text = str(getter("caption_text", "") or "")
            maker = str(getter("maker", "") or "")
            part_number = str(getter("part_number", "") or "")
            normalized_hit_part_number = self._normalize_part_number(part_number)
            category = str(getter("category", "") or "")
            normalized_hit_category = self._normalize_category_key(category)
            description = str(getter("description", "") or "")
            aggregated_text = " ".join(part for part in [metadata_text, ocr_text, caption_text] if part).strip()

            similarity = self._distance_to_similarity(getattr(hit, "distance", 1.0))
            haystacks = [metadata_text, ocr_text, caption_text, aggregated_text, maker, part_number, description]
            lexical_hit = any(query_lower in (haystack or "").lower() for haystack in haystacks)
            normalized_haystacks = [
                self._normalize_text_match(haystack)
                for haystack in [metadata_text, ocr_text, caption_text, aggregated_text, maker, part_number, description, category]
            ]
            if normalized_query_text and any(normalized_query_text and normalized_query_text in haystack for haystack in normalized_haystacks):
                lexical_hit = True
            if query_part_number and normalized_hit_part_number and query_part_number == normalized_hit_part_number:
                lexical_hit = True
            exact_category_match = bool(
                normalized_category_query
                and normalized_hit_category
                and normalized_category_query == normalized_hit_category
            )
            if exact_category_match:
                lexical_hit = True
            lexical_score = compute_lexical_score(cleaned_query, haystacks)
            exact_field_boost = compute_exact_field_boost(
                cleaned_query,
                model_id=model_id,
                maker=maker,
                part_number=part_number,
                description=description,
            )
            if query_part_number and normalized_hit_part_number and query_part_number == normalized_hit_part_number:
                exact_field_boost = max(exact_field_boost, 1.0)
            if exact_category_match:
                exact_field_boost = max(exact_field_boost, 0.9)
            final_score = min(1.0, similarity * 0.65 + lexical_score * 0.20 + exact_field_boost * 0.15)
            if lexical_hit:
                final_score = min(1.0, final_score + 0.08)
            if exact_field_boost >= 0.55:
                final_score = min(1.0, final_score + 0.12)
            elif exact_field_boost > 0.0:
                final_score = min(1.0, final_score + 0.06)
            if not passes_min_score(score=final_score, lexical_hit=lexical_hit, exact_field_boost=exact_field_boost):
                continue

            results.append(
                {
                    "model_id": model_id,
                    "score": final_score,
                    "image_score": 0.0,
                    "ocr_score": similarity,
                    "caption_score": 0.0,
                    "text_query_score": similarity,
                    "maker": maker,
                    "part_number": part_number,
                    "category": category,
                    "description": description,
                    "metadata_text": metadata_text,
                    "ocr_text": ocr_text,
                    "caption_text": caption_text,
                    "aggregated_text": aggregated_text,
                    "images": image_map.get(model_id, []),
                    "lexical_hit": lexical_hit,
                    "lexical_score": lexical_score,
                    "spec_match_score": 0.0,
                    "exact_field_boost": exact_field_boost,
                    "verified": False,
                }
            )
            seen_model_ids.add(model_id)

        for model_id, row in exact_category_rows.items():
            if model_id in seen_model_ids:
                continue
            metadata_text = str(row.get("metadata_text") or "")
            ocr_text = str(row.get("ocr_text") or "")
            caption_text = str(row.get("caption_text") or "")
            maker = str(row.get("maker") or "")
            part_number = str(row.get("part_number") or "")
            category = str(row.get("category") or "")
            description = str(row.get("description") or "")
            final_score = 0.92
            results.append(
                {
                    "model_id": model_id,
                    "score": final_score,
                    "image_score": 0.0,
                    "ocr_score": 0.0,
                    "caption_score": 0.0,
                    "text_query_score": 0.0,
                    "maker": maker,
                    "part_number": part_number,
                    "category": category,
                    "description": description,
                    "metadata_text": metadata_text,
                    "ocr_text": ocr_text,
                    "caption_text": caption_text,
                    "aggregated_text": " ".join(part for part in [metadata_text, ocr_text, caption_text] if part).strip(),
                    "images": image_map.get(model_id, []),
                    "lexical_hit": True,
                    "lexical_score": 1.0,
                    "spec_match_score": 0.0,
                    "exact_field_boost": 0.9,
                    "verified": False,
                }
            )

        if normalized_part_number:
            filtered = [
                item
                for item in results
                if self._normalize_part_number(item.get("part_number")) == normalized_part_number
            ]
            if filtered:
                results = filtered

        for item in results:
            item["verified"] = bool(
                normalized_part_number
                and self._normalize_part_number(item.get("part_number")) == normalized_part_number
            )

        results.sort(
            key=lambda item: (
                item.get("exact_field_boost", 0.0),
                item.get("lexical_hit", False),
                item.get("score", 0.0),
            ),
            reverse=True,
        )
        timings_ms["finalize"] = round((time.perf_counter() - t_finalize) * 1000, 2)
        timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info("Lightweight text-only search completed: results=%d timings_ms=%s", len(results[:top_k]), timings_ms)
        trimmed = results[:top_k]
        self._result_cache.set(cache_key, trimmed)
        return trimmed


class _MetadataPreviewRuntime:
    """Lazy local metadata preview runtime backed by PaddleOCR-VL."""

    def __init__(self) -> None:
        self._ocr_engine = None
        self._lock = Lock()

    @property
    def ocr_engine(self):
        with self._lock:
            if self._ocr_engine is None:
                from smart_match.hybrid_search_pipeline.preprocessing.ocr.OCR import PaddleOCRVLPipeline

                logger.info("Initializing local metadata preview OCR runtime (PaddleOCR-VL)...")
                self._ocr_engine = PaddleOCRVLPipeline()
            return self._ocr_engine


class _QwenMetadataPreviewRuntime:
    """Lazy local metadata preview runtime backed by Qwen3-VL."""

    def __init__(self) -> None:
        self._captioner = None
        self._lock = Lock()

    @property
    def captioner(self):
        with self._lock:
            if self._captioner is None:
                from smart_match.hybrid_search_pipeline.preprocessing.captioning.qwen3_captioner import Qwen3VLCaptioner

                logger.info("Initializing local metadata preview Qwen runtime...")
                self._captioner = Qwen3VLCaptioner(max_new_tokens=220, do_sample=False)
            return self._captioner


def _resolve_metadata_preview_backend(override: str | None = None) -> str:
    backend = (override or settings.METADATA_PREVIEW_BACKEND or "auto").strip().lower()
    if backend and backend != "auto":
        if backend in {"local", "qwen"}:
            return "qwen"
        if backend in {"gpt", "openai"}:
            return "openai"
        return backend
    if settings.LOCAL_MODE:
        return "qwen"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "qwen"


class HybridSearchService:
    """Lazy-initialized orchestrator wrapper for API usage."""

    def __init__(self) -> None:
        self._orchestrator: HybridSearchOrchestrator | None = None
        self._milvus_config = None
        self._fusion_weights = None
        self._query_cache = _TTLCache(ttl_seconds=60, max_items=256)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid-index")
        self._tasks: OrderedDict[str, _IndexTask] = OrderedDict()
        self._max_tasks = 256
        self._tasks_lock = Lock()
        self._text_only_runtime: _TextOnlySearchRuntime | None = None
        self._metadata_preview_runtime: _MetadataPreviewRuntime | None = None
        self._qwen_metadata_preview_runtime: _QwenMetadataPreviewRuntime | None = None
        self._metadata_normalizer = MetadataNormalizer()

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

    def preview_index_asset(
        self,
        *,
        image_b64_list: list[str],
        metadata_mode: str | None = None,
        label_image_b64_list: list[str] | None = None,
    ) -> Dict[str, Any]:
        image_paths: list[Path] = []
        label_image_paths: list[Path] = []
        try:
            image_paths = self._write_temp_images_from_b64_list(image_b64_list)
            label_image_paths = self._write_temp_images_from_b64_list(label_image_b64_list or []) if label_image_b64_list else []
            label_ocr_text = self._extract_label_ocr_text(label_image_paths)
            draft, ocr_image_indices = self._suggest_metadata_from_images(
                image_paths,
                backend_override=metadata_mode,
                label_ocr_text=label_ocr_text,
            )
            duplicate_candidate = self._find_duplicate_candidate(draft)
            return {
                "status": "preview_ready",
                "draft": draft,
                "ocr_image_indices": ocr_image_indices,
                "label_ocr_text": label_ocr_text,
                "duplicate_candidate": duplicate_candidate,
            }
        finally:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            for image_path in label_image_paths:
                image_path.unlink(missing_ok=True)

    def confirm_index_asset(self, *, image_b64_list: list[str], metadata: Dict[str, Any]) -> Dict[str, str]:
        cleaned = self._normalize_confirm_metadata(metadata)
        ocr_image_indices = self._normalize_ocr_image_indices(metadata.get("ocr_image_indices"), len(image_b64_list))
        model_id = cleaned.get("model_id", "").strip()

        task_id = uuid.uuid4().hex
        self._set_task(
            _IndexTask(
                task_id=task_id,
                status="queued",
                model_id=model_id,
                detail="Indexing job queued.",
                created_at=time.time(),
                updated_at=time.time(),
            )
        )
        self._executor.submit(self._run_confirm_index_job, task_id, list(image_b64_list), dict(cleaned), ocr_image_indices)
        return {"status": "queued", "model_id": model_id, "task_id": task_id}

    def get_index_task(self, task_id: str) -> Dict[str, Any]:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            self._tasks.move_to_end(task_id)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "model_id": task.model_id or None,
            "detail": task.detail,
        }

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
        use_reranker: Optional[bool] = None,
    ):
        """Execute hybrid search optionally using both modalities."""
        image_key = hashlib.sha1((image_b64 or "").encode("utf-8")).hexdigest()[:16] if image_b64 else ""
        cache_key = hashlib.sha1(
            f"{query_text or ''}|{image_key}|{top_k}|{part_number or ''}".encode("utf-8")
        ).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        if query_text and not image_b64:
            if self._text_only_runtime is None:
                self._text_only_runtime = _TextOnlySearchRuntime()
            results = self._text_only_runtime.search(query_text=query_text, top_k=top_k, part_number=part_number)
            self._query_cache.set(cache_key, results)
            return results

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
                use_reranker=use_reranker,
            )
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
        self._query_cache.set(cache_key, results)
        return results

    def warmup_text_only(self) -> None:
        if self._text_only_runtime is None:
            self._text_only_runtime = _TextOnlySearchRuntime()
        logger.info("Warming up lightweight text-only search runtime...")
        _ = self._text_only_runtime.text_encoder
        connections.connect(alias="default", uri=settings.MILVUS_URI)
        if utility.has_collection(settings.HYBRID_MODEL_COLLECTION):
            _ = self._text_only_runtime.model_collection
            logger.info("Lightweight text-only search runtime ready.")
            return
        logger.info(
            "Skipping lightweight text-only search collection warmup because '%s' does not exist yet.",
            settings.HYBRID_MODEL_COLLECTION,
        )

    def warmup_qwen_preview(self) -> None:
        if _resolve_metadata_preview_backend() != "qwen":
            logger.info("Skipping Qwen metadata preview warmup because resolved backend is not qwen.")
            return
        if self._qwen_metadata_preview_runtime is None:
            self._qwen_metadata_preview_runtime = _QwenMetadataPreviewRuntime()
        logger.info("Warming up Qwen metadata preview runtime...")
        _ = self._qwen_metadata_preview_runtime.captioner
        logger.info("Qwen metadata preview runtime ready.")

    def collection_stats(self) -> Dict[str, Dict[str, Any]]:
        info: Dict[str, Dict[str, Any]] = {}
        collections = {
            "image": settings.HYBRID_IMAGE_COLLECTION,
            "text": settings.HYBRID_TEXT_COLLECTION,
            "attrs": settings.HYBRID_ATTRS_COLLECTION,
            "model": settings.HYBRID_MODEL_COLLECTION,
            "caption": settings.HYBRID_CAPTION_COLLECTION,
        }
        if self._text_only_runtime is None:
            self._text_only_runtime = _TextOnlySearchRuntime()
        from pymilvus import Collection, utility

        connections.connect(alias="default", uri=settings.MILVUS_URI)
        for label, name in collections.items():
            if not utility.has_collection(name):
                info[label] = {"name": name, "num_entities": 0, "exists": False}
                continue
            collection = Collection(name=name)
            info[label] = {"name": name, "num_entities": collection.num_entities, "exists": True}
        return info

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

    @staticmethod
    def _normalize_ocr_image_indices(value: Any, image_count: int) -> list[int]:
        if not value:
            return []
        if not isinstance(value, (list, tuple, set)):
            return []
        cleaned: list[int] = []
        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= index < image_count and index not in cleaned:
                cleaned.append(index)
        return cleaned

    def _suggest_metadata_from_images(
        self,
        image_paths: list[Path],
        *,
        backend_override: str | None = None,
        label_ocr_text: str = "",
    ) -> tuple[Dict[str, Any], list[int]]:
        backend = _resolve_metadata_preview_backend(backend_override)
        if backend == "qwen":
            qwen_draft = self._suggest_metadata_from_qwen(image_paths, label_ocr_text=label_ocr_text)
            return qwen_draft, []
        if backend in {"gpt", "openai"}:
            openai_draft = self._suggest_metadata_from_openai(image_paths, label_ocr_text=label_ocr_text)
            return openai_draft, []
        local_start = time.perf_counter()
        local_draft, ocr_image_indices = self._suggest_metadata_from_ocr_vl(image_paths)
        logger.info(
            "Local metadata preview completed: source=%s ocr_images=%s duration=%.2fs",
            local_draft.get("source", "paddleocr_vl"),
            ocr_image_indices,
            time.perf_counter() - local_start,
        )

        if backend in {"paddleocr_vl", "paddle", "local"}:
            return local_draft, ocr_image_indices

        if backend in {"openai", "gpt"}:
            openai_draft = self._suggest_metadata_from_openai(image_paths, label_ocr_text=label_ocr_text)
            return openai_draft, ocr_image_indices

        if self._metadata_preview_has_minimum_signal(local_draft):
            return local_draft, ocr_image_indices

        client = self._build_openai_client()
        if client is None:
            return local_draft, ocr_image_indices

        openai_draft = self._suggest_metadata_from_openai(image_paths, label_ocr_text=label_ocr_text)
        merged = self._merge_metadata_preview_drafts(openai_draft, local_draft)
        merged["source"] = "openai+ocr_vl"
        return merged, ocr_image_indices

    def _suggest_metadata_from_qwen(self, image_paths: list[Path], *, label_ocr_text: str = "") -> Dict[str, Any]:
        if self._qwen_metadata_preview_runtime is None:
            self._qwen_metadata_preview_runtime = _QwenMetadataPreviewRuntime()

        selected_paths = image_paths[:4]
        if not selected_paths:
            return {
                "model_id": "",
                "maker": "",
                "part_number": "",
                "category": "",
                "description": "",
                "product_info": "",
                "price_value": None,
                "source": "qwen3_vl",
            }

        prompt = (
            "Analyze this industrial component image and return exactly one JSON object with keys "
            "maker, part_number, category, description, product_info, price_value. "
            "Use empty string for unknown text fields. Use null for unknown price_value. "
            "part_number should prefer visible label text. category should be a short stable indexing label. "
            "description should be concise and retrieval-friendly. Do not include markdown or extra commentary."
        )
        if label_ocr_text.strip():
            prompt += (
                "\nSupplemental label OCR text from separate close-up images is provided below. "
                "Treat it as higher-priority evidence for maker and part_number.\n"
                f"LABEL_OCR_TEXT:\n{label_ocr_text.strip()}"
            )
        start = time.perf_counter()
        per_image_results: list[Dict[str, Any]] = []
        raw_outputs: list[str] = []
        for image_path in selected_paths:
            raw = self._qwen_metadata_preview_runtime.captioner.generate(image_path, prompt=prompt)
            raw_text = str(raw or "").strip()
            raw_outputs.append(raw_text)
            parsed = self._extract_json_object(raw_text)
            if not parsed:
                parsed = {
                    "maker": self._infer_maker_from_text(raw_text),
                    "part_number": self._infer_part_number_from_text(raw_text),
                    "category": self._infer_category_from_text(raw_text),
                    "description": raw_text[:160],
                    "product_info": "",
                    "price_value": None,
                }
            per_image_results.append(parsed)
        logger.info(
            "Qwen metadata preview completed in %.2fs across %d image(s)",
            time.perf_counter() - start,
            len(selected_paths),
        )

        merged = self._merge_qwen_preview_results(per_image_results, raw_outputs)
        price_value = merged.get("price_value")
        if isinstance(price_value, str):
            digits = "".join(ch for ch in price_value if ch.isdigit())
            price_value = int(digits) if digits else None
        elif not isinstance(price_value, int):
            price_value = None

        return {
            "model_id": "",
            "maker": str(merged.get("maker") or "").strip(),
            "part_number": str(merged.get("part_number") or "").strip(),
            "category": str(merged.get("category") or "").strip(),
            "description": str(merged.get("description") or "").strip(),
            "product_info": str(merged.get("product_info") or "").strip(),
            "price_value": price_value,
            "source": "qwen3_vl",
        }

    @staticmethod
    def _merge_qwen_preview_results(per_image_results: list[Dict[str, Any]], raw_outputs: list[str]) -> Dict[str, Any]:
        def _best_text(key: str) -> str:
            counts: Dict[str, int] = {}
            for result in per_image_results:
                value = str(result.get(key) or "").strip()
                if not value:
                    continue
                counts[value] = counts.get(value, 0) + 1
            if not counts:
                return ""
            return sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)[0][0]

        description = _best_text("description")
        if not description:
            merged_raw = " ".join(text for text in raw_outputs if text).strip()
            description = merged_raw[:160]

        product_info = _best_text("product_info")
        category = _best_text("category")
        if category and not product_info:
            product_info = category.replace("_", " ")

        return {
            "maker": _best_text("maker"),
            "part_number": _best_text("part_number"),
            "category": category,
            "description": description,
            "product_info": product_info,
            "price_value": next((result.get("price_value") for result in per_image_results if result.get("price_value") not in (None, "")), None),
        }

    def _suggest_metadata_from_ocr_vl(self, image_paths: list[Path]) -> tuple[Dict[str, Any], list[int]]:
        if self._metadata_preview_runtime is None:
            self._metadata_preview_runtime = _MetadataPreviewRuntime()

        image_analyses: list[Dict[str, Any]] = []
        combined_corpus_parts: list[str] = []
        for index, image_path in enumerate(image_paths[:4]):
            ocr_result = self._metadata_preview_runtime.ocr_engine.extract(str(image_path))
            raw_text = " ".join(
                part.strip()
                for part in [
                    getattr(ocr_result, "structured_text", "") or "",
                    ocr_result.markdown_text or "",
                    ocr_result.combined_text or "",
                    getattr(ocr_result, "spotting_text", "") or "",
                ]
                if part and str(part).strip()
            ).strip()
            tokens = [token.text.strip() for token in ocr_result.tokens if getattr(token, "text", "").strip()]
            label_score = self._compute_label_signal(raw_text, tokens)
            image_analyses.append(
                {
                    "index": index,
                    "text": raw_text,
                    "tokens": tokens,
                    "label_score": label_score,
                }
            )
            if raw_text:
                combined_corpus_parts.append(raw_text)

        selected = [item for item in image_analyses if item["label_score"] >= 4]
        if not selected and image_analyses:
            strongest = max(image_analyses, key=lambda item: item["label_score"])
            if strongest["label_score"] > 0:
                selected = [strongest]
        ocr_image_indices = [int(item["index"]) for item in selected]

        selected_text = " ".join(item["text"] for item in selected if item["text"]).strip()
        full_text = " ".join(combined_corpus_parts).strip()
        working_text = selected_text or full_text
        maker = self._infer_maker_from_text(working_text)
        part_number = self._infer_part_number_from_text(working_text)
        category = self._infer_category_from_text(working_text)
        product_info = category.replace("_", " ") if category else ""
        description_parts = [part for part in [maker, product_info, part_number] if part]
        description = " ".join(description_parts).strip()
        if not description and working_text:
            description = working_text[:160].strip()

        draft = {
            "model_id": "",
            "maker": maker,
            "part_number": part_number,
            "category": category,
            "description": description,
            "product_info": product_info,
            "price_value": None,
            "source": "paddleocr_vl",
        }
        if not any([maker, part_number, category, description]) and full_text:
            draft["description"] = full_text[:160].strip()
        return draft, ocr_image_indices

    def _suggest_metadata_from_openai(self, image_paths: list[Path], *, label_ocr_text: str = "") -> Dict[str, Any]:
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
        if label_ocr_text.strip():
            prompt += (
                "\nSupplemental OCR text extracted from separate label close-ups is provided below. "
                "Use it as the strongest evidence for maker and part_number when it is coherent.\n"
                f"LABEL_OCR_TEXT:\n{label_ocr_text.strip()}"
            )
        content = [{"type": "input_text", "text": prompt}]
        for payload in payloads:
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{payload}", "detail": "low"})

        start = time.perf_counter()
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[{"role": "user", "content": content}],
        )
        raw = getattr(response, "output_text", "") or ""
        parsed = self._extract_json_object(str(raw))
        if not parsed:
            raise RuntimeError("OpenAI metadata preview returned an unreadable response.")
        logger.info("OpenAI metadata preview completed in %.2fs", time.perf_counter() - start)

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

    def _extract_label_ocr_text(self, image_paths: list[Path]) -> str:
        if not image_paths:
            return ""
        if self._metadata_preview_runtime is None:
            self._metadata_preview_runtime = _MetadataPreviewRuntime()
        texts: list[str] = []
        start = time.perf_counter()
        for image_path in image_paths[:4]:
            ocr_result = self._metadata_preview_runtime.ocr_engine.extract(str(image_path))
            text = " ".join(
                part.strip()
                for part in [
                    getattr(ocr_result, "structured_text", "") or "",
                    ocr_result.markdown_text or "",
                    ocr_result.combined_text or "",
                    getattr(ocr_result, "spotting_text", "") or "",
                ]
                if part and str(part).strip()
            ).strip()
            if text:
                texts.append(text)
        merged = "\n".join(texts).strip()
        logger.info("Label OCR completed: images=%d chars=%d duration=%.2fs", len(image_paths[:4]), len(merged), time.perf_counter() - start)
        return merged

    def _find_duplicate_candidate(self, draft: Dict[str, Any]) -> Dict[str, Any] | None:
        part_number_raw = str(draft.get("part_number") or "").strip()
        if not part_number_raw:
            return None

        normalized = self._metadata_normalizer.normalize(
            {
                "maker": str(draft.get("maker") or ""),
                "part_number": part_number_raw,
            }
        )
        normalized_part_number = str(normalized.get("part_number") or "").strip()
        normalized_maker = str(normalized.get("maker") or "").strip().lower()
        if not normalized_part_number:
            return None

        try:
            connections.connect(alias="default", uri=settings.MILVUS_URI)
            if not utility.has_collection(settings.HYBRID_MODEL_COLLECTION):
                return None
            model_collection = Collection(name=settings.HYBRID_MODEL_COLLECTION)
            safe_part_number = normalized_part_number.replace("\\", "\\\\").replace('"', '\\"')
            rows = model_collection.query(
                expr=f'part_number == "{safe_part_number}"',
                output_fields=["pk", "maker", "part_number", "category", "description"],
            )
            if not rows:
                return None

            matched_row = None
            if normalized_maker:
                for row in rows:
                    row_maker = str(row.get("maker") or "").strip().lower()
                    if row_maker == normalized_maker:
                        matched_row = row
                        break
            if matched_row is None and len(rows) == 1:
                matched_row = rows[0]
            if matched_row is None:
                return None

            image_path = ""
            if utility.has_collection(settings.HYBRID_ATTRS_COLLECTION):
                attrs_collection = Collection(name=settings.HYBRID_ATTRS_COLLECTION)
                safe_model_id = str(matched_row.get("pk") or "").replace("\\", "\\\\").replace('"', '\\"')
                attr_rows = attrs_collection.query(
                    expr=f'model_id == "{safe_model_id}"',
                    output_fields=["pk", "image_path"],
                )
                if attr_rows:
                    attr_rows.sort(key=lambda item: str(item.get("pk") or ""))
                    image_path = str(attr_rows[0].get("image_path") or "")

            return {
                "model_id": str(matched_row.get("pk") or ""),
                "maker": str(matched_row.get("maker") or ""),
                "part_number": str(matched_row.get("part_number") or ""),
                "category": str(matched_row.get("category") or ""),
                "description": str(matched_row.get("description") or ""),
                "image_path": image_path,
                "reason": "Existing indexed part with the same part number was found.",
            }
        except Exception:
            logger.exception("Failed to look up duplicate candidate for metadata preview.")
            return None

    @staticmethod
    def _metadata_preview_has_minimum_signal(draft: Dict[str, Any]) -> bool:
        return bool(str(draft.get("maker") or "").strip() or str(draft.get("part_number") or "").strip() or str(draft.get("category") or "").strip())

    @staticmethod
    def _merge_metadata_preview_drafts(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(primary)
        for key in ("maker", "part_number", "category", "description", "product_info", "price_value"):
            value = merged.get(key)
            if value in (None, ""):
                merged[key] = fallback.get(key)
        return merged

    @staticmethod
    def _compute_label_signal(text: str, tokens: list[str]) -> int:
        corpus = " ".join([text, *tokens]).strip()
        if not corpus:
            return 0
        score = 0
        if HybridSearchService._infer_part_number_from_text(corpus):
            score += 5
        if HybridSearchService._infer_maker_from_text(corpus):
            score += 2
        if re.search(r"\b(?:model|type|part|pn|p\/n|mfg)\b", corpus, flags=re.IGNORECASE):
            score += 2
        if len(tokens) >= 3:
            score += 1
        return score

    @staticmethod
    def _infer_maker_from_text(text: str) -> str:
        corpus = (text or "").lower()
        makers = {
            "fuji electric": "Fuji Electric",
            "omron": "Omron",
            "mitsubishi": "Mitsubishi",
            "schneider": "Schneider Electric",
            "schneider electric": "Schneider Electric",
            "siemens": "Siemens",
            "ls electric": "LS Electric",
            "rockwell": "Rockwell Automation",
            "allen bradley": "Allen-Bradley",
            "smc": "SMC",
            "yaskawa": "Yaskawa",
            "keyence": "Keyence",
            "autonics": "Autonics",
            "fanuc": "Fanuc",
            "panasonic": "Panasonic",
        }
        for needle, maker in makers.items():
            if needle in corpus:
                return maker
        return ""

    @staticmethod
    def _infer_part_number_from_text(text: str) -> str:
        corpus = (text or "").upper()
        candidates = re.findall(r"\b[A-Z0-9][A-Z0-9./_-]{3,}\b", corpus)
        filtered: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip("._-/")
            if len(cleaned) < 4:
                continue
            if cleaned.isdigit():
                continue
            if not re.search(r"[A-Z]", cleaned) or not re.search(r"\d", cleaned):
                continue
            filtered.append(cleaned)
        filtered.sort(key=lambda item: (len(item), item.count("-"), item.count("/")), reverse=True)
        return filtered[0] if filtered else ""

    @staticmethod
    def _infer_category_from_text(text: str) -> str:
        corpus = (text or "").lower()
        keywords = {
            "contactor": ("contactor", "magnetic contactor", "mc-"),
            "relay": ("relay", "solid state relay", "ssr"),
            "breaker": ("breaker", "circuit breaker", "mccb", "elcb"),
            "servo_drive": ("servo drive", "servo amplifier", "servopack"),
            "plc": ("plc", "programmable controller"),
            "sensor": ("sensor", "photoelectric", "prox", "proximity"),
            "inverter": ("inverter", "vfd", "ac drive"),
            "power_supply": ("power supply", "smps", "psu"),
            "valve": ("valve", "solenoid valve"),
        }
        for category, hints in keywords.items():
            if any(hint in corpus for hint in hints):
                return category
        return ""

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

    def _run_confirm_index_job(
        self,
        task_id: str,
        image_b64_list: list[str],
        metadata: Dict[str, str],
        ocr_image_indices: list[int],
    ) -> None:
        total_images = max(1, len(image_b64_list))
        self._update_task(task_id, status="running", detail="Preparing images for indexing.")
        image_paths: list[Path] = []
        try:
            model_id = str(metadata.get("model_id") or "").strip()
            if not model_id:
                self._update_task(task_id, status="running", detail="Allocating model ID.")
                model_id = self.orchestrator.allocate_model_id(category=metadata.get("category") or None)
                metadata["model_id"] = model_id
                self._update_task(task_id, status="running", detail="Model ID allocated.", model_id=model_id)
            self._update_task(task_id, status="running", detail="Decoding uploaded images.", model_id=model_id)
            image_paths = self._write_temp_images_from_b64_list(image_b64_list)
            self._update_task(task_id, status="running", detail="Initializing indexing runtime.", model_id=model_id)
            for index, image_path in enumerate(image_paths, start=1):
                per_image_metadata = dict(metadata)
                per_image_metadata.pop("pk", None)
                enable_ocr = not ocr_image_indices or (index - 1) in ocr_image_indices
                self._update_task(
                    task_id,
                    status="running",
                    detail=f"Indexing image {index}/{total_images}: {'OCR + ' if enable_ocr else ''}embeddings and Milvus upsert.",
                    model_id=model_id,
                )
                self.orchestrator.preprocess_and_index(image_path, per_image_metadata, enable_ocr=enable_ocr)
            self._update_task(
                task_id,
                status="completed",
                detail="Indexing completed successfully.",
                model_id=model_id,
            )
        except Exception as exc:
            logger.exception("Async confirm indexing failed: task_id=%s error=%s", task_id, exc)
            self._update_task(task_id, status="failed", detail=str(exc))
        finally:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)

    def _set_task(self, task: _IndexTask) -> None:
        with self._tasks_lock:
            self._tasks[task.task_id] = task
            self._tasks.move_to_end(task.task_id)
            while len(self._tasks) > self._max_tasks:
                self._tasks.popitem(last=False)

    def _update_task(self, task_id: str, *, status: str, detail: str, model_id: str | None = None) -> None:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = status
            task.detail = detail
            if model_id is not None:
                task.model_id = model_id
            task.updated_at = time.time()
            self._tasks.move_to_end(task_id)


hybrid_service = HybridSearchService()
