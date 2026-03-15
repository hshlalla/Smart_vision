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

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - value)

    def search(self, *, query_text: str, top_k: int) -> list[Dict[str, Any]]:
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return []
        cache_key = hashlib.sha1(f"{cleaned_query}|{top_k}".encode("utf-8")).hexdigest()
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

        results: list[Dict[str, Any]] = []
        t_finalize = time.perf_counter()
        query_lower = cleaned_query.lower()
        for hit in hits:
            entity = hit.entity if hasattr(hit, "entity") and hit.entity is not None else {}
            getter = entity.get if hasattr(entity, "get") else lambda key, default="": default
            model_id = str(getattr(hit, "id", "") or getter("pk", ""))
            metadata_text = str(getter("metadata_text", "") or "")
            ocr_text = str(getter("ocr_text", "") or "")
            caption_text = str(getter("caption_text", "") or "")
            maker = str(getter("maker", "") or "")
            part_number = str(getter("part_number", "") or "")
            category = str(getter("category", "") or "")
            description = str(getter("description", "") or "")
            aggregated_text = " ".join(part for part in [metadata_text, ocr_text, caption_text] if part).strip()

            similarity = self._distance_to_similarity(getattr(hit, "distance", 1.0))
            haystacks = [metadata_text, ocr_text, caption_text, aggregated_text, maker, part_number, description]
            lexical_hit = any(query_lower in (haystack or "").lower() for haystack in haystacks)
            lexical_score = compute_lexical_score(cleaned_query, haystacks)
            exact_field_boost = compute_exact_field_boost(
                cleaned_query,
                model_id=model_id,
                maker=maker,
                part_number=part_number,
                description=description,
            )
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
                    "images": [],
                    "lexical_hit": lexical_hit,
                    "lexical_score": lexical_score,
                    "spec_match_score": 0.0,
                    "exact_field_boost": exact_field_boost,
                    "verified": False,
                }
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
        cleaned = self._normalize_confirm_metadata(metadata)
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
        self._executor.submit(self._run_confirm_index_job, task_id, list(image_b64_list), dict(cleaned))
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
            results = self._text_only_runtime.search(query_text=query_text, top_k=top_k)
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
        if utility.has_collection(settings.HYBRID_MODEL_COLLECTION):
            _ = self._text_only_runtime.model_collection
            logger.info("Lightweight text-only search runtime ready.")
            return
        logger.info(
            "Skipping lightweight text-only search collection warmup because '%s' does not exist yet.",
            settings.HYBRID_MODEL_COLLECTION,
        )

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

    def _run_confirm_index_job(self, task_id: str, image_b64_list: list[str], metadata: Dict[str, str]) -> None:
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
                self._update_task(
                    task_id,
                    status="running",
                    detail=f"Indexing image {index}/{total_images}: OCR, embeddings, and Milvus upsert.",
                    model_id=model_id,
                )
                self.orchestrator.preprocess_and_index(image_path, per_image_metadata)
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
