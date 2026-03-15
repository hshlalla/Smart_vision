"""
End-to-End Hybrid Search Pipeline Runner

Provides a reference implementation that wires together:
    - PaddleOCR-VL for OCR extraction
    - BGE-VL for image embeddings
    - BGE-M3 for text embeddings
    - Milvus hybrid index for dense + attribute storage
    - Fusion retriever for scoring and reranking
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from pymilvus import DataType, connections
from smart_match.device_utils import is_apple_mps_device, preferred_torch_device

from .data_collection.tracker_dataset import TrackerDataset
from .preprocessing.captioning.gpt_captioner import GPTVLCaptioner
from .preprocessing.captioning.qwen3_captioner import Qwen3VLCaptioner
from .preprocessing.embedding.bge_m3_encoder import BGEM3TextEncoder
from .preprocessing.embedding.qwen3_vl_embedding import Qwen3VLImageEncoder
from .preprocessing.metadata_normalizer import MetadataNormalizer
from .preprocessing.ocr.OCR import PaddleOCRVLPipeline
from .preprocessing.pipeline import PreprocessingPipeline
from .retrieval.milvus_hybrid_index import CollectionConfig, FieldSpec, HybridMilvusIndex
from .retrieval.milvus_counters import MilvusCounterStore
from .search.fusion_retriever import FusionWeights, HybridFusionRetriever
from .search.qwen3_vl_reranker import Qwen3VLReranker
from .search.ranking_utils import (
    MIN_RESULT_SCORE,
    compute_exact_field_boost,
    compute_lexical_score,
    passes_min_score,
    tokenize_text,
)

logger = logging.getLogger(__name__)

# Enable verbose logging when this module is imported/run directly.
# If the root logger already has handlers (e.g., caller configured logging),
# this will be a no-op.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class MilvusConnectionConfig:
    uri: str = "tcp://standalone:19530"
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    db_name: Optional[str] = None


class HybridSearchOrchestrator:
    """Top-level orchestrator that manages preprocessing, indexing, and search."""

    MIN_RESULT_SCORE = MIN_RESULT_SCORE

    def __init__(
        self,
        *,
        milvus: MilvusConnectionConfig = MilvusConnectionConfig(),
        image_collection: str = "qwen3_vl_image_parts",
        text_collection: str = "bge_m3_text_parts",
        attrs_collection: str = "attrs_parts_v2",
        model_collection: str = "bge_m3_model_texts",
        caption_collection: str = "bge_m3_caption_parts",
        fusion_weights: FusionWeights = FusionWeights(alpha=0.5, beta=0.3, gamma=0.2),
        tracker_dataset_path: str | Path | None = None,
    ) -> None:
        self._connect_milvus(milvus)
        self.vision_encoder = Qwen3VLImageEncoder()
        self.text_encoder = BGEM3TextEncoder()
        enable_ocr_default = _env_flag("ENABLE_OCR", True)
        self.enable_ocr_indexing = _env_flag("ENABLE_OCR_INDEXING", enable_ocr_default)
        self.enable_ocr_query = _env_flag("ENABLE_OCR_QUERY", enable_ocr_default)
        if self.enable_ocr_indexing or self.enable_ocr_query:
            self.ocr_engine = PaddleOCRVLPipeline()
        else:
            logger.info("OCR disabled for both indexing and query paths via environment flags.")
            self.ocr_engine = None
        self.captioner = self._init_captioner()
        self.reranker = self._init_reranker()
        self.metadata_normalizer = MetadataNormalizer()
        self.preprocessing = PreprocessingPipeline(
            vision_encoder=self.vision_encoder,
            ocr_engine=self.ocr_engine,
            text_encoder=self.text_encoder,
            metadata_normalizer=self.metadata_normalizer,
            captioner=self.captioner,
        )
        self._image_pk_counters: Dict[str, int] = {}
        self._tracker_dataset_path = Path(tracker_dataset_path).expanduser().resolve() if tracker_dataset_path else None
        self._tracker_dataset: TrackerDataset | None = None

        counters_namespace = os.getenv("COUNTERS_COLLECTION", "sv_counters")
        self.counters = MilvusCounterStore(collection_name=counters_namespace)

        media_root_env = os.getenv("MEDIA_ROOT", "media")
        self.media_root = Path(media_root_env).expanduser().resolve()
        self.media_root.mkdir(parents=True, exist_ok=True)

        model_extra_fields = [
            FieldSpec("metadata_text", DataType.VARCHAR, max_length=8192),
            FieldSpec("ocr_text", DataType.VARCHAR, max_length=16384),
            FieldSpec("caption_text", DataType.VARCHAR, max_length=8192),
            FieldSpec("maker", DataType.VARCHAR, max_length=512),
            FieldSpec("part_number", DataType.VARCHAR, max_length=512),
            FieldSpec("category", DataType.VARCHAR, max_length=512),
            FieldSpec("description", DataType.VARCHAR, max_length=2048),
        ]

        self.index = HybridMilvusIndex(
            image_cfg=CollectionConfig(name=image_collection, dimension=self.vision_encoder.embedding_dim),
            text_cfg=CollectionConfig(name=text_collection, dimension=self.text_encoder.embedding_dim),
            caption_cfg=CollectionConfig(name=caption_collection, dimension=self.text_encoder.embedding_dim),
            attrs_fields=[
                FieldSpec("model_id", DataType.VARCHAR, max_length=256),
                FieldSpec("maker", DataType.VARCHAR, max_length=256),
                FieldSpec("part_number", DataType.VARCHAR, max_length=256),
                FieldSpec("category", DataType.VARCHAR, max_length=256),
                FieldSpec("ocr_text", DataType.VARCHAR, max_length=2048),
                FieldSpec("caption_text", DataType.VARCHAR, max_length=2048),
                FieldSpec("image_path", DataType.VARCHAR, max_length=512),
            ],
            model_cfg=CollectionConfig(
                name=model_collection,
                dimension=self.text_encoder.embedding_dim,
                extra_fields=model_extra_fields,
            ),
            image_collection_name=image_collection,
            text_collection_name=text_collection,
            attrs_collection_name=attrs_collection,
            model_collection_name=model_collection,
            caption_collection_name=caption_collection,
        )
        self.index.create_indexes()
        self.index.load()

        self.retriever = HybridFusionRetriever(
            cross_encoder=self.reranker or self._noop_cross_encoder,
            weights=fusion_weights,
        )
        logger.info(
            "HybridSearchOrchestrator devices: vision=%s text=%s captioner=%s reranker=%s ocr_enabled(index=%s query=%s)",
            self._component_device(self.vision_encoder),
            self._component_device(self.text_encoder),
            self._component_device(self.captioner),
            self._component_device(self.reranker),
            self.enable_ocr_indexing,
            self.enable_ocr_query,
        )

    @staticmethod
    def _component_device(component: Any) -> str:
        if component is None:
            return "disabled"
        for attr in ("_device", "device"):
            value = getattr(component, attr, None)
            if value:
                return str(value)
        return component.__class__.__name__

    def _init_captioner(self):
        prompt = (
            "Describe this industrial component in detail, including its visible shape, color, "
            "material, labels, and any text on it."
        )

        backend = (os.getenv("CAPTIONER_BACKEND") or "").strip().lower()
        local_mode_env = (os.getenv("LOCAL_MODE") or "").strip().lower()
        local_mode = local_mode_env in {"1", "true", "yes", "y", "on"}
        enable_caption_env = os.getenv("ENABLE_CAPTIONER")
        if enable_caption_env is None:
            enable_caption = True
        else:
            enable_caption = enable_caption_env.strip().lower() in {"1", "true", "yes", "y", "on"}

        if not enable_caption:
            logger.info("Captioner disabled via ENABLE_CAPTIONER=%s.", enable_caption_env)
            return None

        # Default selection:
        # - If CAPTIONER_BACKEND is set, honor it.
        # - Else if an accelerated torch device is available, prefer local Qwen captioner so the
        #   retrieval stack stays inside the Qwen3-VL family.
        # - Else if OPENAI_API_KEY exists, fall back to GPT on CPU.
        # - Else disable captioning (CPU local captioning is slow).
        if not backend or backend == "auto":
            if local_mode:
                backend = "qwen"
            elif os.getenv("OPENAI_API_KEY"):
                backend = "gpt"
            else:
                preferred_device = preferred_torch_device()
                if preferred_device == "cuda":
                    backend = "qwen"
                elif is_apple_mps_device(preferred_device):
                    backend = "none"
                else:
                    backend = "none"

        if backend in {"none", "off", "false", "0"}:
            logger.info("Captioner backend set to %s; captioning disabled.", backend)
            return None

        if backend in {"gpt", "openai"}:
            try:
                return GPTVLCaptioner(prompt=prompt)
            except Exception:
                logger.exception("Failed to initialize GPT captioner; disabling captioning.")
                return None

        if backend in {"qwen", "qwen3"}:
            try:
                return Qwen3VLCaptioner(prompt=prompt)
            except Exception:
                logger.exception("Failed to initialize Qwen captioner; disabling captioning.")
                return None

        logger.warning("Unknown CAPTIONER_BACKEND=%s; captioning disabled.", backend)
        return None

    def _init_reranker(self):
        reranker_setting = os.getenv("ENABLE_RERANKER")
        if reranker_setting is None:
            enabled = not is_apple_mps_device(preferred_torch_device())
        else:
            enabled = reranker_setting.strip().lower() not in {"0", "false", "off", "no"}
        if not enabled:
            logger.info("Reranker disabled by local configuration/default.")
            return None
        try:
            return Qwen3VLReranker()
        except Exception:
            logger.exception("Failed to initialize Qwen3-VL reranker; continuing without reranking.")
            return None

    def _primary_key_exists(self, primary_key: str) -> bool:
        rows = self.index.fetch_attributes([primary_key], output_fields=["pk"])
        return bool(rows)

    def _max_image_index(self, model_id: str) -> int:
        rows = self.index.query_attributes_by_model(model_id, output_fields=["pk"])
        pattern = re.compile(rf"^{re.escape(model_id)}::img_(\d+)$")
        max_index = 0
        for row in rows:
            pk_value = str(row.get("pk") or "")
            match = pattern.match(pk_value)
            if match:
                try:
                    candidate = int(match.group(1))
                except ValueError:
                    continue
                max_index = max(max_index, candidate)
        return max_index

    def _allocate_image_pk(self, model_id: str) -> str:
        next_index = self._image_pk_counters.get(model_id)
        if next_index is None:
            next_index = self._max_image_index(model_id) + 1
        pk_value = f"{model_id}::img_{next_index:03d}"
        self._image_pk_counters[model_id] = next_index + 1
        return pk_value

    def _normalize_images(self, images: Iterable[object]) -> List[Path]:
        normalized: List[Path] = []
        for image in images:
            candidate = None
            if isinstance(image, (str, Path)):
                candidate = Path(image)
            elif isinstance(image, dict):
                candidate = image.get("path") or image.get("name")
                if candidate:
                    candidate = Path(candidate)
            else:
                name_attr = getattr(image, "name", None)
                if name_attr:
                    candidate = Path(name_attr)
            if candidate is None:
                raise ValueError(f"Unsupported image input type: {type(image).__name__}")
            candidate = candidate.expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Image path does not exist: {candidate}")
            normalized.append(candidate)
        return normalized

    def _build_metadata_text(self, metadata: Dict[str, str]) -> str:
        parts = []
        maker = metadata.get("maker")
        if maker:
            parts.append(f"Maker: {maker}")
        part_number = metadata.get("part_number")
        if part_number:
            parts.append(f"Part Number: {part_number}")
        category = metadata.get("category")
        if category:
            parts.append(f"Category: {category}")
        status = metadata.get("status")
        if status:
            parts.append(f"Status: {status}")
        description = metadata.get("description")
        if description:
            parts.append(description)
        web_text = metadata.get("web_text")
        if web_text:
            parts.append(f"Web: {web_text}")
        price_text = metadata.get("price_text")
        if price_text:
            parts.append(f"Prices: {price_text}")
        return ". ".join(part for part in parts if part).strip()

    @staticmethod
    def _combine_model_text(metadata_text: str, ocr_text: str, caption_text: str) -> str:
        sections = []
        if metadata_text:
            sections.append(metadata_text)
        if ocr_text:
            sections.append(ocr_text.replace("\n", " "))
        if caption_text:
            sections.append(caption_text.replace("\n", " "))
        return " ".join(section for section in sections if section).strip()

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        return tokenize_text(text)

    @staticmethod
    def _extract_spec_tokens(text: str) -> List[str]:
        value = (text or "").lower().replace(" ", "")
        # Examples: 16v, 3a, 220vac, 50hz, 500mah
        return re.findall(r"\d+(?:\.\d+)?(?:v|a|w|hz|mah|vac|vdc|amp|amps|volt|volts)", value)

    def _compute_lexical_score(self, query_text: str, haystacks: Sequence[str]) -> float:
        return compute_lexical_score(query_text, haystacks)

    @staticmethod
    def _compute_exact_field_boost(query_text: str, *, model_id: str, maker: str, part_number: str, description: str) -> float:
        return compute_exact_field_boost(
            query_text,
            model_id=model_id,
            maker=maker,
            part_number=part_number,
            description=description,
        )

    @staticmethod
    def _passes_min_score(*, score: float, lexical_hit: bool, exact_field_boost: float) -> bool:
        return passes_min_score(
            score=score,
            lexical_hit=lexical_hit,
            exact_field_boost=exact_field_boost,
        )

    def _compute_spec_score(
        self,
        *,
        query_text: str,
        haystacks: Sequence[str],
        part_number: str,
    ) -> float:
        q_specs = set(self._extract_spec_tokens(query_text))
        if not q_specs and not query_text:
            return 0.0
        joined = " ".join(haystacks).lower().replace(" ", "")
        part_norm = (part_number or "").strip().lower()
        query_norm = (query_text or "").strip().lower()

        spec_overlap = 0.0
        if q_specs:
            matched = sum(1 for token in q_specs if token in joined)
            spec_overlap = matched / max(1.0, float(len(q_specs)))

        exact_part_bonus = 0.0
        if part_norm and query_norm:
            if part_norm == query_norm:
                exact_part_bonus = 1.0
            elif part_norm in query_norm or query_norm in part_norm:
                exact_part_bonus = 0.6

        return min(1.0, spec_overlap * 0.75 + exact_part_bonus * 0.8)

    @staticmethod
    def _merge_caption_text(existing_text: str, new_caption: str) -> str:
        new_caption = (new_caption or "").strip()
        if not new_caption:
            return existing_text or ""
        existing_text = existing_text or ""
        if new_caption.lower() in existing_text.lower():
            return existing_text
        if not existing_text:
            return new_caption
        return f"{existing_text}\n{new_caption}"

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _merge_ocr_text(existing_text: str, new_tokens) -> str:
        existing_lines = [line.strip() for line in (existing_text or "").split("\n") if line.strip()]
        existing_set = set(existing_lines)
        for token in new_tokens:
            text = str(token).strip()
            if not text or text in existing_set:
                continue
            existing_lines.append(text)
            existing_set.add(text)
        return "\n".join(existing_lines)

    def index_model_metadata(self, model_id: str, metadata: Dict[str, str]) -> Dict[str, object]:
        existing = self.index.get_model(
            model_id,
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
        ) or {}
        payload = dict(metadata)
        payload["model_id"] = model_id

        merged_payload = {"model_id": model_id}
        for key in ("maker", "part_number", "category", "description"):
            new_value = payload.get(key)
            if new_value is not None and str(new_value).strip():
                merged_payload[key] = str(new_value).strip()
            else:
                existing_value = existing.get(key)
                if existing_value is not None and str(existing_value).strip():
                    merged_payload[key] = str(existing_value).strip()

        for key in ("status", "web_text", "price_text"):
            new_value = payload.get(key)
            if new_value is not None and str(new_value).strip():
                merged_payload[key] = str(new_value).strip()

        normalized = self.metadata_normalizer.normalize(merged_payload)
        normalized["model_id"] = model_id
        metadata_text = self._build_metadata_text(normalized)
        ocr_text = existing.get("ocr_text", "")
        caption_text = existing.get("caption_text", "")
        logger.debug("Indexing model metadata: model_id=%s, metadata_text_len=%d, existing_ocr_len=%d",
                     model_id, len(metadata_text), len(ocr_text))

        combined_text = self._combine_model_text(metadata_text, ocr_text, caption_text)
        vector = self.text_encoder.encode_document(combined_text or " ").tolist()

        extra_values = {
            "metadata_text": metadata_text,
            "ocr_text": ocr_text,
            "caption_text": caption_text,
            "maker": normalized.get("maker") or existing.get("maker", ""),
            "part_number": normalized.get("part_number") or existing.get("part_number", ""),
            "category": normalized.get("category") or existing.get("category", ""),
            "description": normalized.get("description") or existing.get("description", ""),
        }

        self.index.upsert_model(model_id=model_id, text_vector=vector, extra_values=extra_values)
        updated = self.index.get_model(
            model_id,
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
        )
        logger.info("Model metadata indexed: model_id=%s, combined_text_tokens=%d",
                    model_id, len(vector))
        return updated or {}

    def _update_model_with_texts(
        self,
        model_id: str,
        model_record: Dict[str, object],
        normalized_metadata: Dict[str, str],
        ocr_tokens,
        caption_text: str,
    ) -> Dict[str, object]:
        existing_metadata_text = model_record.get("metadata_text") or self._build_metadata_text(normalized_metadata)
        existing_ocr_text = model_record.get("ocr_text", "")
        existing_caption_text = model_record.get("caption_text", "")
        merged_ocr_text = self._merge_ocr_text(existing_ocr_text, ocr_tokens)
        merged_caption_text = self._merge_caption_text(existing_caption_text, caption_text)

        if merged_ocr_text == existing_ocr_text and merged_caption_text == existing_caption_text:
            logger.debug("No new OCR/caption updates detected for model_id=%s", model_id)
            return model_record

        combined_text = self._combine_model_text(existing_metadata_text, merged_ocr_text, merged_caption_text)
        vector = self.text_encoder.encode_document(combined_text or " ").tolist()
        extra_values = {
            "metadata_text": existing_metadata_text,
            "ocr_text": merged_ocr_text,
            "caption_text": merged_caption_text,
            "maker": normalized_metadata.get("maker", model_record.get("maker", "")),
            "part_number": normalized_metadata.get("part_number", model_record.get("part_number", "")),
            "category": normalized_metadata.get("category", model_record.get("category", "")),
            "description": normalized_metadata.get("description", model_record.get("description", "")),
        }
        self.index.upsert_model(model_id=model_id, text_vector=vector, extra_values=extra_values)
        updated = self.index.get_model(
            model_id,
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
        )
        logger.info(
            "Merged OCR/caption for model_id=%s, ocr_lines=%d, caption_chars=%d",
            model_id,
            len(merged_ocr_text.split("\n")) if merged_ocr_text else 0,
            len(merged_caption_text),
        )
        return updated or model_record

    @staticmethod
    def _connect_milvus(config: MilvusConnectionConfig) -> None:
        logger.info("Connecting to Milvus at %s", config.uri)
        connections.connect(
            alias="default",
            uri=config.uri,
            user=config.user,
            password=config.password,
            token=config.token,
            db_name=config.db_name,
        )

    def preprocess_and_index(self, image_path: str | Path, metadata: Dict[str, str], *, enable_ocr: bool | None = None) -> None:
        """Run preprocessing pipeline and insert results into Milvus."""
        metadata = dict(metadata)
        model_id = metadata.get("model_id")
        if not model_id:
            raise ValueError("metadata must include a non-empty 'model_id' field for indexing.")

        logger.info("Starting indexing for model_id=%s with image=%s", model_id, image_path)

        t_start = time.perf_counter()
        model_record = self.index_model_metadata(model_id, metadata)
        t_after_model = time.perf_counter()
        logger.info("Timing: index_model_metadata done in %.2fs for model_id=%s", t_after_model - t_start, model_id)

        embedding_image_path = None
        try:
            embedding_image_path = self._prepare_embedding_image(image_path)
            record = self.preprocessing(
                str(image_path),
                metadata,
                enable_ocr=self.enable_ocr_indexing if enable_ocr is None else bool(enable_ocr),
                embedding_image_path=str(embedding_image_path),
            )
            t_after_preprocess = time.perf_counter()
            logger.info("Timing: preprocessing done in %.2fs for model_id=%s", t_after_preprocess - t_after_model, model_id)

            normalized_metadata = dict(record.metadata)
            logger.debug(
                "Image preprocessed: model_id=%s, image_dim=%d, ocr_dim=%d, caption_dim=%d",
                model_id,
                len(record.image_vector),
                len(record.ocr_vector),
                len(record.caption_vector),
            )

            model_record = self._update_model_with_texts(
                model_id,
                model_record,
                normalized_metadata,
                record.ocr_tokens,
                record.caption_text,
            )
            t_after_update_texts = time.perf_counter()
            logger.info("Timing: _update_model_with_texts done in %.2fs for model_id=%s",
                        t_after_update_texts - t_after_preprocess, model_id)

            primary_key = normalized_metadata.get("pk") or metadata.get("pk")
            if primary_key and self._primary_key_exists(primary_key):
                logger.warning(
                    "Duplicate primary key detected for model_id=%s (pk=%s); allocating a new key.",
                    model_id,
                    primary_key,
                )
                primary_key = None

            if not primary_key:
                primary_key = self._allocate_image_pk(model_id)

            normalized_metadata["pk"] = primary_key

            ocr_attr_text = record.ocr_text or "\n".join(record.ocr_tokens)
            caption_attr_text = record.caption_text
            attrs_payload = {
                "model_id": model_id,
                "maker": normalized_metadata.get("maker", ""),
                "part_number": normalized_metadata.get("part_number", ""),
                "category": normalized_metadata.get("category", ""),
                "ocr_text": self._truncate_text(ocr_attr_text, 2048),
                "caption_text": self._truncate_text(caption_attr_text, 2048),
                "image_path": "",
            }

            stored_filename = f"{primary_key}.jpg"
            stored_path = self.media_root / stored_filename
            try:
                shutil.copy(str(embedding_image_path), stored_path)
                attrs_payload["image_path"] = str(stored_path)
                logger.debug("Stored resized image copy at %s", stored_path)
            except Exception as exc:
                logger.warning("Failed to store image copy for %s: %s", primary_key, exc)

            t_before_insert = time.perf_counter()
            self.index.insert(
                primary_keys=[primary_key],
                image_vectors=[record.image_vector.tolist()],
                text_vectors=[record.ocr_vector.tolist()],
                attrs_rows=[attrs_payload],
                caption_vectors=[record.caption_vector.tolist()],
            )
            self.index.flush()
            t_after_flush = time.perf_counter()
            logger.info(
                "Timing: insert+flush done in %.2fs for model_id=%s",
                t_after_flush - t_before_insert,
                model_id,
            )
            logger.info("Completed indexing: model_id=%s, image_pk=%s, tokens=%d",
                        model_id, primary_key, len(record.ocr_tokens))
        finally:
            if embedding_image_path is not None:
                Path(embedding_image_path).unlink(missing_ok=True)

    @staticmethod
    def _prepare_embedding_image(image_path: str | Path, *, max_side: int = 1024, quality: int = 85) -> Path:
        source_path = Path(image_path)
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)
            if longest > max_side:
                scale = max_side / float(longest)
                target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                logger.info(
                    "Prepared resized embedding image: source=%s original=%sx%s resized=%sx%s",
                    source_path,
                    width,
                    height,
                    target_size[0],
                    target_size[1],
                )
            else:
                logger.info(
                    "Prepared embedding image without resize: source=%s size=%sx%s",
                    source_path,
                    width,
                    height,
                )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG", quality=quality, optimize=True)
                return Path(tmp.name)

    @staticmethod
    def _prefix_from_category(category: str | None) -> str:
        value = (category or "").strip().lower()
        for ch in value:
            if ch.isalnum():
                return ch
        return "a"

    def allocate_model_id(
        self,
        *,
        category: str | None = None,
        prefix: str | None = None,
        width: int = 6,
    ) -> str:
        """Allocate a sequential model_id like 'a000001' based on a prefix.

        Designed for single-writer deployments (one API instance).
        """
        width = int(width)
        if width < 3 or width > 12:
            raise ValueError("width must be between 3 and 12.")

        prefix_value = (prefix or "").strip().lower() or self._prefix_from_category(category)
        prefix_value = prefix_value[:8]
        counter_key = f"model_id::{prefix_value}"
        counter = self.counters.next(counter_key)
        return f"{prefix_value}{counter.value:0{width}d}"

    def _ensure_tracker_dataset(self, dataset_path: str | Path | None = None) -> TrackerDataset:
        if dataset_path:
            self._tracker_dataset_path = Path(dataset_path).expanduser().resolve()
            self._tracker_dataset = None
        if self._tracker_dataset is None:
            path = self._tracker_dataset_path
            if path is None:
                path = Path(os.getenv("TRACKER_DATASET_PATH", "data/tracker_subset.csv")).expanduser().resolve()
            self._tracker_dataset = TrackerDataset.from_csv(path)
        return self._tracker_dataset

    def index_tracker_model(
        self,
        model_id: str,
        *,
        images_root: str | Path = "data/images",
        dataset_path: str | Path | None = None,
        halt_on_error: bool = False,
    ) -> Dict[str, List[str]]:
        """Index a single model by looking up metadata and images from the tracker dataset."""
        dataset = self._ensure_tracker_dataset(dataset_path)
        cleaned_id = (model_id or "").strip()
        if not cleaned_id:
            raise ValueError("model_id must be provided.")
        record = dataset.get(cleaned_id)
        if record is None:
            raise KeyError(f"Model ID '{cleaned_id}' not found in tracker dataset.")

        images_dir = Path(images_root).expanduser().resolve() / record.model_id
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found for model_id={record.model_id}: {images_dir}")

        image_files = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in {images_dir}")

        metadata = {
            "model_id": record.model_id,
            "maker": record.std_maker_name,
            "category": record.category_code,
            "part_number": record.std_model_name or record.non_std_model_name or "",
            "description": record.model_name,
            "model_name": record.model_name,
        }

        entry = {"metadata": metadata, "images": image_files}
        return self.bulk_index([entry], halt_on_error=halt_on_error)

    def bulk_index(
        self,
        batches: Sequence[Dict[str, Any]],
        *,
        halt_on_error: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Bulk ingest models and associated images.

        Each batch item should follow the shape:
            {
                "metadata": {"model_id": ..., "maker": ..., ...},  # optional; fields can also live at top-level
                "images": ["/path/to/img1.png", "/path/to/img2.png", ...]
            }
        """
        successes: List[str] = []
        failures: List[str] = []

        for entry in batches:
            metadata = dict(entry.get("metadata") or {})
            for key in ("model_id", "maker", "part_number", "category", "description"):
                if key in entry and key not in metadata:
                    metadata[key] = entry[key]

            model_id = (metadata.get("model_id") or "").strip()
            if not model_id:
                message = "Skipped entry without model_id."
                failures.append(message)
                if halt_on_error:
                    raise ValueError(message)
                continue

            metadata["model_id"] = model_id
            images_raw = entry.get("images") or []
            if isinstance(images_raw, (str, Path)):
                images_iterable: Iterable[object] = [images_raw]
            else:
                images_iterable = list(images_raw)

            try:
                logger.info("Bulk indexing metadata for model_id=%s", model_id)
                self.index_model_metadata(model_id, metadata)
                image_paths = self._normalize_images(images_iterable)
                if not image_paths:
                    successes.append(model_id)
                    continue

                for image_path in image_paths:
                    logger.info("Bulk indexing image for model_id=%s path=%s", model_id, image_path)
                    self.preprocess_and_index(image_path, metadata)
                successes.append(model_id)
            except Exception as exc:
                error_message = f"{model_id}: {exc}"
                failures.append(error_message)
                logger.exception("Bulk indexing failed for model_id=%s", model_id)
                if halt_on_error:
                    raise

        return {"indexed": successes, "errors": failures}

    def search(
        self,
        *,
        query_image: Optional[str | Path] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        part_number: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ):
        """Execute hybrid search using provided image and/or text query."""
        if not query_image and not query_text:
            logger.warning("Search invoked without query_image or query_text.")
            return []

        t_start = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        weights = self.retriever.weights
        alpha = weights.alpha
        beta = weights.beta
        gamma = getattr(weights, "gamma", 0.0)

        part_number_query = (part_number or "").strip()
        part_number_query_norm = part_number_query.upper() if part_number_query else ""

        model_scores: Dict[str, Dict[str, object]] = {}
        model_images: Dict[str, List[Dict[str, object]]] = {}
        query_record = None

        search_k = max(top_k, min(100, top_k * 3))

        if query_image:
            logger.info("Running image/OCR/caption search")
            t_preprocess = time.perf_counter()
            query_record = self.preprocessing(
                str(query_image),
                {},
                enable_ocr=self.enable_ocr_query,
            )
            timings_ms["preprocessing"] = round((time.perf_counter() - t_preprocess) * 1000, 2)

            image_vector = query_record.image_vector.tolist()
            t_image_search = time.perf_counter()
            image_results = self.index.search_images(image_vector, top_k=search_k)
            timings_ms["image_search"] = round((time.perf_counter() - t_image_search) * 1000, 2)
            attr_cache = self._fetch_attrs_for_hits(image_results)
            logger.debug("Image search returned %d candidates", len(image_results))
            for hit in image_results:
                data = attr_cache.get(str(hit.id))
                model_id = data.get("model_id") if data else None
                if not model_id:
                    continue
                similarity = self._distance_to_similarity(hit.distance)
                entry = model_scores.setdefault(
                    model_id,
                    {"image_sims": [], "ocr_sims": [], "caption_sims": [], "text_query_sims": []},
                )
                entry["image_sims"].append(similarity)
                image_info = {
                    "image_id": str(hit.id),
                    "distance": hit.distance,
                    "similarity": similarity,
                    "maker": data.get("maker", "") if data else "",
                    "part_number": data.get("part_number", "") if data else "",
                    "category": data.get("category", "") if data else "",
                    "image_path": data.get("image_path", "") if data else "",
                    "caption_text": data.get("caption_text", "") if data else "",
                }
                model_images.setdefault(model_id, []).append(image_info)

            ocr_query_text = query_record.ocr_text or "\n".join(query_record.ocr_tokens)
            if ocr_query_text:
                ocr_query_vector = self.text_encoder.encode_query(ocr_query_text).tolist()
                t_ocr_search = time.perf_counter()
                text_results = self.index.search_texts(ocr_query_vector, top_k=search_k)
                timings_ms["ocr_search"] = round((time.perf_counter() - t_ocr_search) * 1000, 2)
                attr_cache = self._fetch_attrs_for_hits(text_results)
                logger.debug("OCR text search returned %d candidates", len(text_results))
                for hit in text_results:
                    data = attr_cache.get(str(hit.id))
                    model_id = data.get("model_id") if data else None
                    if not model_id:
                        continue
                    similarity = self._distance_to_similarity(hit.distance)
                    entry = model_scores.setdefault(
                        model_id,
                        {"image_sims": [], "ocr_sims": [], "caption_sims": [], "text_query_sims": []},
                    )
                    entry["ocr_sims"].append(similarity)
            else:
                timings_ms["ocr_search"] = 0.0

            if query_record.caption_text:
                caption_query_vector = self.text_encoder.encode_query(query_record.caption_text).tolist()
                t_caption_search = time.perf_counter()
                caption_results = self.index.search_captions(caption_query_vector, top_k=search_k)
                timings_ms["caption_search"] = round((time.perf_counter() - t_caption_search) * 1000, 2)
                attr_cache = self._fetch_attrs_for_hits(caption_results)
                logger.debug("Caption search returned %d candidates", len(caption_results))
                for hit in caption_results:
                    data = attr_cache.get(str(hit.id))
                    model_id = data.get("model_id") if data else None
                    if not model_id:
                        continue
                    similarity = self._distance_to_similarity(hit.distance)
                    entry = model_scores.setdefault(
                        model_id,
                        {"image_sims": [], "ocr_sims": [], "caption_sims": [], "text_query_sims": []},
                    )
                    entry["caption_sims"].append(similarity)
            else:
                timings_ms["caption_search"] = 0.0

        if query_text:
            logger.info("Running text search")
            t_text_search = time.perf_counter()
            text_vector = self.text_encoder.encode_query(query_text).tolist()
            model_results = self.index.search_models(text_vector, top_k=search_k)
            timings_ms["text_search"] = round((time.perf_counter() - t_text_search) * 1000, 2)
            logger.debug("Text search returned %d model candidates", len(model_results))
            for hit in model_results:
                model_id = str(hit.id)
                similarity = self._distance_to_similarity(hit.distance)
                entry = model_scores.setdefault(
                    model_id,
                    {"image_sims": [], "ocr_sims": [], "caption_sims": [], "text_query_sims": []},
                )
                entry["ocr_sims"].append(similarity)
                entry["text_query_sims"].append(similarity)
        elif query_text is not None:
            timings_ms["text_search"] = 0.0

        if not model_scores:
            timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 2)
            logger.warning("No candidates retrieved for query.")
            logger.info(
                "Search completed: query_image=%s, query_text=%s, results=0 timings_ms=%s",
                bool(query_image),
                bool(query_text),
                timings_ms,
            )
            return []

        model_ids = list(model_scores.keys())
        t_fetch_models = time.perf_counter()
        model_rows = self.index.fetch_models(
            model_ids,
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
        )
        timings_ms["fetch_models"] = round((time.perf_counter() - t_fetch_models) * 1000, 2)

        query_lower = (query_text or "").strip().lower() if query_text else ""
        results = []
        t_finalize = time.perf_counter()
        for model_id, score_data in model_scores.items():
            info = model_rows.get(model_id, {})
            image_sims = score_data.get("image_sims", [])
            image_score = max(image_sims) if image_sims else 0.0
            ocr_sims = score_data.get("ocr_sims", [])
            caption_sims = score_data.get("caption_sims", [])
            text_query_sims = score_data.get("text_query_sims", [])
            ocr_score = max(ocr_sims) if ocr_sims else 0.0
            caption_score = max(caption_sims) if caption_sims else 0.0
            text_query_score = max(text_query_sims) if text_query_sims else 0.0

            weight_sum = 0.0
            weighted_score = 0.0
            if image_sims:
                weight_sum += alpha
                weighted_score += alpha * image_score
            if ocr_score > 0.0:
                weight_sum += beta
                weighted_score += beta * ocr_score
            if caption_score > 0.0 and gamma > 0.0:
                weight_sum += gamma
                weighted_score += gamma * caption_score
            final_score = weighted_score / weight_sum if weight_sum > 0 else 0.0

            images = model_images.get(model_id, [])
            images.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)

            metadata_text = info.get("metadata_text", "")
            ocr_text = info.get("ocr_text", "")
            caption_text = info.get("caption_text", "")
            aggregated_text = self._combine_model_text(metadata_text, ocr_text, caption_text)
            part_number_value_raw = info.get("part_number", "")
            part_number_value = part_number_value_raw or ""
            lexical_hit = False
            lexical_score = 0.0
            spec_match_score = 0.0
            exact_field_boost = 0.0
            if query_lower:
                haystacks = [
                    metadata_text,
                    ocr_text,
                    caption_text,
                    aggregated_text,
                    info.get("maker", "") or "",
                    part_number_value,
                    info.get("description", "") or "",
                ]
                lexical_hit = any(query_lower in (haystack or "").lower() for haystack in haystacks)
                lexical_score = self._compute_lexical_score(query_text or "", haystacks)
                spec_match_score = self._compute_spec_score(
                    query_text=query_text or "",
                    haystacks=haystacks,
                    part_number=part_number_value,
                )
                exact_field_boost = self._compute_exact_field_boost(
                    query_text or "",
                    model_id=model_id,
                    maker=info.get("maker", "") or "",
                    part_number=part_number_value,
                    description=info.get("description", "") or "",
                )
                # Blend dense + lexical + exact/spec evidence.
                final_score = min(
                    1.0,
                    final_score * 0.45
                    + lexical_score * 0.20
                    + spec_match_score * 0.15
                    + exact_field_boost * 0.20,
                )
                if lexical_hit:
                    final_score = min(1.0, final_score + 0.08)
                if exact_field_boost >= 0.55:
                    final_score = min(1.0, final_score + 0.12)
                elif exact_field_boost > 0.0:
                    final_score = min(1.0, final_score + 0.06)

            if not self._passes_min_score(
                score=final_score,
                lexical_hit=lexical_hit,
                exact_field_boost=exact_field_boost,
            ):
                continue
            result = {
                "model_id": model_id,
                "score": final_score,
                "image_score": image_score,
                "ocr_score": ocr_score,
                "caption_score": caption_score,
                "text_query_score": text_query_score,
                "maker": info.get("maker", ""),
                "part_number": part_number_value,
                "category": info.get("category", ""),
                "description": info.get("description", ""),
                "metadata_text": metadata_text,
                "ocr_text": ocr_text,
                "caption_text": caption_text,
                "aggregated_text": aggregated_text,
                "images": images,
                "lexical_hit": lexical_hit,
                "lexical_score": lexical_score,
                "spec_match_score": spec_match_score,
                "exact_field_boost": exact_field_boost,
            }
            results.append(result)

        if part_number_query_norm:
            filtered = [item for item in results if (item.get("part_number", "").upper() == part_number_query_norm)]
            if filtered:
                results = filtered

        for item in results:
            item["verified"] = bool(part_number_query_norm and item.get("part_number", "").upper() == part_number_query_norm)

        results.sort(
            key=lambda item: (
                item.get("exact_field_boost", 0.0),
                item.get("lexical_hit", False),
                item.get("score", 0.0),
            ),
            reverse=True,
        )
        rerank_top_n = min(len(results), max(top_k, min(20, top_k * 2)))
        reranker_enabled = self.reranker is not None if use_reranker is None else bool(use_reranker) and self.reranker is not None
        if reranker_enabled and rerank_top_n > 1:
            try:
                reranked = self.retriever.rerank(
                    {
                        "text": query_text or "",
                        "image": str(query_image) if query_image else "",
                    },
                    results[:rerank_top_n],
                )
                results = reranked + results[rerank_top_n:]
            except Exception:
                logger.exception("Qwen3-VL reranking failed; keeping base ranking.")
        timings_ms["finalize"] = round((time.perf_counter() - t_finalize) * 1000, 2)
        timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "Search completed: query_image=%s, query_text=%s, results=%d timings_ms=%s",
            bool(query_image),
            bool(query_text),
            len(results[:top_k]),
            timings_ms,
        )
        return results[:top_k]

    @staticmethod
    def _pad_scores(scores, target_len):
        if len(scores) >= target_len:
            return scores
        return scores + [0.0] * (target_len - len(scores))

    @staticmethod
    def _noop_cross_encoder(query, candidates):
        """Placeholder cross-encoder that returns zero scores."""
        return [0.0 for _ in candidates]

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - value)

    def _fetch_attrs_for_hits(self, hits):
        ids = [str(hit.id) for hit in hits]
        if not ids:
            return {}
        rows = self.index.fetch_attributes(
            ids,
            output_fields=["pk", "model_id", "maker", "part_number", "category", "ocr_text", "caption_text", "image_path"],
        )
        return {row["pk"]: row for row in rows}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = HybridSearchOrchestrator()
    sample_metadata = {
        "maker": "Unknown",
        "part_number": "PN-0000",
        "category": "UNSPECIFIED",
    }
    sample_image = Path("sample.jpg")
    if sample_image.exists():
        orchestrator.preprocess_and_index(sample_image, sample_metadata)
        results = orchestrator.search(query_image=sample_image, top_k=3)
        for result in results:
            logger.info("Result: %s", result)
    else:
        logger.info("Add a sample.jpg alongside this script to test ingestion.")
