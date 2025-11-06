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
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import DataType, connections

from .data_collection.tracker_dataset import TrackerDataset
from .preprocessing.captioning.qwen3_captioner import Qwen3VLCaptioner
from .preprocessing.embedding.bge_m3_encoder import BGEM3TextEncoder
from .preprocessing.embedding.bge_vl_encoder import BGEVLImageEncoder
from .preprocessing.metadata_normalizer import MetadataNormalizer
from .preprocessing.ocr.OCR import PaddleOCRVLPipeline
from .preprocessing.pipeline import PreprocessingPipeline
from .retrieval.milvus_hybrid_index import CollectionConfig, FieldSpec, HybridMilvusIndex
from .search.fusion_retriever import FusionWeights, HybridFusionRetriever

logger = logging.getLogger(__name__)


@dataclass
class MilvusConnectionConfig:
    uri: str = "tcp://standalone:19530"
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    db_name: Optional[str] = None


class HybridSearchOrchestrator:
    """Top-level orchestrator that manages preprocessing, indexing, and search."""

    def __init__(
        self,
        *,
        milvus: MilvusConnectionConfig = MilvusConnectionConfig(),
        image_collection: str = "image_parts",
        text_collection: str = "text_parts",
        attrs_collection: str = "attrs_parts",
        model_collection: str = "model_texts",
        fusion_weights: FusionWeights = FusionWeights(alpha=0.5, beta=0.3, gamma=0.2),
        tracker_dataset_path: str | Path | None = None,
    ) -> None:
        self._connect_milvus(milvus)
        self.vision_encoder = BGEVLImageEncoder()
        self.text_encoder = BGEM3TextEncoder()
        self.ocr_engine = PaddleOCRVLPipeline()
        self.captioner = Qwen3VLCaptioner(
            prompt=(
                "Describe this industrial component in detail, including its visible shape, color, "
                "material, labels, and any text on it."
            )
        )
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
            caption_cfg=CollectionConfig(name="caption_parts", dimension=self.text_encoder.embedding_dim),
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
            caption_collection_name="caption_parts",
        )
        self.index.create_indexes()
        self.index.load()

        self.retriever = HybridFusionRetriever(
            cross_encoder=self._noop_cross_encoder,
            weights=fusion_weights,
        )

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
        payload = dict(metadata)
        payload["model_id"] = model_id
        normalized = self.metadata_normalizer.normalize(payload)
        normalized["model_id"] = model_id
        metadata_text = self._build_metadata_text(normalized)

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
            "maker": normalized.get("maker", existing.get("maker", "")),
            "part_number": normalized.get("part_number", existing.get("part_number", "")),
            "category": normalized.get("category", existing.get("category", "")),
            "description": normalized.get("description", existing.get("description", "")),
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

    def preprocess_and_index(self, image_path: str | Path, metadata: Dict[str, str]) -> None:
        """Run preprocessing pipeline and insert results into Milvus."""
        metadata = dict(metadata)
        model_id = metadata.get("model_id")
        if not model_id:
            raise ValueError("metadata must include a non-empty 'model_id' field for indexing.")

        logger.info("Starting indexing for model_id=%s with image=%s", model_id, image_path)
        model_record = self.index_model_metadata(model_id, metadata)

        record = self.preprocessing(str(image_path), metadata)
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

        ocr_attr_text = record.ocr_text or "\n".join(record.ocr_tokens) or record.text_corpus
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

        stored_filename = f"{primary_key}.png"
        stored_path = self.media_root / stored_filename
        try:
            shutil.copy(str(image_path), stored_path)
            attrs_payload["image_path"] = str(stored_path)
            logger.debug("Stored image copy at %s", stored_path)
        except Exception as exc:
            logger.warning("Failed to store image copy for %s: %s", primary_key, exc)

        self.index.insert(
            primary_keys=[primary_key],
            image_vectors=[record.image_vector.tolist()],
            text_vectors=[record.ocr_vector.tolist()],
            attrs_rows=[attrs_payload],
            caption_vectors=[record.caption_vector.tolist()],
        )
        self.index.flush()
        logger.info("Completed indexing: model_id=%s, image_pk=%s, tokens=%d",
                    model_id, primary_key, len(record.ocr_tokens))

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
    ):
        """Execute hybrid search using provided image and/or text query."""
        if not query_image and not query_text:
            logger.warning("Search invoked without query_image or query_text.")
            return []

        weights = self.retriever.weights
        alpha = weights.alpha
        beta = weights.beta
        gamma = getattr(weights, "gamma", 0.0)

        part_number_query = (part_number or "").strip()
        part_number_query_norm = part_number_query.upper() if part_number_query else ""

        model_scores: Dict[str, Dict[str, object]] = {}
        model_images: Dict[str, List[Dict[str, object]]] = {}
        query_record = None

        if query_image:
            logger.info("Running image/OCR/caption search")
            query_record = self.preprocessing(str(query_image), {})

            image_vector = query_record.image_vector.tolist()
            image_results = self.index.search_images(image_vector, top_k=top_k)
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
                text_results = self.index.search_texts(ocr_query_vector, top_k=top_k)
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

            if query_record.caption_text:
                caption_query_vector = self.text_encoder.encode_query(query_record.caption_text).tolist()
                caption_results = self.index.search_captions(caption_query_vector, top_k=top_k)
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

        if query_text:
            logger.info("Running text search")
            text_vector = self.text_encoder.encode_query(query_text).tolist()
            model_results = self.index.search_models(text_vector, top_k=top_k)
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

        if not model_scores:
            logger.warning("No candidates retrieved for query.")
            return []

        model_ids = list(model_scores.keys())
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

        query_lower = (query_text or "").strip().lower() if query_text else ""
        results = []
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
                if lexical_hit:
                    final_score = min(1.0, final_score + 0.25)
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
            }
            results.append(result)

        if part_number_query_norm:
            filtered = [item for item in results if (item.get("part_number", "").upper() == part_number_query_norm)]
            if filtered:
                results = filtered

        for item in results:
            item["verified"] = bool(part_number_query_norm and item.get("part_number", "").upper() == part_number_query_norm)

        results.sort(key=lambda item: (item.get("lexical_hit", False), item.get("score", 0.0)), reverse=True)
        logger.info("Search completed: query_image=%s, query_text=%s, results=%d",
                    bool(query_image), bool(query_text), len(results[:top_k]))
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
