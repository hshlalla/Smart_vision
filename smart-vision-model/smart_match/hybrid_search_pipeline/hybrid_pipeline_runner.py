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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pymilvus import DataType, connections

from .preprocessing.embedding.bge_m3_encoder import BGEM3TextEncoder
from .preprocessing.embedding.bge_vl_encoder import BGEVLImageEncoder
from .preprocessing.metadata_normalizer import MetadataNormalizer
from .preprocessing.ocr.OCR import PaddleOCRVLPipeline
from .preprocessing.pipeline import PreprocessingPipeline
from .retrieval.milvus_hybrid_index import CollectionConfig, HybridMilvusIndex
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
        fusion_weights: FusionWeights = FusionWeights(alpha=0.5, beta=0.5),
    ) -> None:
        self._connect_milvus(milvus)
        self.vision_encoder = BGEVLImageEncoder()
        self.text_encoder = BGEM3TextEncoder()
        self.ocr_engine = PaddleOCRVLPipeline(use_angle_cls=True,cls_model_dir="~/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer")
        self.metadata_normalizer = MetadataNormalizer()
        self.preprocessing = PreprocessingPipeline(
            vision_encoder=self.vision_encoder,
            ocr_engine=self.ocr_engine,
            text_encoder=self.text_encoder,
            metadata_normalizer=self.metadata_normalizer,
        )

        self.index = HybridMilvusIndex(
            image_cfg=CollectionConfig(name=image_collection, dimension=self.vision_encoder.embedding_dim),
            text_cfg=CollectionConfig(name=text_collection, dimension=self.text_encoder.embedding_dim),
            attrs_fields=[
                ("model_id", DataType.VARCHAR),
                ("maker", DataType.VARCHAR),
                ("part_number", DataType.VARCHAR),
                ("category", DataType.VARCHAR),
                ("ocr_text", DataType.VARCHAR),
            ],
            image_collection_name=image_collection,
            text_collection_name=text_collection,
            attrs_collection_name=attrs_collection,
        )
        self.index.create_indexes()
        self.index.load()

        self.retriever = HybridFusionRetriever(
            cross_encoder=self._noop_cross_encoder,
            weights=fusion_weights,
        )

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
        record = self.preprocessing(str(image_path), metadata)
        model_id = record.metadata.get("model_id") or metadata.get("model_id")
        if not model_id:
            raise ValueError("metadata must include a non-empty 'model_id' field for indexing.")
        primary_key = record.metadata.get("pk") or metadata.get("pk")
        if not primary_key:
            primary_key = str(model_id)

        attrs_payload = {
            "model_id": model_id,
            "maker": record.metadata.get("maker", ""),
            "part_number": record.metadata.get("part_number", ""),
            "category": record.metadata.get("category", ""),
            "ocr_text": record.text_corpus,
        }
        self.index.insert(
            primary_keys=[primary_key],
            image_vectors=[record.image_vector.tolist()],
            text_vectors=[record.text_vector.tolist()],
            attrs_rows=[attrs_payload],
        )
        self.index.flush()
        logger.info("Indexed asset %s with metadata %s", image_path, attrs_payload)

    def search(
        self,
        *,
        query_image: Optional[str | Path] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        part_number: Optional[str] = None,
    ):
        """Execute hybrid search using provided image and/or text query."""
        img_scores = []
        txt_scores = []
        candidates = []

        attr_cache: Dict[str, Dict[str, object]] = {}

        if query_image:
            image_vector = self.vision_encoder.encode(str(query_image)).tolist()
            image_results = self.index.search_images(image_vector, top_k=top_k)
            attr_cache.update(self._fetch_attrs_for_hits(image_results))
            img_scores = [hit.distance for hit in image_results]
            candidates = [
                {
                    "id": str(hit.id),
                    "source": "image",
                    "distance": hit.distance,
                    **attr_cache.get(str(hit.id), {}),
                }
                for hit in image_results
            ]

        if query_text:
            text_vector = self.text_encoder.encode(query_text).tolist()
            text_results = self.index.search_texts(text_vector, top_k=top_k)
            attr_cache.update(self._fetch_attrs_for_hits(text_results))
            txt_scores = [hit.distance for hit in text_results]
            if not candidates:
                candidates = [
                    {
                        "id": str(hit.id),
                        "source": "text",
                        "distance": hit.distance,
                        **attr_cache.get(str(hit.id), {}),
                    }
                    for hit in text_results
                ]

        if not candidates:
            logger.warning("No candidates retrieved for query.")
            return []

        # Align score arrays for fusion; fallback to zeros if one modality missing
        max_len = max(len(img_scores), len(txt_scores), len(candidates))
        img_scores = self._pad_scores(img_scores, max_len)
        txt_scores = self._pad_scores(txt_scores, max_len)

        fusion = self.retriever.fuse_scores(img_scores, txt_scores)
        reranked = sorted(zip(candidates, fusion), key=lambda x: x[1], reverse=True)

        part_number = part_number or ""
        final_results = []
        for candidate, score in reranked[:top_k]:
            candidate["fusion_score"] = score
            candidate["verified"] = self.retriever.verify(candidate, "", part_number)
            final_results.append(candidate)

        return final_results

    @staticmethod
    def _pad_scores(scores, target_len):
        if len(scores) >= target_len:
            return scores
        return scores + [0.0] * (target_len - len(scores))

    @staticmethod
    def _noop_cross_encoder(query, candidates):
        """Placeholder cross-encoder that returns zero scores."""
        return [0.0 for _ in candidates]

    def _fetch_attrs_for_hits(self, hits):
        ids = [str(hit.id) for hit in hits]
        if not ids:
            return {}
        rows = self.index.fetch_attributes(
            ids,
            output_fields=["pk", "model_id", "maker", "part_number", "category", "ocr_text"],
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
