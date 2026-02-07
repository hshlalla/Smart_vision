"""
Preprocessing & Embedding Layer

Orchestrates the multi-modal feature extraction pipeline:
    - Vision encoder (BGE-VL) for image embeddings
    - OCR engine (PaddleOCR VL) for text transcripts
    - Text encoder (BGE-M3) for dense text vectors
    - Metadata normalization for maker, part number, and category fields
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class NormalizedRecord:
    """Structured output of the preprocessing layer."""

    image_vector: Tensor
    ocr_vector: Tensor
    caption_vector: Tensor
    metadata: Dict[str, str]
    ocr_tokens: List[str]
    ocr_text: str
    caption_text: str
    text_corpus: str


class PreprocessingPipeline:
    """Runs the multi-stage preprocessing stack for each captured item."""

    def __init__(
        self,
        vision_encoder,
        ocr_engine,
        text_encoder,
        metadata_normalizer,
        captioner=None,
    ) -> None:
        self._vision_encoder = vision_encoder
        self._ocr_engine = ocr_engine
        self._text_encoder = text_encoder
        self._metadata_normalizer = metadata_normalizer
        self._captioner = captioner

    def __call__(self, image_path: str, metadata: Dict[str, str]) -> NormalizedRecord:
        logger.info(
            "Preprocessing start: image=%s metadata_keys=%s",
            image_path,
            sorted(metadata.keys()),
        )
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        ocr_output = self._ocr_engine.extract(image_path)
        ocr_duration = time.perf_counter() - stage_start
        logger.info(
            "OCR extraction complete: tokens=%d duration=%.2fs",
            len(ocr_output.tokens),
            ocr_duration,
        )
        tokens = [
            token.text if hasattr(token, "text") else str(token)
            for token in ocr_output.tokens
        ]
        ocr_text = " ".join(tokens).strip()

        stage_start = time.perf_counter()
        image_vector = self._vision_encoder.encode(image_path)
        vision_duration = time.perf_counter() - stage_start
        logger.info("Image embedding generated: dim=%d duration=%.2fs", image_vector.shape[-1], vision_duration)

        stage_start = time.perf_counter()
        normalized_metadata = self._metadata_normalizer.normalize(metadata)
        metadata_duration = time.perf_counter() - stage_start
        logger.info("Metadata normalized: %s duration=%.2fs", normalized_metadata, metadata_duration)

        caption_text: str = ""
        if self._captioner is not None:
            try:
                stage_start = time.perf_counter()
                caption_text = self._captioner.generate(image_path)
                caption_duration = time.perf_counter() - stage_start
                if caption_text:
                    logger.info(
                        "Caption generated: path=%s chars=%d preview=%s duration=%.2fs",
                        image_path,
                        len(caption_text),
                        caption_text[:120].replace("\n", " "),
                        caption_duration,
                    )
                else:
                    logger.info(
                        "Caption generation returned empty result for %s duration=%.2fs",
                        image_path,
                        caption_duration,
                    )
            except Exception:  # pragma: no cover - caption fallback
                logger.exception("Caption generation failed for %s", image_path)
                caption_text = ""

        metadata_phrases = []
        for key in ("model_id", "maker", "part_number", "category"):
            value = normalized_metadata.get(key)
            if value:
                label = key.replace("_", " ")
                metadata_phrases.append(f"{label}: {value}")

        combined_parts = metadata_phrases[:]
        if caption_text:
            combined_parts.append(caption_text)
        if ocr_text:
            combined_parts.append(ocr_text)
        text_corpus = ". ".join(part for part in combined_parts if part).strip()
        text_corpus = text_corpus or " "

        if ocr_text:
            stage_start = time.perf_counter()
            ocr_vector = self._text_encoder.encode_document(ocr_text)
            ocr_vector_duration = time.perf_counter() - stage_start
            logger.info("OCR text encoded: dim=%d duration=%.2fs", ocr_vector.shape[-1], ocr_vector_duration)
        else:
            ocr_vector = torch.zeros(self._text_encoder.embedding_dim)
            logger.info("OCR text empty; using zero vector dim=%d", ocr_vector.shape[-1])

        if caption_text:
            stage_start = time.perf_counter()
            caption_vector = self._text_encoder.encode_document(caption_text)
            caption_vector_duration = time.perf_counter() - stage_start
            logger.info("Caption text encoded: dim=%d duration=%.2fs", caption_vector.shape[-1], caption_vector_duration)
        else:
            caption_vector = torch.zeros(self._text_encoder.embedding_dim)
            logger.info("Caption text empty; using zero vector dim=%d", caption_vector.shape[-1])

        total_duration = time.perf_counter() - total_start
        logger.info(
            "Preprocessing complete: image=%s ocr_tokens=%d caption_chars=%d total_duration=%.2fs",
            image_path,
            len(tokens),
            len(caption_text),
            total_duration,
        )
        return NormalizedRecord(
            image_vector=image_vector,
            ocr_vector=ocr_vector,
            caption_vector=caption_vector,
            metadata=normalized_metadata,
            ocr_tokens=tokens,
            ocr_text=ocr_text,
            caption_text=caption_text,
            text_corpus=text_corpus,
        )


__all__ = ["NormalizedRecord", "PreprocessingPipeline"]
