"""
Preprocessing & Embedding Layer

Orchestrates the multi-modal feature extraction pipeline:
    - Vision encoder (BGE-VL) for image embeddings
    - OCR engine (PaddleOCR VL) for text transcripts
    - Text encoder (BGE-M3) for dense text vectors
    - Metadata normalization for maker, part number, and category fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor


@dataclass
class NormalizedRecord:
    """Structured output of the preprocessing layer."""

    image_vector: Tensor
    text_vector: Tensor
    metadata: Dict[str, str]
    ocr_tokens: List[str]
    text_corpus: str


class PreprocessingPipeline:
    """Runs the multi-stage preprocessing stack for each captured item."""

    def __init__(self, vision_encoder, ocr_engine, text_encoder, metadata_normalizer) -> None:
        self._vision_encoder = vision_encoder
        self._ocr_engine = ocr_engine
        self._text_encoder = text_encoder
        self._metadata_normalizer = metadata_normalizer

    def __call__(self, image_path: str, metadata: Dict[str, str]) -> NormalizedRecord:
        ocr_output = self._ocr_engine.extract(image_path)
        tokens = [
            token.text if hasattr(token, "text") else str(token)
            for token in ocr_output.tokens
        ]
        ocr_text = " ".join(tokens).strip()
        image_vector = self._vision_encoder.encode(image_path)
        normalized_metadata = self._metadata_normalizer.normalize(metadata)

        metadata_phrases = []
        for key in ("model_id", "maker", "part_number", "category"):
            value = normalized_metadata.get(key)
            if value:
                label = key.replace("_", " ")
                metadata_phrases.append(f"{label}: {value}")

        combined_parts = metadata_phrases[:]
        if ocr_text:
            combined_parts.append(ocr_text)
        text_corpus = ". ".join(part for part in combined_parts if part).strip()
        text_corpus = text_corpus or " "

        text_vector = self._text_encoder.encode(text_corpus)

        return NormalizedRecord(
            image_vector=image_vector,
            text_vector=text_vector,
            metadata=normalized_metadata,
            ocr_tokens=tokens,
            text_corpus=text_corpus,
        )


__all__ = ["NormalizedRecord", "PreprocessingPipeline"]
