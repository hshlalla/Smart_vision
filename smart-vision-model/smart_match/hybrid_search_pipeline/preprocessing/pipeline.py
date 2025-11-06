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
from typing import Dict, List, Optional

import torch

from torch import Tensor


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
        ocr_output = self._ocr_engine.extract(image_path)
        tokens = [
            token.text if hasattr(token, "text") else str(token)
            for token in ocr_output.tokens
        ]
        ocr_text = " ".join(tokens).strip()
        image_vector = self._vision_encoder.encode(image_path)
        normalized_metadata = self._metadata_normalizer.normalize(metadata)

        caption_text: str = ""
        if self._captioner is not None:
            try:
                caption_text = self._captioner.generate(image_path)
            except Exception:  # pragma: no cover - caption fallback
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
            ocr_vector = self._text_encoder.encode_document(ocr_text)
        else:
            ocr_vector = torch.zeros(self._text_encoder.embedding_dim)

        if caption_text:
            caption_vector = self._text_encoder.encode_document(caption_text)
        else:
            caption_vector = torch.zeros(self._text_encoder.embedding_dim)

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
