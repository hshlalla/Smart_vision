"""
Search & Re-ranking Layer

Implements the hybrid retrieval pipeline that fuses image and text vectors,
applies cross-encoder re-ranking, and performs final verification against
OCR alignment and part-number matching thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


class CrossEncoder(Protocol):
    """Interface for cross-encoders such as Qwen-VL, BLIP2, or InternVL."""

    def score(self, query: dict, candidates: List[dict]) -> List[float]:
        ...


@dataclass
class FusionWeights:
    """Weights for combining image and text cosine similarities."""

    alpha: float
    beta: float


class HybridFusionRetriever:
    """Combine dense retrieval results and perform cross-encoder re-ranking."""

    def __init__(self, cross_encoder: CrossEncoder, weights: FusionWeights) -> None:
        self._cross_encoder = cross_encoder
        self._weights = weights

    def fuse_scores(self, img_scores, txt_scores) -> np.ndarray:
        """Compute fusion score using alpha·cos(img) + beta·cos(txt)."""
        img_scores = np.asarray(img_scores, dtype=np.float32)
        txt_scores = np.asarray(txt_scores, dtype=np.float32)
        return self._weights.alpha * img_scores + self._weights.beta * txt_scores

    def rerank(self, query: dict, candidates: List[dict]) -> List[dict]:
        """Rerank candidates using the configured cross encoder."""
        scores = self._cross_encoder.score(query, candidates)
        sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in sorted_pairs]

    def verify(self, candidate: dict, ocr_text: str, part_number: str, threshold: float = 0.75) -> bool:
        """Basic verification that checks OCR similarity and part number match."""
        ocr_similarity = candidate.get("ocr_similarity", 0.0)
        pn_match = candidate.get("part_number", "") == part_number
        return ocr_similarity >= threshold and pn_match


__all__ = ["CrossEncoder", "FusionWeights", "HybridFusionRetriever"]
