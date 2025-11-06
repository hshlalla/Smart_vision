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
    gamma: float = 0.0


class HybridFusionRetriever:
    """Combine dense retrieval results and perform cross-encoder re-ranking."""

    def __init__(self, cross_encoder: CrossEncoder, weights: FusionWeights) -> None:
        self._cross_encoder = cross_encoder
        self._weights = weights

    @property
    def weights(self) -> FusionWeights:
        return self._weights

    def fuse_scores(self, img_scores, ocr_scores, caption_scores=None) -> np.ndarray:
        """Compute fusion score using alpha·image + beta·ocr (+ gamma·caption when provided)."""
        img_scores = np.asarray(img_scores, dtype=np.float32)
        ocr_scores = np.asarray(ocr_scores, dtype=np.float32)
        combined = self._weights.alpha * img_scores + self._weights.beta * ocr_scores
        weight_sum = self._weights.alpha + self._weights.beta
        if caption_scores is not None:
            caption_scores = np.asarray(caption_scores, dtype=np.float32)
            combined += self._weights.gamma * caption_scores
            weight_sum += self._weights.gamma
        if weight_sum <= 0.0:
            return combined
        return combined / weight_sum

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
