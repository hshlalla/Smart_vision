"""
Metadata Normalizer

Utility for harmonizing maker, part-number, and category fields prior to indexing.
"""

from __future__ import annotations

import re
from typing import Dict


class MetadataNormalizer:
    """Normalize free-form metadata values into consistent canonical forms."""

    def normalize(self, metadata: Dict[str, str]) -> Dict[str, str]:
        normalized = {
            "maker": self._normalize_maker(metadata.get("maker", "")),
            "part_number": self._normalize_part_number(metadata.get("part_number", "")),
            "category": metadata.get("category", "").strip().upper(),
        }
        if "model_id" in metadata:
            normalized["model_id"] = str(metadata.get("model_id", "")).strip()
        if "pk" in metadata:
            normalized["pk"] = str(metadata.get("pk", "")).strip()
        return normalized

    @staticmethod
    def _normalize_maker(value: str) -> str:
        return value.strip().title()

    @staticmethod
    def _normalize_part_number(value: str) -> str:
        alnum = re.sub(r"[^0-9A-Za-z]", "", value.upper())
        return alnum


__all__ = ["MetadataNormalizer"]
