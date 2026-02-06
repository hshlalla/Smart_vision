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
        normalized = {}
        if "maker" in metadata and metadata["maker"] is not None:
            normalized["maker"] = self._normalize_maker(str(metadata.get("maker", "")))
        if "part_number" in metadata and metadata["part_number"] is not None:
            normalized["part_number"] = self._normalize_part_number(str(metadata.get("part_number", "")))
        if "category" in metadata and metadata["category"] is not None:
            normalized["category"] = str(metadata.get("category", "")).strip().upper()
        if "model_id" in metadata:
            normalized["model_id"] = str(metadata.get("model_id", "")).strip()
        if "pk" in metadata:
            normalized["pk"] = str(metadata.get("pk", "")).strip()
        if "description" in metadata and metadata["description"] is not None:
            normalized["description"] = str(metadata.get("description", "")).strip()
        if "status" in metadata and metadata["status"] is not None:
            normalized["status"] = str(metadata.get("status", "")).strip()
        if "web_text" in metadata and metadata["web_text"] is not None:
            normalized["web_text"] = str(metadata.get("web_text", "")).strip()
        if "price_text" in metadata and metadata["price_text"] is not None:
            normalized["price_text"] = str(metadata.get("price_text", "")).strip()
        return normalized

    @staticmethod
    def _normalize_maker(value: str) -> str:
        return value.strip().title()

    @staticmethod
    def _normalize_part_number(value: str) -> str:
        alnum = re.sub(r"[^0-9A-Za-z]", "", value.upper())
        return alnum


__all__ = ["MetadataNormalizer"]
