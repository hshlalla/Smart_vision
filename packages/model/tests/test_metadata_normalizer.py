from __future__ import annotations

from smart_match.hybrid_search_pipeline.preprocessing.metadata_normalizer import (
    MetadataNormalizer,
)


def test_metadata_normalizer_normalizes_known_fields():
    normalizer = MetadataNormalizer()
    data = {
        "maker": "  samsung electronics ",
        "part_number": " ab-12_34 ",
        "category": " etch ",
        "model_id": " a000001 ",
        "description": "  desc ",
    }

    out = normalizer.normalize(data)

    assert out["maker"] == "Samsung Electronics"
    assert out["part_number"] == "AB1234"
    assert out["category"] == "ETCH"
    assert out["model_id"] == "a000001"
    assert out["description"] == "desc"


def test_metadata_normalizer_skips_none_values():
    normalizer = MetadataNormalizer()
    out = normalizer.normalize({"maker": None, "status": " active "})

    assert "maker" not in out
    assert out["status"] == "active"
