from __future__ import annotations

from smart_match.hybrid_search_pipeline.search.ranking_utils import (
    compute_exact_field_boost,
    passes_min_score,
    tokenize_text,
)


def test_tokenize_text_supports_korean():
    tokens = tokenize_text("홍수훈 samsung abc123")
    assert "홍수훈" in tokens
    assert "samsung" in tokens
    assert "abc123" in tokens


def test_exact_field_boost_prefers_exact_korean_match():
    exact = compute_exact_field_boost(
        "홍수훈",
        model_id="a000001",
        maker="",
        part_number="",
        description="홍수훈",
    )
    partial = compute_exact_field_boost(
        "홍수훈",
        model_id="a000002",
        maker="",
        part_number="",
        description="홍수훈 장비",
    )
    none = compute_exact_field_boost(
        "홍수훈",
        model_id="a000003",
        maker="Applied",
        part_number="ABC123",
        description="Etch tool",
    )

    assert exact > partial > none


def test_min_score_filter_drops_weak_non_lexical_candidates():
    assert not passes_min_score(
        score=0.12,
        lexical_hit=False,
        exact_field_boost=0.0,
    )
    assert passes_min_score(
        score=0.12,
        lexical_hit=True,
        exact_field_boost=0.0,
    )
    assert passes_min_score(
        score=0.12,
        lexical_hit=False,
        exact_field_boost=0.55,
    )
