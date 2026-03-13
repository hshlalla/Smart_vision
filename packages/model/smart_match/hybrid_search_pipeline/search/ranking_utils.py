from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

MIN_RESULT_SCORE = 0.18


def tokenize_text(text: str) -> List[str]:
    value = (text or "").lower()
    return re.findall(r"[a-z0-9가-힣]+(?:[-_][a-z0-9가-힣]+)?", value)


def compute_lexical_score(query_text: str, haystacks: Sequence[str]) -> float:
    query_tokens = tokenize_text(query_text)
    if not query_tokens:
        return 0.0
    q_count = Counter(query_tokens)
    joined_haystack = " ".join((h or "").lower() for h in haystacks)
    if not joined_haystack.strip():
        return 0.0

    doc_tokens = tokenize_text(joined_haystack)
    d_count = Counter(doc_tokens)
    if not d_count:
        return 0.0

    overlap = sum(min(freq, d_count.get(tok, 0)) for tok, freq in q_count.items())
    overlap_ratio = overlap / max(1.0, float(sum(q_count.values())))
    tf_score = 0.0
    for tok, qf in q_count.items():
        df = d_count.get(tok, 0)
        if df <= 0:
            continue
        tf_score += (1.0 + math.log1p(df)) * qf
    tf_norm = tf_score / max(1.0, float(len(doc_tokens)))
    substring_bonus = 0.35 if query_text.strip().lower() in joined_haystack else 0.0
    return min(1.0, overlap_ratio * 0.7 + tf_norm * 0.3 + substring_bonus)


def compute_exact_field_boost(
    query_text: str,
    *,
    model_id: str,
    maker: str,
    part_number: str,
    description: str,
) -> float:
    query_norm = (query_text or "").strip().lower()
    if not query_norm:
        return 0.0

    exact_fields = [model_id, maker, part_number, description]
    exact_fields_norm = [str(value or "").strip().lower() for value in exact_fields if str(value or "").strip()]
    if any(value == query_norm for value in exact_fields_norm):
        return 0.55
    if any(query_norm in value for value in exact_fields_norm):
        return 0.30
    return 0.0


def passes_min_score(*, score: float, lexical_hit: bool, exact_field_boost: float) -> bool:
    if exact_field_boost > 0.0 or lexical_hit:
        return True
    return score >= MIN_RESULT_SCORE
