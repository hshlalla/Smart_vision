#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions against query/index manifests.")
    parser.add_argument("--queries", required=True, help="Path to query_manifest.jsonl")
    parser.add_argument("--index", required=True, help="Path to index_manifest.jsonl")
    parser.add_argument("--predictions", required=True, help="Path to predictions jsonl")
    args = parser.parse_args()

    queries = {row["query_id"]: row for row in _read_jsonl(Path(args.queries))}
    index = {row["item_id"]: row for row in _read_jsonl(Path(args.index))}
    predictions = _read_jsonl(Path(args.predictions))

    total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    recall_at_5 = 0.0
    mrr_total = 0.0
    exact_group_hit = 0

    for pred in predictions:
        query_id = pred["query_id"]
        query = queries.get(query_id)
        if query is None:
            continue
        expected_group = query["expected_group_key"]
        ranked = pred.get("predictions", [])
        total += 1

        first_group = None
        found_rank = None
        top5_match_count = 0

        for rank, item in enumerate(ranked, start=1):
            candidate = index.get(item.get("item_id"))
            if candidate is None:
                continue
            group_key = candidate["group_key"]
            if rank == 1:
                first_group = group_key
            if group_key == expected_group and found_rank is None:
                found_rank = rank
            if rank <= 5 and group_key == expected_group:
                top5_match_count += 1

        if first_group == expected_group:
            hit_at_1 += 1
            exact_group_hit += 1
        if found_rank is not None and found_rank <= 5:
            hit_at_5 += 1
            recall_at_5 += 1.0
        if found_rank is not None:
            mrr_total += 1.0 / found_rank

    metrics = {
        "queries_evaluated": total,
        "accuracy_at_1": (hit_at_1 / total) if total else 0.0,
        "accuracy_at_5": (hit_at_5 / total) if total else 0.0,
        "recall_at_5": (recall_at_5 / total) if total else 0.0,
        "mrr": (mrr_total / total) if total else 0.0,
        "exact_group_hit_rate": (exact_group_hit / total) if total else 0.0,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
