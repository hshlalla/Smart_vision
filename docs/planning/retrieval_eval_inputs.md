# Retrieval Eval Inputs

## Purpose

Prepare experiment-ready manifests from `data/datasets/unified_v1` for retrieval
evaluation without touching raw source folders.

## Input

- `data/datasets/unified_v1/train.jsonl`
- `data/datasets/unified_v1/test.jsonl`

## Output

Generated under `data/datasets/unified_v1/eval_v1`:

- `index_manifest.jsonl`
- `query_manifest.jsonl`
- `eval_summary.json`

## Index Manifest

Each row corresponds to one indexed product item.

Fields:

- `item_id`
- `domain`
- `group_key`
- `title`
- `description`
- `maker`
- `part_number`
- `product_info`
- `image_dir`
- `image_paths`
- `image_count`

## Query Manifest

Each row corresponds to one test query item.

Fields:

- `query_id`
- `item_id`
- `domain`
- `group_key`
- `expected_group_key`
- `text_query`
- `maker`
- `part_number`
- `product_info`
- `image_path`
- `image_paths`
- `image_count`

## Query Text Rule

`text_query` is built from:

1. `part_number` if available
2. `maker`
3. `product_info`
4. fallback `title`

This gives a stable text query for text-only and hybrid experiments.

## Prediction Format

Metric script expects a JSONL file with rows like:

```json
{
  "query_id": "query_auto_gparts_cat40_000003",
  "predictions": [
    {"item_id": "auto_gparts_cat40_000001", "score": 0.88},
    {"item_id": "auto_gparts_cat40_000003", "score": 0.81}
  ]
}
```

## Metrics

- `Accuracy@1`
- `Accuracy@5`
- `Recall@5`
- `MRR`
- `exact_group_hit_rate`

Correctness is judged by `group_key`, not only exact `item_id`, so near-duplicate
same-group products are treated as correct retrieval targets.

