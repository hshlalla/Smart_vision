# Unified Dataset Schema And Split

## Goal

Combine the currently collected datasets into one experiment-ready manifest while
preserving source folders:

- `data/raw/gparts_cat40`
- `data/raw/gparts_cat50`
- `data/1090481`

The unified dataset should support:

- hybrid retrieval experiments
- grouped train/test split generation
- later domain-balanced evaluation

## Source Domains

- `gparts_cat40` -> `auto_part`
- `gparts_cat50` -> `auto_part`
- `1090481` -> `semiconductor_equipment_part`

## Unified Record Schema

Each item in the unified manifest follows this schema:

```json
{
  "item_id": "auto_gparts_cat40_000001",
  "source_dataset": "gparts_cat40",
  "domain": "auto_part",
  "subcategory": "interior_electrical_trim",
  "group_key": "auto_part|기아|92800-c50",
  "title": "실내 조명등 기아 / 더 뉴 쏘렌토 (2018년식)",
  "description": "기아 | 더 뉴 쏘렌토 | 2018년식 | 92800-C50 | ...",
  "maker": "기아",
  "vehicle_name": "더 뉴 쏘렌토",
  "year": "2018년식",
  "part_number": "92800-C50",
  "product_info": "03-44 더뉴쏘렌토 실내조명등 ...",
  "price_value": 44000,
  "image_dir": "/abs/path/to/images/<item>",
  "image_paths": ["/abs/path/to/images/<item>/1.jpg"],
  "image_count": 4
}
```

## Field Rules

- `item_id`: stable unified identifier, generated from source dataset + row order
- `source_dataset`: one of `gparts_cat40`, `gparts_cat50`, `semiconductor_equipment_1090481`
- `domain`: top-level experiment domain
- `subcategory`: source-level category group
- `group_key`: grouping key for leakage-safe split
- `image_dir`: absolute directory path for the item images
- `image_paths`: absolute file paths of all images belonging to the item
- `image_count`: actual image file count, not just metadata count

## Split Strategy

Random split is not acceptable because very similar items can leak across train
and test.

The split must satisfy:

1. Items with the same `part_number` stay in the same split.
2. If `part_number` is missing, group by normalized `maker + title/product_info`.
3. Images from the same item stay in the same split.
4. Domain balance should be preserved.

## Recommended Default Split

- `train`: about 80%
- `test`: about 20%

Apply this independently per domain:

- `auto_part`
- `semiconductor_equipment_part`

This gives a more stable evaluation when domains have different metadata styles.

## Output Files

Generated under `data/datasets/unified_v1`:

- `unified_all.jsonl`
- `train.jsonl`
- `test.jsonl`
- `split_summary.json`

## Folder Policy

Do not duplicate images for now.

Use manifest files that point to original source folders:

- less disk usage
- easier regeneration when metadata changes
- easier auditing of original source provenance

