#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "datasets" / "unified_v1"
OUT_DIR = BASE_DIR / "eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _build_text_query(row: dict) -> str:
    parts: list[str] = []
    for key in ("part_number", "maker", "product_info"):
        value = _clean(row.get(key))
        if value:
            parts.append(value)
    if not parts:
        value = _clean(row.get("title"))
        if value:
            parts.append(value)
    return " | ".join(parts)


def main() -> int:
    train = _read_jsonl(BASE_DIR / "train.jsonl")
    test = _read_jsonl(BASE_DIR / "test.jsonl")

    index_rows = []
    for row in train:
        index_rows.append(
            {
                "item_id": row["item_id"],
                "domain": row["domain"],
                "group_key": row["group_key"],
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "maker": row.get("maker", ""),
                "part_number": row.get("part_number", ""),
                "product_info": row.get("product_info", ""),
                "image_dir": row.get("image_dir", ""),
                "image_paths": row.get("image_paths", []),
                "image_count": row.get("image_count", 0),
            }
        )

    query_rows = []
    for row in test:
        image_paths = row.get("image_paths", [])
        query_rows.append(
            {
                "query_id": f"query_{row['item_id']}",
                "item_id": row["item_id"],
                "domain": row["domain"],
                "group_key": row["group_key"],
                "expected_group_key": row["group_key"],
                "text_query": _build_text_query(row),
                "maker": row.get("maker", ""),
                "part_number": row.get("part_number", ""),
                "product_info": row.get("product_info", ""),
                "image_path": image_paths[0] if image_paths else "",
                "image_paths": image_paths,
                "image_count": row.get("image_count", 0),
            }
        )

    _write_jsonl(OUT_DIR / "index_manifest.jsonl", index_rows)
    _write_jsonl(OUT_DIR / "query_manifest.jsonl", query_rows)

    summary = {
        "index_count": len(index_rows),
        "query_count": len(query_rows),
        "domains": {
            domain: {
                "index": sum(1 for row in index_rows if row["domain"] == domain),
                "query": sum(1 for row in query_rows if row["domain"] == domain),
            }
            for domain in sorted({row["domain"] for row in index_rows + query_rows})
        },
    }
    (OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
