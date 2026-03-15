#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("/Users/mac/project/Smart_vision")
BASE_DIR = ROOT / "data" / "datasets" / "unified_v1"
ITEMS_DIR = BASE_DIR / "items"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _materialize_record(record: dict) -> dict:
    item_id = record["item_id"]
    domain = record["domain"]
    item_dir = ITEMS_DIR / domain / item_id
    images_dir = item_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    canonical_paths: list[str] = []
    for idx, image_path in enumerate(record.get("image_paths", []), start=1):
        src = Path(image_path)
        ext = src.suffix.lower() or ".jpg"
        dst = images_dir / f"{idx:03d}{ext}"
        _safe_copy(src, dst)
        canonical_paths.append(str(dst.absolute()))

    materialized = dict(record)
    materialized["image_dir"] = str(images_dir.absolute())
    materialized["image_paths"] = canonical_paths
    materialized["image_count"] = len(canonical_paths)

    meta_path = item_dir / "meta.json"
    meta_path.write_text(json.dumps(materialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return materialized


def main() -> int:
    if ITEMS_DIR.exists():
        shutil.rmtree(ITEMS_DIR)
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)

    unified = _read_jsonl(BASE_DIR / "unified_all.jsonl")
    train = _read_jsonl(BASE_DIR / "train.jsonl")
    test = _read_jsonl(BASE_DIR / "test.jsonl")

    by_id: dict[str, dict] = {}
    materialized_all: list[dict] = []
    for record in unified:
        item = _materialize_record(record)
        by_id[item["item_id"]] = item
        materialized_all.append(item)

    materialized_train = [by_id[record["item_id"]] for record in train]
    materialized_test = [by_id[record["item_id"]] for record in test]

    _write_jsonl(BASE_DIR / "unified_all.jsonl", materialized_all)
    _write_jsonl(BASE_DIR / "train.jsonl", materialized_train)
    _write_jsonl(BASE_DIR / "test.jsonl", materialized_test)

    summary = {
        "total_items": len(materialized_all),
        "train_items": len(materialized_train),
        "test_items": len(materialized_test),
        "items_root": str(ITEMS_DIR.resolve()),
        "domains": sorted({item["domain"] for item in materialized_all}),
    }
    (BASE_DIR / "materialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
