#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/mac/project/Smart_vision")
OUT_DIR = ROOT / "data" / "datasets" / "unified_v1"


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _clean(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_key(value: str) -> str:
    value = _clean(value).lower()
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9가-힣]+", "", value)


def _valid_part_number(value: str) -> bool:
    normalized = _normalize_key(value)
    return bool(normalized and normalized not in {"", "none", "null", "없음", "미상"})


def _list_images(image_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = sorted(str(path.resolve()) for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in exts)
    return files


def _build_group_key(domain: str, maker: str, part_number: str, title: str, product_info: str) -> str:
    maker_key = _normalize_key(maker)
    if _valid_part_number(part_number):
        return f"{domain}|{maker_key}|{_normalize_key(part_number)}"
    fallback = _normalize_key(product_info) or _normalize_key(title)
    return f"{domain}|{maker_key}|{fallback}"


def _convert_gparts(dataset_name: str, category_code: str, subcategory: str) -> list[dict]:
    base_dir = ROOT / "data" / "raw" / dataset_name
    image_root = base_dir / "images"
    items = []
    for idx, record in enumerate(_read_jsonl(base_dir / "gparts_items.jsonl"), start=1):
        image_dirs = sorted(p for p in image_root.iterdir() if p.is_dir())
        # rely on meta files to reconstruct stable order
        break
    by_meta = sorted(p for p in image_root.iterdir() if p.is_dir() and (p / "meta.json").exists())
    for idx, image_dir in enumerate(by_meta, start=1):
        record = json.loads((image_dir / "meta.json").read_text(encoding="utf-8"))
        image_paths = _list_images(image_dir)
        if not image_paths:
            continue
        domain = "auto_part"
        item = {
            "item_id": f"auto_{dataset_name}_{idx:06d}",
            "source_dataset": dataset_name,
            "domain": domain,
            "subcategory": subcategory,
            "group_key": _build_group_key(
                domain,
                _clean(record.get("maker")),
                _clean(record.get("part_number")),
                _clean(record.get("title")),
                _clean(record.get("product_info")),
            ),
            "title": _clean(record.get("title")),
            "description": _clean(record.get("description")),
            "maker": _clean(record.get("maker")),
            "vehicle_name": _clean(record.get("vehicle_name")),
            "year": _clean(record.get("year")),
            "part_number": _clean(record.get("part_number")),
            "product_info": _clean(record.get("product_info")),
            "price_value": record.get("price_value"),
            "image_dir": str(image_dir.resolve()),
            "image_paths": image_paths,
            "image_count": len(image_paths),
            "source_category_code": category_code,
        }
        items.append(item)
    return items


def _convert_1090481() -> list[dict]:
    base_dir = ROOT / "data" / "1090481"
    items = []
    for idx, image_dir in enumerate(sorted(p for p in base_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()), start=1):
        record = json.loads((image_dir / "meta.json").read_text(encoding="utf-8"))
        image_paths = _list_images(image_dir)
        if not image_paths:
            continue
        domain = "semiconductor_equipment_part"
        item = {
            "item_id": f"semi_1090481_{idx:06d}",
            "source_dataset": "semiconductor_equipment_1090481",
            "domain": domain,
            "subcategory": "equipment_component",
            "group_key": _build_group_key(
                domain,
                _clean(record.get("maker")),
                _clean(record.get("part_number")),
                _clean(record.get("title")),
                _clean(record.get("product_info")),
            ),
            "title": _clean(record.get("title")),
            "description": _clean(record.get("description")),
            "maker": _clean(record.get("maker")),
            "vehicle_name": _clean(record.get("vehicle_name")),
            "year": _clean(record.get("year")),
            "part_number": _clean(record.get("part_number")),
            "product_info": _clean(record.get("product_info")),
            "price_value": record.get("price_value"),
            "image_dir": str(image_dir.resolve()),
            "image_paths": image_paths,
            "image_count": len(image_paths),
            "source_category_code": "",
        }
        items.append(item)
    return items


def _assign_split(items: list[dict], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    groups_by_domain: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups_by_domain[item["domain"]][item["group_key"]].append(item)

    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []

    for domain, grouped in groups_by_domain.items():
        groups = list(grouped.values())
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        target_test = max(1, round(total * test_ratio))
        running = 0
        for group in groups:
            bucket = test if running < target_test else train
            for item in group:
                cloned = dict(item)
                cloned["split"] = "test" if bucket is test else "train"
                bucket.append(cloned)
            if bucket is test:
                running += len(group)
    return train, test


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    items.extend(_convert_gparts("gparts_cat40", "40", "interior_electrical_trim"))
    items.extend(_convert_gparts("gparts_cat50", "50", "engine_chassis"))
    items.extend(_convert_1090481())

    train, test = _assign_split(items)
    unified = sorted(train + test, key=lambda x: x["item_id"])
    train = sorted(train, key=lambda x: x["item_id"])
    test = sorted(test, key=lambda x: x["item_id"])

    _write_jsonl(OUT_DIR / "unified_all.jsonl", unified)
    _write_jsonl(OUT_DIR / "train.jsonl", train)
    _write_jsonl(OUT_DIR / "test.jsonl", test)

    source_names = sorted({item["source_dataset"] for item in unified})
    domain_names = sorted({item["domain"] for item in unified})
    summary = {
        "total_items": len(unified),
        "train_items": len(train),
        "test_items": len(test),
        "by_source_dataset": {name: sum(1 for item in unified if item["source_dataset"] == name) for name in source_names},
        "by_domain": {name: sum(1 for item in unified if item["domain"] == name) for name in domain_names},
        "train_by_domain": {name: sum(1 for item in train if item["domain"] == name) for name in domain_names},
        "test_by_domain": {name: sum(1 for item in test if item["domain"] == name) for name in domain_names},
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
