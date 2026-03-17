"""
Unified JSONL Ingestion
=======================

Ingest items from the unified dataset JSONL into Milvus.

Expected row shape:
    {
        "item_id": "...",
        "maker": "...",
        "part_number": "...",
        "subcategory": "...",
        "description": "...",
        "product_info": "...",
        "price_value": 33000,
        "image_paths": ["/abs/path/a.jpg", ...],
        ...
    }

This script resolves stale absolute paths produced on another machine by
rewriting them against the current repository root when possible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
    HybridSearchOrchestrator,
    MilvusConnectionConfig,
)

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_image_path(raw_path: str, repo_root: Path) -> Path | None:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    parts = candidate.parts
    try:
        idx = parts.index("data")
    except ValueError:
        return None

    rewritten = repo_root.joinpath(*parts[idx:])
    if rewritten.exists():
        return rewritten.resolve()
    return None


def _resolve_image_paths(row: dict[str, Any], repo_root: Path) -> list[Path]:
    image_paths = row.get("image_paths") or []
    resolved: list[Path] = []
    for raw_path in image_paths:
        if not raw_path:
            continue
        path = _resolve_image_path(str(raw_path), repo_root)
        if path is not None:
            resolved.append(path)
    return resolved


def _build_metadata(row: dict[str, Any]) -> dict[str, str]:
    description = _clean_text(row.get("description") or row.get("title"))
    product_info = _clean_text(row.get("product_info"))
    price_value = row.get("price_value")

    metadata: dict[str, str] = {
        "model_id": _clean_text(row.get("item_id")),
        "maker": _clean_text(row.get("maker")),
        "part_number": _clean_text(row.get("part_number")),
        "category": _clean_text(row.get("subcategory") or row.get("domain") or row.get("source_category_code")),
        "description": description,
    }
    if product_info:
        metadata["web_text"] = product_info
    if price_value not in (None, ""):
        metadata["price_text"] = str(price_value)
    if _clean_text(row.get("generated_caption")):
        metadata["generated_caption"] = _clean_text(row.get("generated_caption"))
    return metadata


def _build_caption_map(row: dict[str, Any], repo_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    image_entries = row.get("generated_image_captions") or []
    if not isinstance(image_entries, list):
        return out
    for item in image_entries:
        if not isinstance(item, dict):
            continue
        raw_path = _clean_text(item.get("image_path"))
        caption = _clean_text(item.get("caption"))
        if not raw_path or not caption:
            continue
        resolved = _resolve_image_path(raw_path, repo_root)
        if resolved is None:
            continue
        out[str(resolved)] = caption
    return out


def _deterministic_image_pk(model_id: str, image_index: int) -> str:
    return f"{model_id}::img_{image_index:03d}"


def run_unified_ingestion(
    *,
    dataset_path: Path,
    repo_root: Path,
    milvus_uri: str,
    split: str | None = None,
    domain: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    orchestrator = None if dry_run else HybridSearchOrchestrator(
        milvus=MilvusConnectionConfig(uri=milvus_uri),
    )

    indexed: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    prepared = 0
    skipped_images = 0

    for row in _iter_jsonl(dataset_path):
        if split and _clean_text(row.get("split")) != split:
            continue
        if domain and _clean_text(row.get("domain")) != domain:
            continue

        metadata = _build_metadata(row)
        model_id = metadata.get("model_id", "")
        if not model_id:
            skipped.append("missing model_id")
            continue

        image_paths = _resolve_image_paths(row, repo_root)
        if not image_paths:
            skipped.append(f"{model_id} (no resolvable images)")
            continue
        caption_map = _build_caption_map(row, repo_root)

        prepared += 1
        if dry_run:
            indexed.append(f"{model_id} ({len(image_paths)} images, dry-run)")
        else:
            try:
                assert orchestrator is not None
                orchestrator.index_model_metadata(model_id, metadata)
                default_caption = _clean_text(row.get("generated_caption"))
                for image_index, path in enumerate(image_paths, start=1):
                    pk = _deterministic_image_pk(model_id, image_index)
                    existing = orchestrator.index.fetch_attributes([pk], output_fields=["pk"])
                    if existing:
                        skipped_images += 1
                        continue
                    per_image_metadata = dict(metadata)
                    per_image_metadata["pk"] = pk
                    per_image_metadata["caption_text"] = caption_map.get(str(path), default_caption)
                    orchestrator.preprocess_and_index(path, per_image_metadata)
                indexed.append(f"{model_id} ({len(image_paths)} images)")
            except Exception as exc:
                logger.exception("Unified ingestion failed for %s", model_id)
                errors.append(f"{model_id}: {exc}")

        if limit is not None and prepared >= limit:
            break

    return {
        "prepared": prepared,
        "indexed": indexed,
        "skipped": skipped,
        "skipped_images": skipped_images,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest unified JSONL dataset into Milvus.")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/unified_v1/unified_all.jsonl"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--milvus-uri", default="tcp://localhost:19530")
    parser.add_argument("--split", default="", help="Optional split filter, e.g. train or test")
    parser.add_argument("--domain", default="", help="Optional domain filter, e.g. auto_part")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of items to ingest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    summary = run_unified_ingestion(
        dataset_path=args.dataset,
        repo_root=args.repo_root.resolve(),
        milvus_uri=args.milvus_uri,
        split=(args.split or "").strip() or None,
        domain=(args.domain or "").strip() or None,
        limit=args.limit or None,
        dry_run=args.dry_run,
    )
    logger.info("Prepared: %d", summary["prepared"])
    logger.info("Indexed: %d", len(summary["indexed"]))
    logger.info("Skipped: %d", len(summary["skipped"]))
    logger.info("Skipped images already present: %d", summary["skipped_images"])
    if summary["errors"]:
        logger.error("Errors: %s", summary["errors"])


if __name__ == "__main__":
    main()
