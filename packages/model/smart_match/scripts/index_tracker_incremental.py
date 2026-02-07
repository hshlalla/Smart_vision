"""
Incremental Tracker Dataset Ingestion
=====================================

Scans the tracker metadata CSV and the images directory to ingest only new
models or images into Milvus. Existing models are left untouched unless new
image files (determined via SHA-256 hash comparison with stored copies) are
found.

Example
-------
    python -m smart_match.scripts.index_tracker_incremental \\
        --images-root data/images \\
        --dataset data/tracker_subset.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from smart_match.hybrid_search_pipeline.data_collection.tracker_dataset import (
    TrackerDataset,
)
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
    HybridSearchOrchestrator,
    MilvusConnectionConfig,
)

logger = logging.getLogger(__name__)


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_image_hashes(orchestrator: HybridSearchOrchestrator, model_id: str) -> Set[str]:
    rows = orchestrator.index.query_attributes_by_model(model_id, output_fields=["image_path"])
    hashes: Set[str] = set()
    for row in rows:
        image_path = row.get("image_path")
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            logger.debug("Stored image path missing for model_id=%s: %s", model_id, path)
            continue
        try:
            hashes.add(_hash_file(path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to hash stored image for %s (%s): %s", model_id, path, exc)
    return hashes


def _discover_new_images(image_dir: Path, existing_hashes: Set[str]) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str]] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        image_hash = _hash_file(path)
        if image_hash in existing_hashes:
            continue
        candidates.append((path, image_hash))
    return candidates


def _build_metadata(record) -> Dict[str, str]:
    return {
        "model_id": record.model_id,
        "maker": record.std_maker_name,
        "category": record.category_code,
        "part_number": record.std_model_name or record.non_std_model_name or "",
        "description": record.model_name,
        "model_name": record.model_name,
    }


def _iter_model_directories(images_root: Path) -> Iterable[Path]:
    for item in sorted(images_root.iterdir()):
        if item.is_dir():
            yield item


def run_incremental_indexing(
    *,
    dataset_path: Path,
    images_root: Path,
    milvus_uri: str,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    orchestrator = HybridSearchOrchestrator(
    # flake8: noqa: E121 - keep keyword args aligned
        milvus=MilvusConnectionConfig(uri=milvus_uri),
        tracker_dataset_path=dataset_path,
    )
    dataset = TrackerDataset.from_csv(dataset_path)

    summary: Dict[str, List[str]] = {"indexed": [], "skipped": [], "errors": []}

    for model_dir in _iter_model_directories(images_root):
        model_id = model_dir.name
        record = dataset.get(model_id)
        if record is None:
            logger.warning("Skipping %s: not present in tracker dataset.", model_id)
            summary["skipped"].append(f"{model_id} (missing metadata)")
            continue

        metadata = _build_metadata(record)
        try:
            existing_model = orchestrator.index.get_model(model_id, output_fields=["pk"])
        except Exception as exc:
            logger.error("Failed to query existing model %s: %s", model_id, exc)
            summary["errors"].append(f"{model_id}: query failed ({exc})")
            if dry_run:
                continue
            raise

        if not existing_model:
            logger.info("Model %s not found in Milvus; indexing all images.", model_id)
            if dry_run:
                summary["indexed"].append(f"{model_id} (dry-run)")
                continue
            try:
                orchestrator.index_tracker_model(
                    model_id,
                    images_root=images_root,
                    dataset_path=dataset_path,
                    halt_on_error=True,
                )
                summary["indexed"].append(model_id)
            except Exception as exc:
                logger.exception("Bulk indexing failed for %s", model_id)
                summary["errors"].append(f"{model_id}: {exc}")
            continue

        existing_hashes = _existing_image_hashes(orchestrator, model_id)
        new_images = _discover_new_images(model_dir, existing_hashes)
        if not new_images:
            logger.info("No new images detected for %s; skipping.", model_id)
            summary["skipped"].append(model_id)
            continue

        logger.info("Indexing %d new images for %s.", len(new_images), model_id)
        if dry_run:
            summary["indexed"].append(f"{model_id} (+{len(new_images)} images, dry-run)")
            continue

        try:
            orchestrator.index_model_metadata(model_id, metadata)
            for image_path, _hash in new_images:
                orchestrator.preprocess_and_index(image_path, metadata)
            summary["indexed"].append(f"{model_id} (+{len(new_images)} images)")
        except Exception as exc:
            logger.exception("Incremental indexing failed for %s", model_id)
            summary["errors"].append(f"{model_id}: {exc}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally ingest tracker dataset images into Milvus.")
    parser.add_argument("--dataset", type=Path, default=Path("data/tracker_subset.csv"), help="Path to tracker CSV file.")
    parser.add_argument("--images-root", type=Path, default=Path("data/images"), help="Root directory containing model folders.")
    parser.add_argument("--milvus-uri", default="tcp://standalone:19530", help="Milvus connection URI.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying Milvus.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    summary = run_incremental_indexing(
        dataset_path=args.dataset,
        images_root=args.images_root,
        milvus_uri=args.milvus_uri,
        dry_run=args.dry_run,
    )

    logger.info("Indexed: %s", summary.get("indexed"))
    logger.info("Skipped: %s", summary.get("skipped"))
    if summary.get("errors"):
        logger.error("Errors: %s", summary["errors"])


if __name__ == "__main__":
    main()
