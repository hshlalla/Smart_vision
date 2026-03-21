#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ENV_FILE = REPO_ROOT / "apps" / "api" / ".env"
UNIFIED_DATASET = REPO_ROOT / "data" / "datasets" / "unified_v1" / "unified_all.jsonl"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"


def _load_env_file(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _build_text_query(row: dict[str, Any]) -> str:
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


def _sample_rows(rows: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if len(row.get("image_paths") or []) >= 2]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_domain[str(row.get("domain") or "unknown")].append(row)

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    domains = sorted(by_domain.keys())
    domain_targets: dict[str, int] = {}
    total = len(eligible)
    remaining = sample_size
    for idx, domain in enumerate(domains, start=1):
        count = len(by_domain[domain])
        if idx == len(domains):
            domain_targets[domain] = remaining
        else:
            target = max(1, round(sample_size * (count / total)))
            domain_targets[domain] = min(target, count)
            remaining -= domain_targets[domain]

    for domain in domains:
        candidates = list(by_domain[domain])
        rng.shuffle(candidates)
        picked.extend(sorted(candidates[: domain_targets[domain]], key=lambda row: str(row.get("item_id") or "")))

    return sorted(picked, key=lambda row: str(row.get("item_id") or ""))


def _build_image_holdout_sample(
    *,
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    max_index_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sampled_rows = _sample_rows(rows, sample_size=sample_size, seed=seed)
    index_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for row in sampled_rows:
        image_paths = list(row.get("image_paths") or [])
        query_image = image_paths[0]
        index_images = image_paths[1:]
        if max_index_images and max_index_images > 0:
            index_images = index_images[:max_index_images]
        index_row = dict(row)
        index_row["image_paths"] = index_images
        index_row["image_count"] = len(index_images)
        index_rows.append(index_row)
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
                "image_path": query_image,
                "image_paths": [query_image],
                "image_count": 1,
            }
        )

    summary = {
        "sample_size": len(sampled_rows),
        "domains": {
            domain: sum(1 for row in sampled_rows if str(row.get("domain") or "") == domain)
            for domain in sorted({str(row.get("domain") or "") for row in sampled_rows})
        },
        "protocol": "n-1 index images + 1 held-out query image per item",
        "max_index_images": max_index_images or 0,
    }
    return index_rows, query_rows, summary


def _build_text_light_variant(
    *,
    index_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create a text-light/image-dominant variant.

    - Indexed rows keep the same images and item identity but strip most textual
      metadata so the text channel contributes minimally.
    - Query rows keep the same held-out images but clear text_query and text
      helper fields, effectively making the query image-dominant.
    """

    stripped_index_rows: list[dict[str, Any]] = []
    for row in index_rows:
        item = dict(row)
        for key in (
            "maker",
            "part_number",
            "subcategory",
            "description",
            "product_info",
            "title",
            "vehicle_name",
            "year",
            "group_key",
        ):
            if key in item:
                item[key] = ""
        stripped_index_rows.append(item)

    stripped_query_rows: list[dict[str, Any]] = []
    for row in query_rows:
        item = dict(row)
        item["text_query"] = ""
        item["maker"] = ""
        item["part_number"] = ""
        item["product_info"] = ""
        stripped_query_rows.append(item)

    return stripped_index_rows, stripped_query_rows


def _drop_collections(names: list[str], milvus_uri: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "model"))
    from pymilvus import connections, utility  # type: ignore

    connections.connect(alias="default", uri=milvus_uri)
    for name in names:
        if utility.has_collection(name):
            utility.drop_collection(name)


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _collection_names(prefix: str) -> dict[str, str]:
    return {
        "image": f"{prefix}_image",
        "text": f"{prefix}_text",
        "attrs": f"{prefix}_attrs",
        "model": f"{prefix}_model",
        "caption": f"{prefix}_caption",
    }


def _extract_run_dir(stdout: str) -> str:
    try:
        start = stdout.rfind("{")
        payload = json.loads(stdout[start:].strip()) if start >= 0 else None
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload.get("run_dir"):
        return str(payload["run_dir"])
    raise RuntimeError("Failed to parse run_dir from experiment runner output.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sampled retrieval ablation on isolated Milvus collections.")
    parser.add_argument("--env-file", type=Path, default=API_ENV_FILE)
    parser.add_argument("--dataset", type=Path, default=UNIFIED_DATASET)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-index-images", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--latency-limit", type=int, default=30)
    parser.add_argument("--warm-passes", type=int, default=1)
    parser.add_argument("--scenario-count", type=int, default=8)
    parser.add_argument("--configs", default="c1,c2,c3", help="Comma-separated configs: c1,c2,c3,c4")
    args = parser.parse_args()

    loaded_env = _load_env_file(args.env_file)
    milvus_uri = os.environ.get("MILVUS_URI", "tcp://localhost:19530")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_DIR / f"{timestamp}_sampled_ablation"
    run_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(args.dataset)
    index_rows, query_rows, split_summary = _build_image_holdout_sample(
        rows=rows,
        sample_size=args.sample_size,
        seed=args.seed,
        max_index_images=args.max_index_images,
    )
    dataset_dir = run_root / "dataset"
    index_manifest = dataset_dir / "index_manifest.jsonl"
    query_manifest = dataset_dir / "query_manifest.jsonl"
    _write_jsonl(index_manifest, index_rows)
    _write_jsonl(query_manifest, query_rows)
    _write_json(dataset_dir / "summary.json", split_summary)
    c4_index_rows, c4_query_rows = _build_text_light_variant(index_rows=index_rows, query_rows=query_rows)
    c4_dataset_dir = run_root / "dataset_c4"
    c4_index_manifest = c4_dataset_dir / "index_manifest.jsonl"
    c4_query_manifest = c4_dataset_dir / "query_manifest.jsonl"
    _write_jsonl(c4_index_manifest, c4_index_rows)
    _write_jsonl(c4_query_manifest, c4_query_rows)
    _write_json(
        c4_dataset_dir / "summary.json",
        {
            **split_summary,
            "protocol_variant": "text-light / image-dominant",
        },
    )

    configs = [item.strip().lower() for item in args.configs.split(",") if item.strip()]
    config_defs = {
        "c1": {"enable_ocr": "0", "enable_reranker": "1", "label": "ocr_off_reranker_on"},
        "c2": {"enable_ocr": "1", "enable_reranker": "1", "label": "ocr_on_reranker_on"},
        "c3": {
            "enable_ocr": "0",
            "enable_reranker": "0",
            "label": "ocr_off_reranker_off",
            "reuse_from": "c1",
        },
        "c4": {"enable_ocr": "0", "enable_reranker": "1", "label": "ocr_off_text_light_reranker_on"},
    }
    aggregate: dict[str, Any] = {
        "run_root": str(run_root),
        "sample_summary": split_summary,
        "configs": {},
        "loaded_env_keys": sorted(loaded_env.keys()),
    }

    for name in configs:
        if name not in config_defs:
            raise ValueError(f"Unsupported config: {name}")
        config = config_defs[name]
        reuse_from = str(config.get("reuse_from") or "").strip().lower()
        if reuse_from:
            reused = aggregate["configs"].get(reuse_from)
            if not reused:
                raise RuntimeError(f"Config '{name}' requires prior config '{reuse_from}' to run first.")
            collections = dict(reused["collections"])
        else:
            prefix = f"exp_{timestamp}_{name}"
            collections = _collection_names(prefix)
            _drop_collections(list(collections.values()), milvus_uri)

        selected_index_manifest = index_manifest
        selected_query_manifest = query_manifest
        if name == "c4":
            selected_index_manifest = c4_index_manifest
            selected_query_manifest = c4_query_manifest

        env = os.environ.copy()
        env.update(
            {
                "MILVUS_URI": milvus_uri,
                "ENABLE_OCR": config["enable_ocr"],
                "ENABLE_OCR_INDEXING": config["enable_ocr"],
                "ENABLE_OCR_QUERY": config["enable_ocr"],
                "ENABLE_RERANKER": config["enable_reranker"],
                "CAPTIONER_BACKEND": "none",
            }
        )

        if not reuse_from:
            index_cmd = [
                str(VENV_PYTHON),
                "-m",
                "smart_match.scripts.index_unified_jsonl",
                "--dataset",
                str(selected_index_manifest),
                "--repo-root",
                str(REPO_ROOT),
                "--milvus-uri",
                milvus_uri,
                "--image-collection",
                collections["image"],
                "--text-collection",
                collections["text"],
                "--attrs-collection",
                collections["attrs"],
                "--model-collection",
                collections["model"],
                "--caption-collection",
                collections["caption"],
            ]
            rc, stdout, stderr = _run_command(index_cmd, env=env, cwd=REPO_ROOT)
            if rc != 0:
                _write_json(
                    run_root / f"{name}_failure.json",
                    {"stage": "index", "returncode": rc, "stdout": stdout[-4000:], "stderr": stderr[-4000:]},
                )
                raise RuntimeError(f"Indexing failed for {name}")

        suite_cmd = [
            str(VENV_PYTHON),
            str(REPO_ROOT / "experiments" / "run_current_index_suite.py"),
            "--env-file",
            str(args.env_file),
            "--queries",
            str(selected_query_manifest),
            "--index",
            str(selected_index_manifest),
            "--mode",
            "hybrid",
            "--top-k",
            str(args.top_k),
            "--latency-limit",
            str(min(args.latency_limit, len(selected_query_manifest.read_text(encoding="utf-8").splitlines()))),
            "--warm-passes",
            str(args.warm_passes),
            "--scenario-count",
            str(min(args.scenario_count, len(selected_query_manifest.read_text(encoding="utf-8").splitlines()))),
            "--captioner-backend",
            "none",
            "--enable-ocr-query",
            config["enable_ocr"],
            "--use-reranker",
            config["enable_reranker"],
            "--image-collection",
            collections["image"],
            "--text-collection",
            collections["text"],
            "--attrs-collection",
            collections["attrs"],
            "--model-collection",
            collections["model"],
            "--caption-collection",
            collections["caption"],
            "--skip-e5",
        ]
        rc, stdout, stderr = _run_command(suite_cmd, env=env, cwd=REPO_ROOT)
        if rc != 0:
            _write_json(
                run_root / f"{name}_failure.json",
                {"stage": "eval", "returncode": rc, "stdout": stdout[-4000:], "stderr": stderr[-4000:]},
            )
            raise RuntimeError(f"Evaluation failed for {name}")

        suite_run_dir = Path(_extract_run_dir(stdout))
        e1_metrics = json.loads((suite_run_dir / "e1_metrics.json").read_text(encoding="utf-8"))
        e3_summary = json.loads((suite_run_dir / "e3_summary.json").read_text(encoding="utf-8"))
        aggregate["configs"][name] = {
            "label": config["label"],
            "collections": collections,
            "reused_from": reuse_from or None,
            "suite_run_dir": str(suite_run_dir),
            "e1_metrics": e1_metrics,
            "e3_summary": e3_summary,
        }

    _write_json(run_root / "ablation_summary.json", aggregate)
    print(json.dumps({"run_root": str(run_root), "configs": configs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
