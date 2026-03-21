#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ENV_FILE = REPO_ROOT / "apps" / "api" / ".env"
UNIFIED_DATASET = REPO_ROOT / "data" / "datasets" / "unified_v1" / "unified_all.jsonl"
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


def _resolve_image_path(raw_path: str) -> Path | None:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    try:
        idx = parts.index("data")
    except ValueError:
        return None
    rewritten = REPO_ROOT.joinpath(*parts[idx:])
    if rewritten.exists():
        return rewritten.resolve()
    return None


def _sample_rows(rows: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("part_number") or "").strip() and list(row.get("image_paths") or [])
    ]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_domain[str(row.get("domain") or "unknown")].append(row)

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    domains = sorted(by_domain.keys())
    total = len(eligible)
    remaining = sample_size
    for idx, domain in enumerate(domains, start=1):
        candidates = list(by_domain[domain])
        rng.shuffle(candidates)
        if idx == len(domains):
            target = min(len(candidates), remaining)
        else:
            target = min(len(candidates), max(1, round(sample_size * (len(candidates) / total))))
            remaining -= target
        picked.extend(candidates[:target])
    return sorted(picked[:sample_size], key=lambda row: str(row.get("item_id") or ""))


def _normalize_part_number(value: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(value or "").upper())


def _normalize_maker(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _join_ocr_text(result: Any) -> str:
    return " ".join(
        part.strip()
        for part in [
            getattr(result, "structured_text", "") or "",
            getattr(result, "markdown_text", "") or "",
            getattr(result, "combined_text", "") or "",
            getattr(result, "spotting_text", "") or "",
        ]
        if part and str(part).strip()
    ).strip()


def _summarize_method(rows: list[dict[str, Any]], *, method_key: str) -> dict[str, Any]:
    total = 0
    part_exact = 0
    part_recall = 0
    maker_exact = 0
    maker_recall = 0
    cer_values: list[float] = []

    for row in rows:
        gt_part = str(row.get("ground_truth_part_number_norm") or "")
        gt_maker = str(row.get("ground_truth_maker_norm") or "")
        pred_part = str(row.get(f"{method_key}_part_number_norm") or "")
        pred_maker = str(row.get(f"{method_key}_maker_norm") or "")
        corpus = str(row.get(f"{method_key}_corpus_norm") or "")
        total += 1

        if gt_part and pred_part == gt_part:
            part_exact += 1
        if gt_part and gt_part in corpus:
            part_recall += 1
        if gt_maker and pred_maker == gt_maker:
            maker_exact += 1
        if gt_maker and gt_maker in corpus:
            maker_recall += 1
        if gt_part:
            cer_values.append(_levenshtein(gt_part, pred_part) / max(len(gt_part), 1))

    def _rate(value: int) -> float:
        return round((value / total) if total else 0.0, 4)

    return {
        "samples": total,
        "part_number_exact_rate": _rate(part_exact),
        "part_number_recall_rate": _rate(part_recall),
        "maker_exact_rate": _rate(maker_exact),
        "maker_recall_rate": _rate(maker_recall),
        "part_number_cer_mean": round(sum(cer_values) / len(cer_values), 4) if cer_values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sampled OCR/Qwen identifier extraction pilot benchmark.")
    parser.add_argument("--env-file", type=Path, default=API_ENV_FILE)
    parser.add_argument("--dataset", type=Path, default=UNIFIED_DATASET)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-item", type=int, default=4)
    parser.add_argument("--skip-qwen", action="store_true", help="Run OCR-only pilot without Qwen metadata extraction.")
    parser.add_argument(
        "--write-partial-every",
        type=int,
        default=1,
        help="Write partial rows/progress every N completed samples.",
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "model"))
    from smart_vision_api.services.hybrid import HybridSearchService  # type: ignore
    from smart_match.hybrid_search_pipeline.preprocessing.ocr.OCR import PaddleOCRVLPipeline  # type: ignore

    rows = _read_jsonl(args.dataset)
    sampled = _sample_rows(rows, sample_size=args.sample_size, seed=args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_ocr_pilot"
    run_dir.mkdir(parents=True, exist_ok=True)

    service = HybridSearchService()
    ocr_engine = PaddleOCRVLPipeline()

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    partial_every = max(1, int(args.write_partial_every))
    for row in sampled:
        candidate_paths = [
            path
            for raw_path in list(row.get("image_paths") or [])[: args.max_images_per_item]
            if (path := _resolve_image_path(str(raw_path))) is not None
        ]
        if not candidate_paths:
            continue

        scored_candidates: list[tuple[int, Path, str]] = []
        for image_path in candidate_paths:
            ocr_result = ocr_engine.extract(str(image_path))
            ocr_text = _join_ocr_text(ocr_result)
            label_score = service._compute_label_signal(  # type: ignore[attr-defined]
                ocr_text,
                [token.text.strip() for token in ocr_result.tokens if getattr(token, "text", "").strip()],
            )
            scored_candidates.append((label_score, image_path, ocr_text))

        scored_candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
        best_score, best_image, best_ocr_text = scored_candidates[0]

        if args.skip_qwen:
            qwen_only: dict[str, Any] = {}
            qwen_merged: dict[str, Any] = {}
        else:
            qwen_only = service._suggest_metadata_from_qwen([best_image])  # type: ignore[attr-defined]
            qwen_merged = service._suggest_metadata_from_qwen([best_image], label_ocr_text=best_ocr_text)  # type: ignore[attr-defined]

        paddle_part = service._infer_part_number_from_text(best_ocr_text)  # type: ignore[attr-defined]
        paddle_maker = service._infer_maker_from_text(best_ocr_text)  # type: ignore[attr-defined]

        gt_part = str(row.get("part_number") or "")
        gt_maker = str(row.get("maker") or "")

        result = {
            "item_id": row.get("item_id"),
            "domain": row.get("domain"),
            "selected_image_path": str(best_image),
            "selected_image_label_score": best_score,
            "ground_truth_part_number": gt_part,
            "ground_truth_part_number_norm": _normalize_part_number(gt_part),
            "ground_truth_maker": gt_maker,
            "ground_truth_maker_norm": _normalize_maker(gt_maker),
            "paddle_ocr_text": best_ocr_text,
            "paddle_part_number": paddle_part,
            "paddle_part_number_norm": _normalize_part_number(paddle_part),
            "paddle_maker": paddle_maker,
            "paddle_maker_norm": _normalize_maker(paddle_maker),
            "paddle_corpus_norm": _normalize_part_number(best_ocr_text) + " " + _normalize_maker(best_ocr_text),
            "qwen_part_number": str(qwen_only.get("part_number") or ""),
            "qwen_part_number_norm": _normalize_part_number(qwen_only.get("part_number")),
            "qwen_maker": str(qwen_only.get("maker") or ""),
            "qwen_maker_norm": _normalize_maker(qwen_only.get("maker")),
            "qwen_category": str(qwen_only.get("category") or ""),
            "qwen_description": str(qwen_only.get("description") or ""),
            "qwen_corpus_norm": " ".join(
                [
                    _normalize_part_number(qwen_only.get("part_number")),
                    _normalize_maker(qwen_only.get("maker")),
                    _normalize_part_number(qwen_only.get("description")),
                    _normalize_maker(qwen_only.get("description")),
                ]
            ).strip(),
            "merged_part_number": str(qwen_merged.get("part_number") or ""),
            "merged_part_number_norm": _normalize_part_number(qwen_merged.get("part_number")),
            "merged_maker": str(qwen_merged.get("maker") or ""),
            "merged_maker_norm": _normalize_maker(qwen_merged.get("maker")),
            "merged_category": str(qwen_merged.get("category") or ""),
            "merged_description": str(qwen_merged.get("description") or ""),
            "merged_corpus_norm": " ".join(
                [
                    _normalize_part_number(qwen_merged.get("part_number")),
                    _normalize_maker(qwen_merged.get("maker")),
                    _normalize_part_number(qwen_merged.get("description")),
                    _normalize_maker(qwen_merged.get("description")),
                    _normalize_part_number(best_ocr_text),
                    _normalize_maker(best_ocr_text),
                ]
            ).strip(),
        }
        results.append(result)
        if len(results) % partial_every == 0:
            _write_jsonl(run_dir / "e2_ocr_pilot_rows.partial.jsonl", results)
            _write_json(
                run_dir / "progress.json",
                {
                    "completed_samples": len(results),
                    "requested_samples": args.sample_size,
                    "skip_qwen": bool(args.skip_qwen),
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                },
            )

    summary = {
        "run_dir": str(run_dir),
        "protocol": "sampled heuristic OCR pilot; best OCR-signal image selected per item from up to 4 images",
        "sample_size_requested": args.sample_size,
        "sample_size_completed": len(results),
        "skip_qwen": bool(args.skip_qwen),
        "duration_sec": round(time.perf_counter() - started, 2),
        "methods": {
            "paddle": _summarize_method(results, method_key="paddle"),
            "qwen": _summarize_method(results, method_key="qwen"),
            "merged": _summarize_method(results, method_key="merged"),
        },
    }

    _write_jsonl(run_dir / "e2_ocr_pilot_rows.jsonl", results)
    _write_json(run_dir / "e2_ocr_pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
