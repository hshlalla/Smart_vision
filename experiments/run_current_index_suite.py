#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ENV_FILE = REPO_ROOT / "apps" / "api" / ".env"
DEFAULT_INDEX_MANIFEST = REPO_ROOT / "data" / "datasets" / "unified_v1" / "unified_all.jsonl"
DEFAULT_QUERY_MANIFEST = REPO_ROOT / "data" / "datasets" / "unified_v1" / "eval_v1" / "query_manifest.jsonl"
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_image_path(raw_path: str, repo_root: Path) -> Path | None:
    candidate = Path(str(raw_path)).expanduser()
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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if upper == lower:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(_percentile(values, 0.50), 2),
        "p90": round(_percentile(values, 0.90), 2),
        "p95": round(_percentile(values, 0.95), 2),
    }


def _select_scenarios(query_rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)

    selected: list[dict[str, Any]] = []
    domains = sorted(by_domain.keys())
    while len(selected) < count:
        progressed = False
        for domain in domains:
            if not by_domain[domain]:
                continue
            selected.append(by_domain[domain].pop(0))
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _compute_metrics_for_predictions(
    *,
    queries: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_map = {row["query_id"]: row for row in queries}
    index_map = {row["item_id"]: row for row in index_rows}

    total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    recall_at_5 = 0.0
    mrr_total = 0.0
    exact_group_hit = 0

    error_rows: list[dict[str, Any]] = []

    for pred in predictions:
        query_id = str(pred.get("query_id") or "")
        query = query_map.get(query_id)
        if query is None:
            continue
        domain = str(query.get("domain") or "unknown")

        expected_group = str(query.get("expected_group_key") or "")
        ranked = pred.get("predictions") or []
        total += 1

        first_group = None
        found_rank = None
        top5_match_count = 0

        for rank, item in enumerate(ranked, start=1):
            candidate = index_map.get(str(item.get("item_id") or ""))
            if candidate is None:
                continue
            group_key = str(candidate.get("group_key") or "")
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

        if first_group != expected_group:
            if found_rank is None:
                failure_type = "missing_top5"
            elif found_rank <= 5:
                failure_type = "wrong_top1_but_correct_top5"
            else:
                failure_type = "found_below_top5"
            error_rows.append(
                {
                    "query_id": query_id,
                    "item_id": query.get("item_id"),
                    "domain": domain,
                    "expected_group_key": expected_group,
                    "first_group_key": first_group,
                    "found_rank": found_rank,
                    "failure_type": failure_type,
                    "top_prediction_id": (ranked[0].get("item_id") if ranked else ""),
                    "query_text": query.get("text_query", ""),
                }
            )

    metrics = {
        "queries_evaluated": total,
        "accuracy_at_1": round((hit_at_1 / total) if total else 0.0, 4),
        "accuracy_at_5": round((hit_at_5 / total) if total else 0.0, 4),
        "recall_at_5": round((recall_at_5 / total) if total else 0.0, 4),
        "mrr": round((mrr_total / total) if total else 0.0, 4),
        "exact_group_hit_rate": round((exact_group_hit / total) if total else 0.0, 4),
    }
    return metrics, error_rows


def _evaluate_predictions(
    *,
    queries: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metrics, error_rows = _compute_metrics_for_predictions(
        queries=queries,
        index_rows=index_rows,
        predictions=predictions,
    )

    by_domain: dict[str, dict[str, Any]] = {}
    predictions_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        predictions_by_domain[str(pred.get("domain") or "unknown")].append(pred)

    for domain, domain_predictions in sorted(predictions_by_domain.items()):
        domain_queries = [row for row in queries if str(row.get("domain") or "unknown") == domain]
        domain_metrics, _ = _compute_metrics_for_predictions(
            queries=domain_queries,
            index_rows=index_rows,
            predictions=domain_predictions,
        )
        by_domain[domain] = domain_metrics

    return metrics, by_domain, error_rows


def _build_orchestrator(
    *,
    image_collection: str,
    text_collection: str,
    attrs_collection: str,
    model_collection: str,
    caption_collection: str,
):
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "model"))
    from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (  # type: ignore
        HybridSearchOrchestrator,
        MilvusConnectionConfig,
    )

    return HybridSearchOrchestrator(
        milvus=MilvusConnectionConfig(uri=os.environ["MILVUS_URI"]),
        image_collection=image_collection,
        text_collection=text_collection,
        attrs_collection=attrs_collection,
        model_collection=model_collection,
        caption_collection=caption_collection,
        load_vector_collections=True,
        load_metadata_collections=True,
    )


def _run_query(
    orchestrator,
    *,
    row: dict[str, Any],
    mode: str,
    top_k: int,
    use_reranker: bool | None,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    query_text = row.get("text_query") if mode in {"text", "hybrid"} else None
    image_path = None
    image_source = row.get("image_path") if mode in {"image", "hybrid"} else None
    if image_source:
        image_path = _resolve_image_path(str(image_source), REPO_ROOT)

    if mode in {"image", "hybrid"} and image_path is None:
        return [], {"total": 0.0}, {"skipped": True, "reason": "missing_query_image"}

    search_kwargs = {
        "query_image": image_path,
        "query_text": query_text,
        "top_k": top_k,
        "part_number": None,
        "use_reranker": use_reranker,
        "return_timings": True,
    }
    started = time.perf_counter()
    results, timings = orchestrator.search(**search_kwargs)
    wall_total_ms = round((time.perf_counter() - started) * 1000, 2)
    timings = dict(timings or {})
    timings.setdefault("wall_total", wall_total_ms)
    meta = {
        "skipped": False,
        "query_image_path": str(image_path) if image_path else "",
    }
    return results, timings, meta


def _run_e0(
    *,
    orchestrator,
    query_rows: list[dict[str, Any]],
    mode: str,
    top_k: int,
    scenario_count: int,
    use_reranker: bool | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for row in _select_scenarios(query_rows, scenario_count):
        results, timings, meta = _run_query(
            orchestrator,
            row=row,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
        )
        predicted_ids = [str(item.get("model_id") or "") for item in results]
        expected_group = str(row.get("expected_group_key") or "")
        top1_model_id = predicted_ids[0] if predicted_ids else ""
        scenarios.append(
            {
                "query_id": row.get("query_id"),
                "item_id": row.get("item_id"),
                "domain": row.get("domain"),
                "expected_group_key": expected_group,
                "query_text": row.get("text_query", ""),
                "timings_ms": timings,
                "top1_model_id": top1_model_id,
                "top1_part_number": (results[0].get("part_number") if results else ""),
                "top1_score": (results[0].get("score") if results else 0.0),
                "returned_count": len(results),
                "has_image_results": bool(results and results[0].get("images")),
                "query_meta": meta,
            }
        )
    summary = {
        "scenario_count": len(scenarios),
        "successful_searches": sum(1 for row in scenarios if row.get("returned_count", 0) > 0),
        "top1_with_image": sum(1 for row in scenarios if row.get("has_image_results")),
    }
    return scenarios, summary


def _run_e1(
    *,
    orchestrator,
    query_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    mode: str,
    top_k: int,
    use_reranker: bool | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(query_rows, start=1):
        results, timings, meta = _run_query(
            orchestrator,
            row=row,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
        )
        prediction_items = [
            {
                "item_id": str(item.get("model_id") or ""),
                "score": round(float(item.get("score") or 0.0), 6),
                "part_number": str(item.get("part_number") or ""),
                "maker": str(item.get("maker") or ""),
                "category": str(item.get("category") or ""),
            }
            for item in results
        ]
        predictions.append(
            {
                "query_id": row.get("query_id"),
                "item_id": row.get("item_id"),
                "domain": row.get("domain"),
                "expected_group_key": row.get("expected_group_key"),
                "predictions": prediction_items,
                "query_meta": meta,
            }
        )
        timing_rows.append(
            {
                "query_id": row.get("query_id"),
                "domain": row.get("domain"),
                "mode": mode,
                "order": idx,
                "timings_ms": timings,
                "result_count": len(results),
            }
        )
    metrics, by_domain, error_rows = _evaluate_predictions(
        queries=query_rows,
        index_rows=index_rows,
        predictions=predictions,
    )
    return predictions, timing_rows, metrics, by_domain, error_rows


def _run_latency_pass(
    *,
    query_rows: list[dict[str, Any]],
    mode: str,
    top_k: int,
    use_reranker: bool | None,
    pass_label: str,
    image_collection: str,
    text_collection: str,
    attrs_collection: str,
    model_collection: str,
    caption_collection: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_started = time.perf_counter()
    orchestrator = _build_orchestrator(
        image_collection=image_collection,
        text_collection=text_collection,
        attrs_collection=attrs_collection,
        model_collection=model_collection,
        caption_collection=caption_collection,
    )
    init_ms = round((time.perf_counter() - init_started) * 1000, 2)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(query_rows, start=1):
        results, timings, meta = _run_query(
            orchestrator,
            row=row,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
        )
        rows.append(
            {
                "pass_label": pass_label,
                "query_id": row.get("query_id"),
                "domain": row.get("domain"),
                "order": idx,
                "timings_ms": timings,
                "result_count": len(results),
                "query_meta": meta,
            }
        )
    return rows, {"orchestrator_init_ms": init_ms}


def _run_e3(
    *,
    query_rows: list[dict[str, Any]],
    mode: str,
    top_k: int,
    use_reranker: bool | None,
    latency_limit: int,
    warm_passes: int,
    image_collection: str,
    text_collection: str,
    attrs_collection: str,
    model_collection: str,
    caption_collection: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_rows = query_rows[:latency_limit]
    cold_rows, cold_meta = _run_latency_pass(
        query_rows=sample_rows,
        mode=mode,
        top_k=top_k,
        use_reranker=use_reranker,
        pass_label="cold",
        image_collection=image_collection,
        text_collection=text_collection,
        attrs_collection=attrs_collection,
        model_collection=model_collection,
        caption_collection=caption_collection,
    )
    all_rows = list(cold_rows)
    warm_meta: list[dict[str, Any]] = []
    for index in range(1, warm_passes + 1):
        rows, meta = _run_latency_pass(
            query_rows=sample_rows,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
            pass_label=f"warm_{index}",
            image_collection=image_collection,
            text_collection=text_collection,
            attrs_collection=attrs_collection,
            model_collection=model_collection,
            caption_collection=caption_collection,
        )
        all_rows.extend(rows)
        warm_meta.append(meta)

    summary: dict[str, Any] = {
        "latency_sample_size": len(sample_rows),
        "cold": {"orchestrator_init_ms": cold_meta["orchestrator_init_ms"]},
        "warm": warm_meta,
        "operations": {},
    }
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        pass_label = str(row.get("pass_label") or "")
        condition = "cold" if pass_label == "cold" else "warm"
        timings = row.get("timings_ms") or {}
        for key, value in timings.items():
            try:
                by_condition[condition][key].append(float(value))
            except (TypeError, ValueError):
                continue

    for condition, ops in sorted(by_condition.items()):
        summary["operations"][condition] = {
            op: _summarize_series(values)
            for op, values in sorted(ops.items())
        }
    return all_rows, summary


def _run_e5(*, run_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    commands = [
        (
            "api_pytest",
            [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "apps/api/tests", "-q"],
            REPO_ROOT,
        ),
        (
            "model_pytest",
            [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "packages/model/tests", "-q"],
            REPO_ROOT,
        ),
        (
            "web_build",
            ["npm", "run", "build"],
            REPO_ROOT / "apps" / "web",
        ),
        (
            "evaluation_input_generation",
            [str(REPO_ROOT / ".venv" / "bin" / "python"), "data/scripts/prepare_retrieval_eval_inputs.py"],
            REPO_ROOT,
        ),
    ]
    for label, command, cwd in commands:
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        checks[label] = {
            "status": "passed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "duration_ms": duration_ms,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }

    manifest_paths = [
        REPO_ROOT / "data" / "datasets" / "unified_v1" / "unified_all.jsonl",
        REPO_ROOT / "data" / "datasets" / "unified_v1" / "train.jsonl",
        REPO_ROOT / "data" / "datasets" / "unified_v1" / "test.jsonl",
        DEFAULT_INDEX_MANIFEST,
        DEFAULT_QUERY_MANIFEST,
    ]
    checks["dataset_manifest_generation"] = {
        "status": "passed" if all(path.exists() for path in manifest_paths) else "failed",
        "paths": [str(path) for path in manifest_paths],
    }
    _write_json(run_dir / "e5_reliability.json", checks)
    return checks


def _write_manual_templates(run_dir: Path, query_rows: list[dict[str, Any]], sample_size: int) -> dict[str, str]:
    manual_dir = run_dir / "manual_inputs"
    manual_dir.mkdir(parents=True, exist_ok=True)

    e2_rows = []
    for row in query_rows[:sample_size]:
        e2_rows.append(
            {
                "query_id": row.get("query_id"),
                "item_id": row.get("item_id"),
                "domain": row.get("domain"),
                "image_path": row.get("image_path"),
                "ground_truth_identifier": row.get("part_number", ""),
                "ground_truth_part_number": row.get("part_number", ""),
                "ground_truth_maker": row.get("maker", ""),
                "condition": "",
                "notes": "",
            }
        )
    e2_path = manual_dir / "e2_identifier_subset_seed.jsonl"
    _write_jsonl(e2_path, e2_rows)

    e4_path = manual_dir / "e4_pilot_log_template.csv"
    with e4_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "participant_id",
                "task_id",
                "query_id",
                "completion_time_sec",
                "success",
                "manual_edit_count",
                "external_search_used",
                "usability_score",
                "trust_score",
                "notes",
            ]
        )

    task_sheet = manual_dir / "e4_task_sheet.md"
    task_sheet.write_text(
        "\n".join(
            [
                "# E4 Pilot Task Sheet",
                "",
                "1. Upload or search for a part and review the shortlist.",
                "2. Inspect the evidence shown with the result.",
                "3. Decide whether the candidate is trustworthy enough to reuse or store.",
                "",
                "Record completion time, manual edits, external search usage, usability score, and trust score.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "e2_seed": str(e2_path),
        "e4_log_template": str(e4_path),
        "e4_task_sheet": str(task_sheet),
    }


def _write_summary_markdown(
    *,
    run_dir: Path,
    config: dict[str, Any],
    e0_summary: dict[str, Any],
    e1_metrics: dict[str, Any],
    e1_by_domain: dict[str, dict[str, Any]],
    e3_summary: dict[str, Any],
    e5_checks: dict[str, Any] | None,
    manual_templates: dict[str, str],
) -> None:
    lines = [
        "# Current-Index Experiment Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Mode: `{config['mode']}`",
        f"- Query count: `{config['query_count']}`",
        f"- Top-K: `{config['top_k']}`",
        "",
        "## E0",
        "",
        f"- Scenario count: `{e0_summary.get('scenario_count', 0)}`",
        f"- Successful searches: `{e0_summary.get('successful_searches', 0)}`",
        f"- Top-1 results with image attached: `{e0_summary.get('top1_with_image', 0)}`",
        "",
        "## E1",
        "",
        f"- Accuracy@1: `{e1_metrics.get('accuracy_at_1', 0.0)}`",
        f"- Accuracy@5: `{e1_metrics.get('accuracy_at_5', 0.0)}`",
        f"- Recall@5: `{e1_metrics.get('recall_at_5', 0.0)}`",
        f"- MRR: `{e1_metrics.get('mrr', 0.0)}`",
        f"- Exact group hit rate: `{e1_metrics.get('exact_group_hit_rate', 0.0)}`",
        "",
        "### E1 By Domain",
        "",
    ]
    for domain, metrics in sorted(e1_by_domain.items()):
        lines.append(
            f"- `{domain}`: acc@1={metrics.get('accuracy_at_1', 0.0)}, "
            f"acc@5={metrics.get('accuracy_at_5', 0.0)}, "
            f"mrr={metrics.get('mrr', 0.0)}"
        )

    lines.extend(
        [
            "",
            "## E3",
            "",
            f"- Latency sample size: `{e3_summary.get('latency_sample_size', 0)}`",
        ]
    )
    cold_total = (
        e3_summary.get("operations", {})
        .get("cold", {})
        .get("total", {})
    )
    warm_total = (
        e3_summary.get("operations", {})
        .get("warm", {})
        .get("total", {})
    )
    if cold_total:
        lines.append(f"- Cold total mean/p95: `{cold_total.get('mean', 0.0)} / {cold_total.get('p95', 0.0)} ms`")
    if warm_total:
        lines.append(f"- Warm total mean/p95: `{warm_total.get('mean', 0.0)} / {warm_total.get('p95', 0.0)} ms`")

    lines.extend(
        [
            "",
            "## E2 / E4",
            "",
            "- Manual templates generated because these tracks still require human annotation or pilot participants.",
            f"- E2 seed: `{manual_templates['e2_seed']}`",
            f"- E4 log template: `{manual_templates['e4_log_template']}`",
            f"- E4 task sheet: `{manual_templates['e4_task_sheet']}`",
        ]
    )

    if e5_checks is not None:
        lines.extend(["", "## E5", ""])
        for name, result in sorted(e5_checks.items()):
            lines.append(f"- `{name}`: `{result.get('status', 'unknown')}`")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run current-index experiment suite against existing Milvus collections.")
    parser.add_argument("--env-file", type=Path, default=API_ENV_FILE)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_MANIFEST)
    parser.add_argument("--mode", choices=["text", "image", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for E0/E1. 0 means full set.")
    parser.add_argument("--latency-limit", type=int, default=50)
    parser.add_argument("--warm-passes", type=int, default=1)
    parser.add_argument("--scenario-count", type=int, default=8)
    parser.add_argument("--manual-e2-sample-size", type=int, default=200)
    parser.add_argument("--captioner-backend", choices=["inherit", "none", "gpt", "qwen"], default="none")
    parser.add_argument("--enable-ocr-query", choices=["inherit", "0", "1"], default="0")
    parser.add_argument("--use-reranker", choices=["inherit", "0", "1"], default="inherit")
    parser.add_argument("--image-collection", default="qwen3_vl_image_parts")
    parser.add_argument("--text-collection", default="bge_m3_text_parts")
    parser.add_argument("--attrs-collection", default="attrs_parts_v2")
    parser.add_argument("--model-collection", default="bge_m3_model_texts")
    parser.add_argument("--caption-collection", default="bge_m3_caption_parts")
    parser.add_argument("--skip-e5", action="store_true")
    args = parser.parse_args()

    loaded_env = _load_env_file(args.env_file)
    if args.captioner_backend != "inherit":
        os.environ["CAPTIONER_BACKEND"] = args.captioner_backend
    if args.enable_ocr_query != "inherit":
        os.environ["ENABLE_OCR_QUERY"] = args.enable_ocr_query

    use_reranker: bool | None
    if args.use_reranker == "inherit":
        use_reranker = None
    else:
        use_reranker = args.use_reranker == "1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_current_index_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    query_rows = _read_jsonl(args.queries)
    index_rows = _read_jsonl(args.index)
    if args.limit > 0:
        query_rows = query_rows[: args.limit]

    config = {
        "timestamp": timestamp,
        "env_file": str(args.env_file),
        "loaded_env_keys": sorted(loaded_env.keys()),
        "milvus_uri": os.environ.get("MILVUS_URI", ""),
        "mode": args.mode,
        "top_k": args.top_k,
        "query_count": len(query_rows),
        "index_count": len(index_rows),
        "captioner_backend": os.environ.get("CAPTIONER_BACKEND", ""),
        "enable_ocr_query": os.environ.get("ENABLE_OCR_QUERY", ""),
        "use_reranker": args.use_reranker,
        "image_collection": args.image_collection,
        "text_collection": args.text_collection,
        "attrs_collection": args.attrs_collection,
        "model_collection": args.model_collection,
        "caption_collection": args.caption_collection,
    }
    _write_json(run_dir / "config.json", config)

    orchestrator = _build_orchestrator(
        image_collection=args.image_collection,
        text_collection=args.text_collection,
        attrs_collection=args.attrs_collection,
        model_collection=args.model_collection,
        caption_collection=args.caption_collection,
    )

    e0_rows, e0_summary = _run_e0(
        orchestrator=orchestrator,
        query_rows=query_rows,
        mode=args.mode,
        top_k=args.top_k,
        scenario_count=args.scenario_count,
        use_reranker=use_reranker,
    )
    _write_jsonl(run_dir / "e0_scenarios.jsonl", e0_rows)
    _write_json(run_dir / "e0_summary.json", e0_summary)

    predictions, e1_timing_rows, e1_metrics, e1_by_domain, e1_errors = _run_e1(
        orchestrator=orchestrator,
        query_rows=query_rows,
        index_rows=index_rows,
        mode=args.mode,
        top_k=args.top_k,
        use_reranker=use_reranker,
    )
    _write_jsonl(run_dir / "e1_predictions.jsonl", predictions)
    _write_jsonl(run_dir / "e1_query_timings.jsonl", e1_timing_rows)
    _write_json(run_dir / "e1_metrics.json", e1_metrics)
    _write_json(run_dir / "e1_by_domain.json", e1_by_domain)
    _write_jsonl(run_dir / "e1_error_rows.jsonl", e1_errors)

    e3_rows, e3_summary = _run_e3(
        query_rows=query_rows,
        mode=args.mode,
        top_k=args.top_k,
        use_reranker=use_reranker,
        latency_limit=min(args.latency_limit, len(query_rows)),
        warm_passes=args.warm_passes,
        image_collection=args.image_collection,
        text_collection=args.text_collection,
        attrs_collection=args.attrs_collection,
        model_collection=args.model_collection,
        caption_collection=args.caption_collection,
    )
    _write_jsonl(run_dir / "e3_latency_rows.jsonl", e3_rows)
    _write_json(run_dir / "e3_summary.json", e3_summary)

    manual_templates = _write_manual_templates(
        run_dir,
        query_rows=query_rows,
        sample_size=min(args.manual_e2_sample_size, len(query_rows)),
    )
    _write_json(
        run_dir / "manual_status.json",
        {
            "e2_status": "manual_pending",
            "e4_status": "manual_pending",
            "templates": manual_templates,
        },
    )

    e5_checks = None
    if not args.skip_e5:
        e5_checks = _run_e5(run_dir=run_dir)

    _write_summary_markdown(
        run_dir=run_dir,
        config=config,
        e0_summary=e0_summary,
        e1_metrics=e1_metrics,
        e1_by_domain=e1_by_domain,
        e3_summary=e3_summary,
        e5_checks=e5_checks,
        manual_templates=manual_templates,
    )

    print(json.dumps({"run_dir": str(run_dir), "mode": args.mode}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
