#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_latency(summary: dict[str, Any]) -> dict[str, float]:
    warm_total = (((summary.get("operations") or {}).get("warm") or {}).get("total") or {})
    cold_total = (((summary.get("operations") or {}).get("cold") or {}).get("total") or {})
    return {
        "cold_mean": float(cold_total.get("mean") or 0.0),
        "cold_p95": float(cold_total.get("p95") or 0.0),
        "warm_mean": float(warm_total.get("mean") or 0.0),
        "warm_p95": float(warm_total.get("p95") or 0.0),
    }


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter()
    by_domain = Counter()
    top_examples: list[dict[str, Any]] = []
    for row in rows:
        by_type[str(row.get("failure_type") or "unknown")] += 1
        by_domain[str(row.get("domain") or "unknown")] += 1
    for row in rows[:10]:
        top_examples.append(
            {
                "query_id": row.get("query_id"),
                "item_id": row.get("item_id"),
                "domain": row.get("domain"),
                "failure_type": row.get("failure_type"),
                "found_rank": row.get("found_rank"),
                "top_prediction_id": row.get("top_prediction_id"),
                "query_text": row.get("query_text"),
            }
        )
    return {
        "failure_count": len(rows),
        "by_type": dict(sorted(by_type.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "examples": top_examples,
    }


def _derive_findings(configs: dict[str, dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    c1 = configs.get("c1")
    c3 = configs.get("c3")
    if c1 and c3:
        hit1_delta = round(float(c1["metrics"]["accuracy_at_1"]) - float(c3["metrics"]["accuracy_at_1"]), 4)
        mrr_delta = round(float(c1["metrics"]["mrr"]) - float(c3["metrics"]["mrr"]), 4)
        warm_delta_ms = round(float(c1["latency"]["warm_mean"]) - float(c3["latency"]["warm_mean"]), 2)
        if abs(hit1_delta) < 0.01 and abs(mrr_delta) < 0.01:
            findings.append(
                "Reranker effect is marginal in this sampled holdout run; shortlist quality changed by less than 0.01 on Hit@1 and MRR."
            )
        elif hit1_delta > 0:
            findings.append(
                f"Reranker improved retrieval in the sampled holdout run: Hit@1 delta {hit1_delta:+.4f}, MRR delta {mrr_delta:+.4f}."
            )
        else:
            findings.append(
                f"Reranker did not help on this sampled holdout run: Hit@1 delta {hit1_delta:+.4f}, MRR delta {mrr_delta:+.4f}."
            )
        if warm_delta_ms > 50:
            findings.append(
                f"Reranker cost is noticeable in warm latency (+{warm_delta_ms:.2f} ms mean). This overhead should be justified by a measurable ranking gain."
            )
        elif warm_delta_ms < -50:
            findings.append(
                f"Unexpectedly, the reranker-on condition was faster by {-warm_delta_ms:.2f} ms mean warm latency. This likely indicates runtime variance rather than a true speed advantage."
            )
    for name, payload in sorted(configs.items()):
        by_domain = payload.get("by_domain") or {}
        auto = by_domain.get("auto_part") or {}
        semi = by_domain.get("semiconductor_equipment_part") or {}
        if auto and semi:
            delta = round(float(auto.get("accuracy_at_1") or 0.0) - float(semi.get("accuracy_at_1") or 0.0), 4)
            if abs(delta) >= 0.03:
                worse = "semiconductor_equipment_part" if delta > 0 else "auto_part"
                findings.append(
                    f"{worse} is harder in this run; domain gap in Hit@1 is {abs(delta):.4f}."
                )
    return findings


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Ablation Analysis",
        "",
        f"- Run root: `{analysis['run_root']}`",
        f"- Sample summary: `{json.dumps(analysis['sample_summary'], ensure_ascii=False)}`",
        "",
        "## Config Summary",
        "",
    ]
    for name, payload in sorted(analysis["configs"].items()):
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Label: `{payload['label']}`",
                f"- Hit@1: `{payload['metrics']['accuracy_at_1']}`",
                f"- Hit@5: `{payload['metrics']['accuracy_at_5']}`",
                f"- MRR: `{payload['metrics']['mrr']}`",
                f"- Warm mean / p95: `{payload['latency']['warm_mean']} / {payload['latency']['warm_p95']} ms`",
                f"- Failure count: `{payload['errors']['failure_count']}`",
                "",
            ]
        )
    lines.extend(["## Findings", ""])
    for finding in analysis["findings"]:
        lines.append(f"- {finding}")
    lines.extend(["", "## Error Notes", ""])
    for name, payload in sorted(analysis["configs"].items()):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- By failure type: `{json.dumps(payload['errors']['by_type'], ensure_ascii=False)}`")
        lines.append(f"- By domain: `{json.dumps(payload['errors']['by_domain'], ensure_ascii=False)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse sampled ablation results and derive concise findings.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.run_root / "ablation_summary.json"
    payload = _read_json(summary_path)
    configs_out: dict[str, dict[str, Any]] = {}
    for name, config in sorted((payload.get("configs") or {}).items()):
        suite_run_dir = Path(config["suite_run_dir"])
        metrics = _read_json(suite_run_dir / "e1_metrics.json")
        by_domain = _read_json(suite_run_dir / "e1_by_domain.json")
        latency = _extract_latency(_read_json(suite_run_dir / "e3_summary.json"))
        error_rows = _read_jsonl(suite_run_dir / "e1_error_rows.jsonl")
        configs_out[name] = {
            "label": config["label"],
            "reused_from": config.get("reused_from"),
            "metrics": metrics,
            "by_domain": by_domain,
            "latency": latency,
            "errors": _error_summary(error_rows),
        }

    analysis = {
        "run_root": str(args.run_root),
        "sample_summary": payload.get("sample_summary") or {},
        "configs": configs_out,
    }
    analysis["findings"] = _derive_findings(configs_out)

    out_json = args.run_root / "analysis_summary.json"
    out_md = args.run_root / "analysis_summary.md"
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, analysis)
    print(json.dumps({"analysis_json": str(out_json), "analysis_md": str(out_md)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
