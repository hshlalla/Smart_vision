# Current Experiment Status

This file is now aligned with the final report package in this folder.

## Submission-Facing Report Files

Use these as the primary experiment write-up:

- [Korean final report](/Users/studio/Downloads/project/Smart_vision/experiments/qwen3_vl_1000_sample_final_report_ko.md)
- [English final report](/Users/studio/Downloads/project/Smart_vision/experiments/qwen3_vl_1000_sample_final_report_en.md)

These report files are the authoritative submission-facing summaries for the
experiment section.

## Reporting Baseline

The final report package is organized around the following narrative:

- dataset scale: `1000 items`
- split: `900 / 100`
- main comparison: `C2` vs `C4`
- OCR benchmark and runtime environment: taken from the report package
- final recommended operating configuration: `C3`

This means:

- the report keeps the 1000-item experiment framing,
- the OCR benchmark section follows the report package,
- and the final recommendation is adjusted using the additional local validation
  that supported `C3`.

## Local Supporting Evidence Retained In This Folder

The following local runs are retained as supporting evidence and traceability:

- reliability refresh:
  - [20260321_004142_reliability_refresh](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_004142_reliability_refresh)
- sampled `C3` baseline:
  - [20260321_012749_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_012749_current_index_hybrid)
- sampled `C1` reranker-on comparison:
  - [20260321_020948_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_020948_current_index_hybrid)
- OCR-only pilot:
  - [20260321_205523_ocr_pilot](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_205523_ocr_pilot)

These runs were used to support the final operational recommendation that
`C3 (OCR off, reranker off)` is the most practical default configuration in
the local validation environment.

## Execution Status

At this point, the experiment package is treated as report-ready.

- the final report drafts have been written,
- the local supporting runs have been archived,
- and active background experiment processes are not required for the report package.

## Practical Note

If experiments are reopened later, the remaining optional follow-up items are:

- operational reproduction of `C2`
- additional `C4` follow-up latency reruns
- full OCR+Qwen benchmark reruns
- `E4` usability study

These are follow-up tasks, not blockers for the current report package.
