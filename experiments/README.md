# Experiments

This directory contains:

- the final report drafts used for submission,
- the experiment runners used to generate supporting evidence,
- and the local run artifacts under `experiments/runs/`.

## Submission-Facing Documents

Use these two files as the primary experiment write-up:

- [Korean final report](/Users/studio/Downloads/project/Smart_vision/experiments/qwen3_vl_1000_sample_final_report_ko.md)
- [English final report](/Users/studio/Downloads/project/Smart_vision/experiments/qwen3_vl_1000_sample_final_report_en.md)

Use this file for quick context on how the folder is organized:

- [CURRENT_EXPERIMENT_STATUS.md](/Users/studio/Downloads/project/Smart_vision/experiments/CURRENT_EXPERIMENT_STATUS.md)

## Runner Scripts

- [run_current_index_suite.py](/Users/studio/Downloads/project/Smart_vision/experiments/run_current_index_suite.py)
- [run_sampled_ablation.py](/Users/studio/Downloads/project/Smart_vision/experiments/run_sampled_ablation.py)
- [run_ocr_pilot_benchmark.py](/Users/studio/Downloads/project/Smart_vision/experiments/run_ocr_pilot_benchmark.py)

These scripts:

- reuse the existing Milvus collections or dedicated experiment collections,
- run retrieval evaluation without going through the web UI,
- and save raw predictions, latency logs, summaries, and manual-study templates.

## Local Output

`experiments/runs/` is treated as local output storage rather than source-controlled documentation.

Typical outputs include:

- `config.json`
- `e0_*`
- `e1_*`
- `e3_*`
- `e5_*`
- `manual_inputs/*`

## Note

The final report files above are the authoritative submission-facing summaries.
The run artifacts and local supporting experiments in this folder are retained
for traceability, debugging, and optional follow-up reruns.
