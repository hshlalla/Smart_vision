# Current Experiment Status

This note distinguishes:

- results that are already usable as evidence,
- results that are only sanity checks,
- experiments that are still in progress,
- and planned experiments that should not yet be claimed as completed.

## 1. Usable Now

### E5 Reliability Refresh

Latest refresh:

- [20260321_004142_reliability_refresh](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_004142_reliability_refresh)
- [e5_reliability.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_004142_reliability_refresh/e5_reliability.json)

Current conclusion:

- API regression tests pass.
- Model-package tests pass.
- Frontend production build passes.
- Retrieval-eval input generation still works.

This is valid report evidence for engineering reliability.

### Current-Index Sanity Benchmark

Existing run:

- [20260320_233206_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260320_233206_current_index_hybrid)
- [summary.md](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260320_233206_current_index_hybrid/summary.md)

Observed values:

- `E0`: 8/8 scenario success
- `E1 sanity`: `Accuracy@1 = 0.9732`, `Hit@5 = 1.0`, `MRR = 0.9855`
- `E3`: warm total mean about `653.65 ms`

Current conclusion:

- The deployed hybrid pipeline is operational.
- Warm interactive retrieval on the already-indexed collection is feasible.
- The current stack can usually return shortlist results quickly after warm-up.

### Sampled Holdout Baseline (`C3`)

Completed sampled holdout run:

- [20260321_011908_sampled_ablation](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_011908_sampled_ablation)
- baseline suite run: [20260321_012749_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_012749_current_index_hybrid)
- analysis note: [analysis_summary.md](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_011908_sampled_ablation/analysis_summary.md)

Protocol:

- `30` sampled items
- `1` held-out query image per item
- `1` index image per item
- `OCR off`
- `reranker off`

Observed values:

- group-level `Hit@1 = 1.0`
- group-level `Hit@5 = 1.0`
- `MRR = 1.0`
- exact `item_id@1 = 0.9667`
- no `E1` error rows

Latency observations:

- warm mean total latency: about `731.13 ms`
- warm mean preprocessing time: about `698.96 ms`
- warm mean text search time: about `20.39 ms`
- warm mean image search time: about `3.72 ms`

Domain-specific latency:

- `auto_part`: mean total `641.38 ms`
- `semiconductor_equipment_part`: mean total `3230.2 ms`

Current conclusion:

- On the sampled holdout task, the retrieval baseline is very strong.
- Most of the latency comes from query-side preprocessing and embedding, not Milvus search itself.
- Semiconductor equipment images are substantially slower than automotive part images in the current local setup.
- The group-level metric is slightly optimistic; one query retrieved the correct group but a different item with the same group/part identity.

## 2. Not Valid As Main Evidence Yet

The current-index sanity benchmark above should **not** be used as the main retrieval-ablation result.

Reasons:

1. The current `train/test` split is item/group-based, not image-holdout.
2. The current indexed Milvus state already contains all images, so query-image leakage is possible if this run is interpreted as a strict generalisation benchmark.
3. The old metric set duplicated information:
   - `accuracy_at_1` and `exact_group_hit_rate` are effectively the same
   - `accuracy_at_5` and `recall_at_5` are effectively the same

Current conclusion:

- The sanity run is useful as an operational benchmark.
- It is not sufficient as the main controlled retrieval-ablation result.

## 3. Controlled Sampled Retrieval Ablation

What was attempted:

- separate experimental Milvus collections
- image-holdout protocol instead of the original item/group split
- direct `C1` vs `C3` comparison

What is now known:

- the first `C1` sampled run hit a collection-loading bug in the experiment harness; this was fixed by loading vector collections in the evaluation runner
- the original reranker initialization bug has now been fixed in code
- the reranker now loads through the official `Qwen3VLForConditionalGeneration` path and a derived `yes/no` scoring head
- the reranker still cannot run on local Apple MPS for this model because MetalPerformanceShaders crashes with an assertion during real scoring
- the reranker does run on `cpu`, but the per-query latency is very high on this machine

Compared configurations:

- `C1`: `OCR off + reranker on`
- `C3`: `OCR off + reranker off`

Completed reranker-on run:

- [20260321_020948_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_020948_current_index_hybrid)
- [e1_metrics.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_020948_current_index_hybrid/e1_metrics.json)
- [e3_summary.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_020948_current_index_hybrid/e3_summary.json)

Observed values:

- `C1` group-level `Hit@1 = 1.0`
- `C1` group-level `Hit@5 = 1.0`
- `C1` `MRR = 1.0`
- exact `item_id@1 = 1.0`
- `C1` warm total mean latency: `89337.71 ms`
- `C1` warm `finalize` mean latency: `65199.13 ms`
- `C1` warm `preprocessing` mean latency: `23790.06 ms`

Current conclusion:

- the baseline holdout retrieval can be evaluated now
- the reranker path is no longer broken at initialization time
- on the sampled holdout task, reranker-on did not improve retrieval quality over the reranker-off baseline
- on the current local machine, reranker-on increased warm latency from roughly `0.73s` (`C3`) to roughly `89.34s` (`C1`)
- the current practical conclusion is that reranker-on is not operationally justified on this Apple Silicon setup

Reference files:

- older failure note: [c1_reranker_failure.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_011908_sampled_ablation/c1_reranker_failure.json)
- sampled analysis note: [analysis_summary.md](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_011908_sampled_ablation/analysis_summary.md)
- completed reranker-on run directory: [20260321_020948_current_index_hybrid](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_020948_current_index_hybrid)

## 4. OCR Evaluation Status

The OCR-related full retrieval ablation is still **not complete**, but a lighter OCR-only pilot has now completed.

What was attempted:

- OCR/Qwen pilot script:
  - [run_ocr_pilot_benchmark.py](/Users/studio/Downloads/project/Smart_vision/experiments/run_ocr_pilot_benchmark.py)
- micro-pilot run directory:
  - [20260321_013000_ocr_pilot](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_013000_ocr_pilot)

Observed runtime issue in the earlier full OCR+Qwen pilot:

- even a `2`-sample micro-pilot did not complete within the local time budget after PaddleOCR-VL and Qwen runtime initialization
- the process had to be stopped manually after several minutes without producing completed result files

Completed lighter OCR-only pilot:

- [20260321_205523_ocr_pilot](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_205523_ocr_pilot)
- [e2_ocr_pilot_summary.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_205523_ocr_pilot/e2_ocr_pilot_summary.json)

Protocol:

- `10` sampled items
- up to `1` image per item
- `skip_qwen = true`
- heuristic best-image selection by OCR signal

Observed OCR-only values:

- Paddle part-number exact rate: `0.1`
- Paddle part-number recall rate: `0.1`
- Paddle maker exact rate: `0.1`
- Paddle maker recall rate: `0.1`
- Paddle part-number mean CER: `1.9625`
- total duration: `523.45s`

Current conclusion:

- OCR remains an unresolved design choice for the full retrieval pipeline.
- On the current Apple Silicon local environment, the OCR+Qwen identifier path is operationally expensive enough to be a practical bottleneck.
- Even the lighter OCR-only pilot shows weak structured identifier recovery on these sampled images.
- The current evidence does not support enabling OCR as a routine local retrieval signal on this hardware/workload.

Reference file:

- timeout note: [timeout_note.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_013000_ocr_pilot/timeout_note.json)
- completed OCR-only pilot: [e2_ocr_pilot_summary.json](/Users/studio/Downloads/project/Smart_vision/experiments/runs/20260321_205523_ocr_pilot/e2_ocr_pilot_summary.json)

## 5. Recommended Report Framing Right Now

Safe claims:

- the system works end-to-end
- current-index sanity benchmarking shows strong operational retrieval on the already indexed collection
- engineering reliability evidence is available
- sampled image-holdout baseline retrieval on isolated collections is available
- reranker-on does not improve the sampled holdout result but dramatically increases latency on the current local hardware
- OCR+Qwen is operationally expensive, and a lighter OCR-only pilot also shows weak identifier recovery on sampled images

Unsafe claims:

- that the full 1491-item controlled ablation is already completed
- that OCR-on is quantitatively better for full end-to-end retrieval without a dedicated OCR-on retrieval benchmark
- that the current sanity benchmark alone proves strict generalisation performance
