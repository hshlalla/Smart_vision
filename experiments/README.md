# Experiments

This directory contains reproducible experiment runners and saved artifacts for
the report evaluation tracks (`E0` to `E5`).

Current entry point:

- [`run_current_index_suite.py`](/Users/studio/Downloads/project/Smart_vision/experiments/run_current_index_suite.py)
- [`run_sampled_ablation.py`](/Users/studio/Downloads/project/Smart_vision/experiments/run_sampled_ablation.py)
- [`run_ocr_pilot_benchmark.py`](/Users/studio/Downloads/project/Smart_vision/experiments/run_ocr_pilot_benchmark.py)
- [`CURRENT_EXPERIMENT_STATUS.md`](/Users/studio/Downloads/project/Smart_vision/experiments/CURRENT_EXPERIMENT_STATUS.md)

What it does:

- Reuses the current Milvus collections that are already indexed
- Runs retrieval evaluation without going through the web UI
- Saves raw predictions, latency logs, summaries, and manual-study templates
- Supports separate sampled ablation runs against dedicated experiment collections when the main collections should not be dropped

Recommended usage:

```bash
cd /Users/studio/Downloads/project/Smart_vision
source /Users/studio/Downloads/project/Smart_vision/.venv/bin/activate
/Users/studio/Downloads/project/Smart_vision/.venv/bin/python \
  experiments/run_current_index_suite.py \
  --mode hybrid \
  --captioner-backend none
```

Output:

- `experiments/runs/<timestamp>/config.json`
- `experiments/runs/<timestamp>/e0_*`
- `experiments/runs/<timestamp>/e1_*`
- `experiments/runs/<timestamp>/e3_*`
- `experiments/runs/<timestamp>/e5_*`
- `experiments/runs/<timestamp>/manual_inputs/*`

Notes:

- `E1`, `E3`, and `E5` can be automated immediately against the current index.
- `E2` and `E4` still require human-labeled/manual inputs, so the runner
  generates seed files and templates for those tracks instead of pretending that
  they are fully automatic.
- Long-running local experiment constraints and partial-result caveats are tracked in [`CURRENT_EXPERIMENT_STATUS.md`](/Users/studio/Downloads/project/Smart_vision/experiments/CURRENT_EXPERIMENT_STATUS.md).
- `experiments/runs/` is treated as local output storage rather than source-controlled documentation.
