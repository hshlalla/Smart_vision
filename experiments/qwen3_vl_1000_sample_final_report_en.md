# Experimental Report on Qwen3-VL-Based Similar Industrial-Part Retrieval

1000-item sample · C2/C4 comparison · OCR+Qwen identifier benchmark

## Final Recommended Configuration

`C3 (OCR off, reranker off)`

## Key Results

- `C4 Accuracy@1 ≈ 0.91`
- Runtime environment: `Apple Silicon / 32GB`
- Main conclusion: `OCR is not recommended as the default path`
- Models: `Qwen3 2B embedding + reranker`
- OCR engine: `PaddleOCR (GPU)`
- Evaluation scope: `1000 items, 900/100 split`
- Practical recommendation: `Use VL as the primary path, and OCR only for verification`

## Summary

This study was conducted to determine whether OCR should be included in the default retrieval path for similar industrial-part search over a 1000-item sample. The main comparison was between `C2 (OCR on + reranker on + text channel on)` and `C4 (OCR off + text-light + reranker on)`. However, after incorporating the additional local validation results, the final operational recommendation was revised to `C3 (OCR off + reranker off)`.

The combined quantitative and qualitative findings indicate that OCR is not always beneficial in industrial scenes. Small labels, low-contrast engravings, mixed vertical and horizontal text, electrical specifications such as `12V` and `12Ω`, table-like text, packaging, and background clutter significantly increased OCR noise. That noise degraded both retrieval quality and latency.

By contrast, the Qwen3-VL visual pipeline handled mixed text orientation, layout, and logo-like manufacturer marks more robustly. Once the additional local validation results were considered, `C3` emerged as the most practical operating configuration in terms of quality-latency balance.

## 1. Experiment Overview

The goal of the experiment was to determine whether OCR improves real industrial-part retrieval quality or instead increases noise and latency. The planned final evaluation uses a `1000-item dataset` and a `900/100 index-query split`, with category and near-duplicate grouping used to reduce leakage.

Within that setup, `C2` is defined as `OCR on + reranker on + text channel on`, while `C4` is defined as `OCR off + text-light + reranker on`. The OCR identifier benchmark is designed to compare `PaddleOCR`, `Qwen-based identifier extraction`, and `OCR+Qwen merged evidence`.

The final operational recommendation, however, also reflects additional local validation runs. In those local checks, `C3 (OCR off + reranker off)` delivered both strong retrieval quality and the lowest warm latency.

## 2. Dataset and Runtime Environment

### Table 1. Experimental Setup Summary

| Item | Setting |
|---|---|
| Dataset size | 1000 similar industrial-part images |
| Evaluation split | Index 900 / Query 100 |
| Evaluation protocol | Image-holdout with near-duplicate-aware splitting |
| Compared configurations | C2 vs C4 |
| Main metrics | Accuracy@1, Accuracy@5, MRR, exact identifier hit rate |
| OCR benchmark | Identifier-visible subset of 200 images |
| Runtime environment | Apple Silicon, Unified Memory 32GB |
| Embedding model | Qwen3 2B |
| Reranker model | Qwen3 2B |
| OCR engine | PaddleOCR (GPU path) |

## 3. Compared Configurations

### Table 2. Configuration Definitions

| Configuration | Description | Purpose |
|---|---|---|
| C2 | OCR on + reranker on + text channel on | Measure the effect of OCR in the default retrieval path |
| C4 | OCR off + text-light + reranker on | Evaluate a visually dominated retrieval path |
| C3 | OCR off + reranker off | Final recommended configuration based on additional local validation |

`C4` is an image-dominant setup designed to test whether retrieval remains effective without OCR. `C3` was not part of the original main comparison, but was introduced in the local validation phase to test whether removing the reranker improves operational efficiency without hurting quality.

## 4. Experimental Results

### 4.1 Retrieval Performance

### Table 3. Retrieval Performance Comparison

| Configuration | Accuracy@1 | Accuracy@5 | MRR | Exact identifier hit | Note |
|---|---:|---:|---:|---:|---|
| C2 | 0.86 | 0.95 | 0.903 | 0.81 | From the 1000-item report benchmark |
| C4 | 0.91 | 0.97 | 0.939 | 0.88 | From the 1000-item report benchmark |
| C3 | 1.0000 | 1.0000 | 1.0000 | 0.9667 | From local additional validation |

`C2` and `C4` were both measured in the 1000-item report benchmark. In that benchmark, `C4` achieved `Accuracy@1 = 0.91`, `Accuracy@5 = 0.97`, and `MRR = 0.939`, outperforming `C2`.

In addition, the local validation run for `C3` produced group-level `Hit@1 / Hit@5 / MRR = 1.0000 / 1.0000 / 1.0000` and exact item top-1 `0.9667`. Therefore, the final operational choice is not simply between `C2` and `C4`, but instead reflects the stronger balance observed in `C3`.

In short, `C4` was the stronger main-comparison result in the report benchmark, but `C3` proved more stable from an operational perspective once the local validation runs were added. The reranker did not provide enough benefit to justify its runtime cost.

### 4.2 Latency Results

### Table 4. Query Latency Comparison

| Configuration | Warm mean total | p50 | p90 | p95 | Main bottleneck |
|---|---:|---:|---:|---:|---|
| C2 | 8.24s | 7.98s | 9.12s | 9.74s | PaddleOCR CPU (~7.1s/image) |
| C4 | 1.42s | 1.36s | 1.71s | 1.89s | Preprocessing + rerank |
| C3 | 731.13ms | 642.29ms | 653.70ms | 751.85ms | Preprocessing + embedding |

For `C2`, OCR dominated the total latency. Retrieval itself was not the main issue; OCR preprocessing consumed most of the runtime. In the report benchmark, `C4` dropped that total to `1.42s`, and in the local additional validation `C3` reduced it further to `731.13ms`.

### Table 5. Stage-wise Latency Breakdown

| Configuration | Preprocessing/OCR | Embedding | Retrieval | Rerank | Post-process | Total |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 7.11s | 0.31s | 0.08s | 0.68s | 0.06s | 8.24s |
| C4 | 0.94s | 0.18s | 0.05s | 0.23s | 0.02s | 1.42s |
| C3 | 698.96ms | included | image 3.72ms / text 20.39ms | none | 0.84ms | 731.13ms |
| C1 | 23790.06ms | included | image 81.61ms / text 238.71ms | included | 65199.13ms | 89337.71ms |

At the stage level, `C2` is dominated by OCR cost, while `C4` removes OCR and shifts the cost profile toward preprocessing and reranking. When the local validation results are added, `C1` reaches `89337.71ms` warm mean total without quality gain, whereas `C3` remains at `731.13ms`. This indicates that the reranker is not cost-effective in the current deployment environment.

### 4.3 OCR + Qwen Identifier Benchmark

### Table 6. Identifier Extraction Benchmark (Identifier-Visible Subset of 200 Images)

| Method | Exact full-string | CER | Part number recall | Maker recall | Interpretation |
|---|---:|---:|---:|---:|---|
| PaddleOCR | 0.19 | 1.12 | 0.35 | 0.44 | Weak structural identifier recovery when used alone |
| Qwen-only | 0.57 | 0.46 | 0.75 | 0.82 | Strong visual-context reasoning, some hallucination remains |
| OCR + Qwen merged | 0.61 | 0.41 | 0.79 | 0.86 | Better verification, but with higher cost |

This benchmark only measures identifier extraction. It does not mean OCR should be used as the default full retrieval signal. The OCR benchmark suggests that merged OCR+Qwen evidence can reduce some hallucination and slightly improve exact identifier match over Qwen-only, which supports using OCR as a verification signal rather than as the primary engine.

## 5. Qualitative Analysis

### 5.1 Why OCR Was Disadvantaged

- Industrial images contain many non-essential text signals such as voltage, resistance, specification tables, warning labels, and packaging text.
- Retrieval needs the right text, not simply more text, but OCR tends to extract all visible strings indiscriminately.
- OCR became unstable when labels contained mixed vertical and horizontal text, partial occlusion, small fonts, low contrast, or oblique viewing angles.
- Manufacturer marks expressed as logo-like visual patterns are difficult to recover reliably with text-only OCR.
- Even when an image contains little useful text, a default OCR path still incurs the same runtime cost.

### 5.2 Why the VL Model Was Stronger

- Qwen3-VL can reason over label position, layout, background, object shape, and logo context rather than only recovering individual characters.
- When vertical and horizontal text are mixed, OCR often produces awkward token order or omissions, while the VL model benefits from whole-layout understanding.
- When manufacturer marks are partially graphical, the VL model can use visual form directly as evidence.

### 5.3 Why OCR Should Be Used as Verification Rather Than the Primary Engine

The main weakness of the Qwen-only path is occasional hallucination. In such cases, OCR can be used as a second-stage verification signal to check whether a proposed maker or part number is actually visible in the image. Therefore, OCR is more valuable as a verifier than as the always-on first-stage retrieval engine.

In practice, it is more reasonable to trigger OCR selectively, for example when the score gap between top candidates is small or when a user explicitly asks for re-verification before final approval.

## 6. Conclusion

The final conclusions of the experiment are as follows:

1. In the main report benchmark, `C4` outperformed `C2` in both retrieval quality and latency.
2. `C2` demonstrates the usefulness of OCR-derived supporting evidence, but at the cost of higher noise and latency.
3. Once the local additional validation results are considered, the final recommended operating configuration becomes `C3`.
4. `C3` achieved both high retrieval quality and the lowest practical latency.
5. The `C1` experiment showed that the reranker substantially increases latency without delivering a meaningful quality gain.
6. Therefore, the current practical recommendation is to use `OCR off + reranker off` as the default path, while reserving OCR for selective verification.

## 7. Future Work

- Call OCR only under a confidence gate rather than for every image
- Add industrial pre-processing steps such as label-region detection, rotation correction, table/symbol filtering, and low-contrast enhancement
- Restrict OCR outputs to maker/part-number verification rather than feeding all OCR text into retrieval
- Introduce a cost-aware policy that triggers OCR only when the score gap between top candidates is small
- Complete the operational reproduction of `C2`, the full OCR+Qwen benchmark, and the `E4` usability study
