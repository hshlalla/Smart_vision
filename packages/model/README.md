# smart-match-model

`packages/model` contains the reusable Smart Match indexing and retrieval package used by the API and demo.

## Pipeline Overview

### Indexing

- Image embedding
  - `Qwen3-VL-Embedding-2B`
- Text embedding
  - `BGE-M3`
- Optional captioning
  - GPT or Qwen depending on runtime mode
- Optional OCR
  - Controlled by env flags
- Metadata normalization
  - maker / part number / category cleanup
- Duplicate-aware ingestion
  - interactive preview can surface a duplicate candidate for user review
  - batch ingestion can reuse an existing model when normalized `maker + part_number` or `part_number` matches
  - richer later metadata is merged instead of blindly discarded
- Milvus persistence
  - image vectors
  - text vectors
  - attrs records
  - model-level text vectors

### Search

- Text-only
  - lightweight `BGE-M3 + model collection`
- Multimodal
  - heavy orchestrator path using image/text signals
- Optional reranking
  - `Qwen3-VL-Reranker-2B`

## Important Architecture Changes

- `model_id` counters no longer use a Milvus `sv_counters` collection.
- Counters are now stored in local SQLite for simpler single-node operation.
- Apple Silicon device selection supports `mps`.
- OCR can be fully disabled or separated between indexing and query-time.
- Metadata preview experiments moved away from OCR-first on Apple Silicon because OCR preview was too slow and inaccurate for the current workflow.
- Repeated ingestion is no longer treated as a pure duplicate-drop problem. The current policy is to preserve richer later data by merging text fields and appending genuinely new images under the existing `model_id`.

## Installation

```bash
python -m pip install --upgrade pip
pip install -e .
```

Recommended Python: `3.11+`

## Example

```python
from smart_match import HybridSearchOrchestrator
from smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import FusionWeights, MilvusConnectionConfig

orchestrator = HybridSearchOrchestrator(
    milvus=MilvusConnectionConfig(uri="tcp://localhost:19530"),
    fusion_weights=FusionWeights(alpha=0.6, beta=0.4),
)

model_id = orchestrator.allocate_model_id(category="ACTUATOR")
metadata = {
    "model_id": model_id,
    "maker": "Hyundai",
    "part_number": "91200-4F310",
    "category": "ACTUATOR",
    "description": "example",
}
orchestrator.preprocess_and_index("sample.jpg", metadata)

results = orchestrator.search(query_text="91200 4F310", top_k=5)
print(results)
```

## Runtime Controls

- `ENABLE_OCR`
- `ENABLE_OCR_INDEXING`
- `ENABLE_OCR_QUERY`
- `ENABLE_RERANKER`
- `RERANKER_DEVICE`
- `RERANKER_MAX_LENGTH`
- `CAPTIONER_BACKEND`
- `LOCAL_MODE`
- `HYBRID_IMAGE_COLLECTION`
- `HYBRID_TEXT_COLLECTION`
- `HYBRID_ATTRS_COLLECTION`
- `HYBRID_MODEL_COLLECTION`
- `HYBRID_CAPTION_COLLECTION`

## Performance Notes

- Text-only retrieval should use the lightweight path whenever possible.
- Apple Silicon local experiments are viable, but not CUDA-class for heavy multimodal indexing.
- The Qwen multimodal reranker now uses the official `Qwen3VLForConditionalGeneration` loading path, but on the current Apple Silicon machine it may still need `RERANKER_DEVICE=cpu` because `mps` can fail during scoring.
- The main local bottlenecks are usually:
  - first model load
  - image embedding
  - Milvus insert/flush
  - reranker scoring when enabled on `cpu`
- Current preprocessing stores a resized image copy for embedding/media, while preserving the larger source image path for operations that need higher fidelity.
- Practical caption policy in this repo:
  - hosted/default mode: `CAPTIONER_BACKEND=gpt`
  - local mode experiments: `CAPTIONER_BACKEND=qwen`
  - `auto` follows `LOCAL_MODE` first, then API key availability
