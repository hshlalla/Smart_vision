Smart Vision Model Release Notes
================================

## 1.0.0

### Added
- Derived `text_corpus` that blends OCR output with structured metadata so dense text search can match maker, part number, and category cues in addition to raw OCR text.
- Utility helpers on `HybridMilvusIndex` to report collection statistics and drop collections, enabling downstream tooling (demo status tab) without duplicating Milvus logic.

### Changed
- Switched Milvus primary keys for `image_parts`, `text_parts`, and `attrs_parts` to VARCHAR and drive them from caller-provided `pk` values, allowing multiple images per `model_id`.
- Normalizer now preserves both `model_id` and `pk` values while standardising maker/part/category fields.
- Stored `ocr_text` now uses the combined corpus (metadata phrases + OCR text) for consistency with the new embedding.

### Fixed
- Prevented OCR token join failures by ensuring tokens are converted to strings before concatenation.
- Ensured Milvus vector dimensions are inferred from the underlying BGE-VL projection (768 dim) to avoid 1024-dimension schema mismatches.

================ pyproject.toml version 변경 ==============================