Smart Vision API Release Notes
===============================

## 1.0.0

### Added
- Server-side enforcement of `model_id` in `HybridSearchService.index_asset`; requests lacking the field now return a validation error.
- Automatic PK generation (`model_id::api_<uuid>`) so API uploads follow the new Milvus schema while keeping `model_id` available for grouping.

### Changed
- Metadata forwarded to the orchestrator now includes both `model_id` (business identifier) and `pk` (Milvus primary key), aligning API ingestion with demo/cli workflows.

### Fixed
- Prevented silent ingestion when clients omitted `model_id`, ensuring downstream collections never receive anonymous records.


================ pyproject.toml version 변경, ./core/config.py version 변경 ===============