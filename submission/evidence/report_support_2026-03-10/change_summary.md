## Validated Change Summary

### 1. Agent writeback safety
- `update_milvus` default changed from `true` to `false`.
- Intent: prevent uncertain agent outputs from automatically polluting Milvus.
- Backend reference: `apps/api/smart_vision_api/schemas/agent.py`
- Frontend reference: `apps/web/src/views/AgentChatPage.tsx`

### 2. Explicit UI control for Milvus updates
- Added a UI switch so operators can opt into writeback only when appropriate.
- This supports a human-review-first workflow for reportable product identification.

### 3. Hybrid search timing instrumentation
- Added structured timing capture for:
  - `preprocessing`
  - `image_search`
  - `ocr_search`
  - `caption_search`
  - `text_search`
  - `fetch_models`
  - `finalize`
  - `total`
- Backend reference: `packages/model/smart_match/hybrid_search_pipeline/hybrid_pipeline_runner.py`
- Reporting value: supports later `p50/p90/p95` latency analysis.

### 4. Test coverage updates
- Added API tests verifying:
  - `update_milvus` defaults to `false`
  - `update_milvus=true` is still accepted when explicitly requested
- Backend test reference: `apps/api/tests/test_agent_chat_api.py`

### 5. Model package testability fix
- Converted package-level eager imports to lazy imports in:
  - `packages/model/__init__.py`
  - `packages/model/smart_match/__init__.py`
- Intent: allow lightweight tests to import submodules without requiring heavyweight runtime dependencies such as `torch`.

### 6. Planning document update
- Updated `docs/to_do_list.md` to reflect:
  - completed safety hardening
  - completed latency instrumentation
  - new high-priority item for human-review-based writeback gating
  - new high-priority item for evaluation automation scripts
  - legacy test asset cleanup backlog

### Git diff summary at time of artifact creation
- `7 files changed, 172 insertions(+), 15 deletions(-)`
