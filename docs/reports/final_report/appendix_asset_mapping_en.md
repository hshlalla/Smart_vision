# Appendix Asset Mapping (EN)

This document maps the current workspace files to the final `Appendix A~F` structure.

Priority labels:
- `Primary`: strong candidates that can be inserted immediately
- `Secondary`: useful supplementary material
- `Manual capture needed`: materials that still need to be captured, exported, or cleaned

## Appendix A. User Requirements and Survey Evidence

Primary:
- `docs/planning/google_form_usability_questions_final.md`
- `docs/planning/usability_pilot_plan.md`
- survey / requirement passages in `submission/reports/Draft.docx`

Secondary:
- `submission/feedback/draft_feedback.md`
- `submission/feedback/preliminary_feedback.md`

Manual capture needed:
- survey recruitment message screenshot
- cleaned raw-response table
- usability-response summary derived from `userbillaty.xlsx`

## Appendix B. Evolution of System Architecture (Draft vs. Final)

Primary:
- early design/architecture material in `submission/reports/Draft.docx`
- `docs/architecture/ARCHITECTURE_MERMAID.md`
- `docs/architecture/hybrid_pipeline_overview_ko.md`
- `docs/planning/prd_qwen3_vl_retrieval_v1_to_v5.md`

Secondary:
- `docs/reports/final_report_docx_ready_ko.md`
- `docs/reports/final_report_status.md`

Manual capture needed:
- exported early design figures
- short “superseded design” captions

## Appendix C. User Interface Iterations

Primary:
- `docs/images/fig_index_ui.png`
- `docs/images/fig_4_2_web_search_ui.png`
- `docs/images/fig_catalog_ui.png`
- `docs/images/fig_4_4_agent_chat_ui.png`

Secondary:
- `docs/reports/video_script_korean_final.md`
- `docs/reports/video_recording_cues_english.md`
- any early Gradio screenshots retained from the draft

Manual capture needed:
- login page screenshot
- language-toggle screenshot
- preview/edit/confirm sequence
- early Gradio UI screenshot if available

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

Primary:
- failure-analysis sections in `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- failure-analysis sections in `experiments/qwen3_vl_1000_sample_final_report_en.md`
- OCR failure / Figure 5-2 material in `submission/reports/Draft.docx`

Secondary:
- `experiments/CURRENT_EXPERIMENT_STATUS.md`
- representative sample folders under `data/datasets/unified_v1/items/...`

Manual capture needed:
- noisy label crop screenshots
- wrong Top-1 / correct Top-5 examples
- packaged / low-resolution failure examples

## Appendix E. Project Management and Risk Assessment

Primary:
- `docs/planning/to_do_list.md`
- `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- `docs/reports/future_work_ko.md`
- `submission/feedback/draft_feedback.md`

Secondary:
- `submission/feedback/preliminary_feedback.md`
- `docs/reports/final_report_revision_checklist.md`
- `docs/reports/final_report_status.md`

Manual capture needed:
- GitHub Projects / Kanban screenshot
- milestone / sprint summary table
- risk register table

## Appendix F. Evaluation Data and Extended Results

Primary:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`
- `experiments/CURRENT_EXPERIMENT_STATUS.md`
- `docs/planning/retrieval_eval_inputs.md`
- `docs/planning/usability_results_table_template.md`
- `docs/planning/unified_dataset_schema_and_split.md`
- `data/datasets/unified_v1/eval_v1/eval_summary.json`
- `data/datasets/unified_v1/split_summary.json`
- `data/datasets/unified_v1/materialization_summary.json`
- `apps/api/smart_vision_api/api/v1/hybrid.py`
- `apps/api/smart_vision_api/services/hybrid.py`
- `apps/api/smart_vision_api/schemas/payload.py`
- `packages/model/smart_match/hybrid_search_pipeline/hybrid_pipeline_runner.py`
- `docs/architecture/ARCHITECTURE_MERMAID.md`

Secondary:
- `experiments/run_sampled_ablation.py`
- `experiments/analyze_ablation_results.py`
- `experiments/run_current_index_suite.py`
- `experiments/run_ocr_pilot_benchmark.py`
- `data/scripts/build_unified_dataset.py`
- `data/scripts/prepare_retrieval_eval_inputs.py`
- `packages/model/smart_match/hybrid_search_pipeline/preprocessing/embedding/qwen3_vl_embedding.py`
- `packages/model/smart_match/hybrid_search_pipeline/preprocessing/embedding/bge_m3_encoder.py`

Manual capture needed:
- extended result tables as report figures
- latency charts
- OCR benchmark detail export
- cleaned API/schema tables

## Fast Editing Priority

Best appendices to fill first:
1. `A`
2. `B`
3. `C`
4. `D`
5. `F`

Appendix that can be completed later:
1. `E`
