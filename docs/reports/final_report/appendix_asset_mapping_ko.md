# Appendix Asset Mapping (KO)

이 문서는 최종 `Appendix A~F`에 실제로 넣을 수 있는 현재 작업공간의 파일을 섹션별로 매핑한 작업용 문서다.

우선순위:
- `우선 넣을 것`: 바로 appendix에 넣기 좋은 핵심 자료
- `보조 후보`: 필요할 때 보강용으로 넣을 자료
- `수동 추가 필요`: 현재 파일 형태로 없어서 캡처/정리해야 하는 자료

## Appendix A. User Requirements and Survey Evidence

우선 넣을 것:
- `docs/planning/google_form_usability_questions_final.md`
- `docs/planning/usability_pilot_plan.md`
- `submission/reports/Draft.docx` 내 survey / requirements 관련 부분

보조 후보:
- `submission/feedback/draft_feedback.md`
- `submission/feedback/preliminary_feedback.md`

수동 추가 필요:
- survey recruitment message screenshot
- cleaned raw-response table
- `userbillaty.xlsx` 기반 요약 표 또는 캡처

## Appendix B. Evolution of System Architecture (Draft vs. Final)

우선 넣을 것:
- `submission/reports/Draft.docx` 내 초기 architecture / design figures
- `docs/architecture/ARCHITECTURE_MERMAID.md`
- `docs/architecture/hybrid_pipeline_overview_ko.md`
- `docs/planning/prd_qwen3_vl_retrieval_v1_to_v5.md`

보조 후보:
- `docs/reports/final_report_docx_ready_ko.md`
- `docs/reports/final_report_status.md`

수동 추가 필요:
- early design figure export
- “superseded design” 캡션 문장

## Appendix C. User Interface Iterations

우선 넣을 것:
- `docs/images/fig_index_ui.png`
- `docs/images/fig_4_2_web_search_ui.png`
- `docs/images/fig_catalog_ui.png`
- `docs/images/fig_4_4_agent_chat_ui.png`

보조 후보:
- `docs/reports/video_script_korean_final.md`
- `docs/reports/video_recording_cues_english.md`
- draft에 포함된 Gradio UI 캡처

수동 추가 필요:
- login page screenshot
- language-toggle screenshot
- preview/edit/confirm sequence screenshot
- early Gradio UI screenshot if available

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

우선 넣을 것:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`
- `submission/reports/Draft.docx`의 OCR failure / Figure 5-2 관련 내용

보조 후보:
- `experiments/CURRENT_EXPERIMENT_STATUS.md`
- representative sample item folders under `data/datasets/unified_v1/items/...`

수동 추가 필요:
- noisy label crop screenshots
- wrong Top-1 / correct Top-5 examples
- packaged / low-resolution failure examples

## Appendix E. Project Management and Risk Assessment

우선 넣을 것:
- `docs/planning/to_do_list.md`
- `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- `docs/reports/future_work_ko.md`
- `submission/feedback/draft_feedback.md`

보조 후보:
- `submission/feedback/preliminary_feedback.md`
- `docs/reports/final_report_revision_checklist.md`
- `docs/reports/final_report_status.md`

수동 추가 필요:
- GitHub Projects / Kanban screenshot
- milestone / sprint summary table
- risk register table

## Appendix F. Evaluation Data and Extended Results

우선 넣을 것:
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

보조 후보:
- `experiments/run_sampled_ablation.py`
- `experiments/analyze_ablation_results.py`
- `experiments/run_current_index_suite.py`
- `experiments/run_ocr_pilot_benchmark.py`
- `data/scripts/build_unified_dataset.py`
- `data/scripts/prepare_retrieval_eval_inputs.py`
- `packages/model/smart_match/hybrid_search_pipeline/preprocessing/embedding/qwen3_vl_embedding.py`
- `packages/model/smart_match/hybrid_search_pipeline/preprocessing/embedding/bge_m3_encoder.py`

수동 추가 필요:
- extended result tables as report figures
- latency charts
- OCR benchmark detail export
- cleaned API/schema table

## 빠른 편집 우선순위

먼저 채우면 좋은 Appendix:
1. `A`
2. `B`
3. `C`
4. `D`
5. `F`

나중에 보강해도 되는 Appendix:
1. `E`
