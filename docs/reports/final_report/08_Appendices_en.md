# 8. Appendices

This section contains supplementary materials that support the main report while preserving the project’s iterative development from the earlier draft to the final prototype. Earlier artefacts should be labelled clearly as earlier concepts or superseded designs where they no longer match the final implementation.

## Appendix A. User Requirements and Survey Evidence

This appendix contains the supporting material for the early requirements-elicitation survey (`n = 6`) and the later pilot usability evaluation. It should include the survey prompt, a cleaned response summary, requirement-implication notes, the usability questionnaire, and a cleaned summary table derived from the pilot responses.

Suggested insertions:
- Survey prompt and question list from `docs/planning/google_form_usability_questions_final.md`
- Usability pilot setup from `docs/planning/usability_pilot_plan.md`
- Cleaned requirements-summary table derived from the draft survey material
- Cleaned usability-response summary derived from `userbillaty.xlsx`

## Appendix B. Evolution of System Architecture (Draft vs. Final)

This appendix preserves selected architecture artefacts from the draft stage, including earlier OCR-heavy pipeline concepts, broader orchestration sketches, and early interaction flows. These materials are retained to document the design transition toward the final preview-confirm, retrieval-first, and vision-dominant architecture described in Chapter 3.

Suggested insertions:
- Early architecture figures exported from `submission/reports/Draft.docx`
- Current architecture diagrams from `docs/architecture/ARCHITECTURE_MERMAID.md`
- Pipeline summary from `docs/architecture/hybrid_pipeline_overview_ko.md`
- Retrieval-iteration planning evidence from `docs/planning/prd_qwen3_vl_retrieval_v1_to_v5.md`

## Appendix C. User Interface Iterations

This appendix contains supplementary interface screenshots that are too detailed for the core report or the final demonstration video. It may combine final web screenshots with a small number of earlier prototype screens in order to show how the workflow evolved into the current login, search, indexing, catalog, and agent experience.

**Figure C1. Final login page of the Smart Vision web prototype.**  
This screen provides the authenticated entry point to the retrieval and indexing workflow used throughout the final user study and video demonstration.  
Source: manual capture needed.

**Figure C2. Final search interface supporting image-assisted retrieval.**  
The page allows the user to submit a query image and optional text, inspect the returned shortlist, and review evidence-backed candidate matches.  
Source: `docs/images/fig_4_2_web_search_ui.png`.

**Figure C3. Indexing interface showing the preview-before-confirm workflow.**  
Uploaded images are used to draft listing metadata, after which the user can review, edit, and confirm the final item record.  
Source: `docs/images/fig_index_ui.png`.

**Figure C4. Catalog interface used to retrieve supporting information from indexed documentation.**  
This page extends the core retrieval workflow by allowing the user to consult internal catalogue-style evidence.  
Source: `docs/images/fig_catalog_ui.png`.

**Figure C5. Agent interface used for conversational evidence gathering.**  
The page illustrates how hybrid retrieval, catalogue lookup, and supporting reasoning can be combined in a single user-facing workflow.  
Source: `docs/images/fig_4_4_agent_chat_ui.png`.

**Figure C6. Bilingual interface example showing the language-toggle support added to the final web prototype.**  
This feature improves accessibility and reflects the practical deployment orientation of the system.  
Source: manual capture needed.

**Figure C7. Earlier prototype interface retained for comparison.**  
This figure is included as an early concept to show how the project progressed from a simpler prototype interaction model toward the final structured web workflow.  
Source: early Gradio screenshot from the draft materials, if available.

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

This appendix extends the qualitative discussion in Chapter 5 by presenting additional OCR and retrieval failure cases. The examples should be captioned briefly to show the failure type, such as irrelevant specification noise, mixed-orientation text, glare, blur, ambiguous labels, or wrong Top-1 but correct Top-5 retrieval.

**Figure D1. Example of irrelevant specification noise in industrial imagery.**  
OCR extracts visible strings such as voltage or resistance values, but these do not uniquely identify the part and can instead distort retrieval ranking.  
Source: failure-case screenshot or material summarised in `experiments/qwen3_vl_1000_sample_final_report_en.md`.

**Figure D2. Mixed-orientation identifier text.**  
This type of layout frequently causes OCR token fragmentation or ordering errors, whereas the vision-language path remains more robust because it reasons over the full label layout.  
Source: manual failure-case screenshot.

**Figure D3. Engraved or low-contrast identifier region.**  
Traditional OCR is brittle under weak contrast and partial wear, which makes this class of example useful for explaining why OCR was demoted from the default retrieval path.  
Source: manual failure-case screenshot.

**Figure D4. Logo-like manufacturer mark that functions as visual evidence rather than clean text.**  
This type of case illustrates the benefit of the vision-language pathway over a purely text-extraction-based approach.  
Source: manual failure-case screenshot.

**Figure D5. Example where the system misses the exact item at rank 1 but still returns the correct match within the shortlist.**  
Such cases support the human-in-the-loop design because they show why a ranked candidate list is more realistic than a single forced prediction.  
Source: result screenshot based on the sampled-evaluation material or `experiments/CURRENT_EXPERIMENT_STATUS.md`.

**Figure D6. Difficult query image with blur, glare, occlusion, or low resolution.**  
This figure is useful for showing the remaining boundary conditions of the prototype even after the shift to a vision-dominant retrieval pipeline.  
Source: manual failure-case screenshot.

## Appendix E. Project Management and Risk Assessment

This appendix contains the project-management and risk-assessment material developed during the earlier planning stages. Suitable contents include milestone structures, work-breakdown notes, planning screenshots, sprint summaries, risk tables, and impact-mitigation notes.

Suggested insertions:
- Work-breakdown structure and milestone plan
- Risk table and impact-mitigation summary
- Planning or Kanban screenshots
- Sprint or revision summaries from the draft stage

## Appendix F. Evaluation Data and Extended Results

This appendix serves as the extended technical appendix for Chapters 4 and 5. It should contain longer result tables, protocol notes, split descriptions, schema material, usability raw summaries, and API or response-structure summaries that would otherwise overload the main report.

Suggested insertions:
- Extended benchmark tables from `experiments/qwen3_vl_1000_sample_final_report_en.md`
- Current experiment summary from `experiments/CURRENT_EXPERIMENT_STATUS.md`
- Retrieval-evaluation input description from `docs/planning/retrieval_eval_inputs.md`
- Dataset split/schema summaries from `docs/planning/unified_dataset_schema_and_split.md`
- Usability raw-summary table derived from `userbillaty.xlsx`
- API/schema summary tables derived from `apps/api/smart_vision_api/api/v1/hybrid.py`, `apps/api/smart_vision_api/services/hybrid.py`, and `apps/api/smart_vision_api/schemas/payload.py`
