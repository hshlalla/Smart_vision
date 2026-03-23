# 8. Appendices

이 부록은 본문을 보조하는 자료를 정리하기 위해 포함한다. 특히 초기 draft 이후 설계와 구현이 크게 바뀐 프로젝트이므로, 부록은 초기 자료와 최종 자료를 함께 보존하면서도 최종 본문을 과도하게 길어지지 않게 하는 역할을 한다. 초기 자료는 최종 구현과 다를 경우 `초기 설계안` 또는 `superseded design`으로 명확히 표기한다.

## Appendix A. User Requirements and Survey Evidence

이 부록에는 1장에서 요약한 초기 요구사항 조사(`n = 6`)와 이후 usability pilot의 보조 자료를 함께 정리한다. survey 질문지, 정리된 응답 요약, requirement implication 메모, usability questionnaire summary, 그리고 cleaned response table을 포함한다.

권장 삽입 자료:
- `docs/planning/google_form_usability_questions_final.md` 기반 survey prompt
- `docs/planning/usability_pilot_plan.md` 기반 usability pilot setup
- draft survey 자료를 정리한 requirement summary table
- `userbillaty.xlsx` 기반 cleaned usability summary table

## Appendix B. Evolution of System Architecture (Draft vs. Final)

이 부록은 초기 draft 이후 시스템 구조가 어떻게 바뀌었는지를 보여주기 위해 포함한다. 초기 architecture diagram, OCR-heavy pipeline sketch, broader orchestration concept 등은 여기에서 정리하며, `초기 설계안` 또는 `superseded design`으로 명확히 표시한다.

권장 삽입 자료:
- `submission/reports/Draft.docx`에서 추출한 초기 architecture figure
- `docs/architecture/ARCHITECTURE_MERMAID.md`의 최신 구조도
- `docs/architecture/hybrid_pipeline_overview_ko.md`의 파이프라인 정리 자료
- `docs/planning/prd_qwen3_vl_retrieval_v1_to_v5.md`의 retrieval iteration evidence

## Appendix C. User Interface Iterations

이 부록에는 본문과 영상에서 모두 보여주기 어려운 UI 자료를 정리한다. 로그인 페이지, 검색 페이지, 인덱싱 페이지, Catalog 페이지, Agent 페이지, metadata preview/edit/confirm 흐름, evidence section close-up 등을 포함할 수 있다.

**Figure C1. Final login page of the Smart Vision web prototype.**  
retrieval 및 indexing workflow의 authenticated entry point를 보여주는 최종 로그인 화면.  
자료: 수동 캡처 필요.

**Figure C2. Final search interface supporting image-assisted retrieval.**  
query image와 optional text를 함께 제출하고, returned shortlist와 evidence-backed candidate match를 확인할 수 있는 검색 화면.  
자료: `docs/images/fig_4_2_web_search_ui.png`.

**Figure C3. Indexing interface showing the preview-before-confirm workflow.**  
업로드 이미지로 listing metadata를 초안화하고, 사용자가 review, edit, confirm을 수행하는 indexing 화면.  
자료: `docs/images/fig_index_ui.png`.

**Figure C4. Catalog interface used to retrieve supporting information from indexed documentation.**  
core retrieval workflow를 보완하기 위해 internal catalogue-style evidence를 조회하는 catalog 화면.  
자료: `docs/images/fig_catalog_ui.png`.

**Figure C5. Agent interface used for conversational evidence gathering.**  
hybrid retrieval, catalogue lookup, supporting reasoning이 하나의 user-facing workflow로 결합되는 agent 화면.  
자료: `docs/images/fig_4_4_agent_chat_ui.png`.

**Figure C6. Bilingual interface example showing the language-toggle support added to the final web prototype.**  
언어 전환 기능을 보여주는 화면으로, 실사용 배치 가능성과 접근성 개선을 함께 보여준다.  
자료: 수동 캡처 필요.

**Figure C7. Earlier prototype interface retained for comparison.**  
초기 prototype interaction model에서 현재의 structured web workflow로 어떻게 발전했는지를 보여주기 위한 early concept 화면.  
자료: draft 자료에 남아 있는 Gradio 스크린샷이 있으면 사용.

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

이 부록은 5장에서 일부만 제시한 OCR 실패 사례와 retrieval failure 사례를 더 넉넉하게 정리하는 공간이다. 복잡한 배경, irrelevant specification noise, mixed vertical-horizontal text, glare, blur, occlusion, wrong Top-1 but correct Top-5 사례 등을 짧은 캡션과 함께 제시한다.

**Figure D1. Example of irrelevant specification noise in industrial imagery.**  
OCR이 voltage나 resistance 같은 visible string을 추출하지만, 이러한 값은 부품을 고유하게 식별하지 못하고 retrieval ranking을 왜곡할 수 있음을 보여주는 사례.  
자료: `experiments/qwen3_vl_1000_sample_final_report_en.md` 또는 수동 failure screenshot.

**Figure D2. Mixed-orientation identifier text.**  
mixed vertical-horizontal layout이 OCR token fragmentation이나 ordering error를 일으키는 사례.  
자료: 수동 failure screenshot.

**Figure D3. Engraved or low-contrast identifier region.**  
약한 contrast와 partial wear 조건에서 traditional OCR이 얼마나 brittle한지를 보여주는 사례.  
자료: 수동 failure screenshot.

**Figure D4. Logo-like manufacturer mark that functions as visual evidence rather than clean text.**  
텍스트 추출보다 visual pattern understanding이 더 중요한 manufacturer-mark 사례.  
자료: 수동 failure screenshot.

**Figure D5. Example where the system misses the exact item at rank 1 but still returns the correct match within the shortlist.**  
single forced prediction보다 ranked shortlist가 더 현실적이라는 점을 보여주는 wrong Top-1 but correct Top-5 사례.  
자료: sampled-evaluation result screenshot 또는 `experiments/CURRENT_EXPERIMENT_STATUS.md`.

**Figure D6. Difficult query image with blur, glare, occlusion, or low resolution.**  
vision-dominant retrieval로 전환한 이후에도 남아 있는 boundary condition을 보여주는 난해한 입력 사례.  
자료: 수동 failure screenshot.

## Appendix E. Project Management and Risk Assessment

이 부록은 프로젝트 관리 자료와 risk assessment 자료를 함께 정리하는 공간이다. risk table, impact-mitigation 표, work breakdown structure, milestone plan, sprint summary, planning screenshot 등을 여기에 둘 수 있다.

권장 삽입 자료:
- work breakdown structure
- milestone plan
- risk register table
- planning 또는 Kanban screenshot
- draft 단계 sprint/revision summary

## Appendix F. Evaluation Data and Extended Results

이 부록은 4장과 5장의 확장판으로 사용한다. 더 긴 세부 결과표, 프로토콜 설명, split note, schema 자료, usability raw summary, API 또는 response structure 요약 등을 이곳에 정리한다.

권장 삽입 자료:
- `experiments/qwen3_vl_1000_sample_final_report_en.md` 기반 extended benchmark table
- `experiments/CURRENT_EXPERIMENT_STATUS.md` 기반 current experiment summary
- `docs/planning/retrieval_eval_inputs.md` 기반 evaluation input note
- `docs/planning/unified_dataset_schema_and_split.md` 기반 split/schema summary
- `userbillaty.xlsx` 기반 raw-summary table
- `apps/api/smart_vision_api/api/v1/hybrid.py`, `services/hybrid.py`, `schemas/payload.py` 기반 API/schema summary table
