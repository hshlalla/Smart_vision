# Final Report Figures and Tables Plan

이 문서는 최종보고서 본문에 넣을 표와 그림의 추천 배치, 캡션, 근거 출처를 정리한 조립 가이드다.  
기준 본문은 `docs/reports/final_report_docx_ready.md`이다.

## 1. Usage Rule

- 먼저 `final_report_docx_ready.md`의 `[Insert ... here]` 위치를 기준으로 배치한다.
- 가능한 한 기존 제출본의 시각자료를 재사용하되, 현재 코드와 맞지 않는 설명은 캡션에서 바로잡는다.
- 새로운 스크린샷이 필요하면 React 웹 UI 기준으로 다시 캡처한다.
- Gradio는 보조 디버그 인터페이스로만 배치한다.

## 2. Recommended Tables

### Table 1-1. Survey findings and requirement implications

`Placement`  
Introduction 1.1-1.3 뒤, Domain/User requirements 설명 직후

`Recommended caption`  
`Table 1-1. Summary of requirements elicitation findings and their implications for system design.`

`Suggested columns`
- Survey finding
- Evidence or response count
- Requirement implication

`Content source`
- `submission/reports/Draft.docx`의 survey table
- `Appendix C` raw responses

`Key message`
- 사용자는 사진 기반 도움에는 열려 있지만 무조건 신뢰하지는 않음
- 따라서 Top-K, evidence, editability가 필요함

### Table 3-1. Requirement-to-component traceability

`Placement`  
Design 3.3 직전 또는 직후

`Recommended caption`  
`Table 3-1. Mapping from domain and user requirements to the main architectural components.`

`Suggested columns`
- Requirement
- Design response
- Main component
- Current status

`Content source`
- `docs/reports/final_report_status.md`
- `docs/reports/final_report_revision_checklist.md`

`Key message`
- R1-R4, U1-U3가 구현 요소와 직접 연결됨을 보여줌

### Table 5-1. Evaluation overview and evidence status

`Placement`  
Evaluation 5.1 시작 직후

`Recommended caption`  
`Table 5-1. Evaluation goals, metrics, current evidence status, and remaining gaps.`

`Suggested columns`
- Evaluation area
- Metric / evidence
- Why it matters
- Current status

`Suggested rows`
- Retrieval effectiveness / Accuracy@1, Accuracy@5 / supported
- OCR robustness / identifier benchmark + qualitative failure analysis / supported
- Latency / benchmark latency + local supplementary latency / supported
- Engineering reliability / pytest regression evidence / supported
- User usefulness / workflow readiness + pilot protocol / partial

`Content source`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`
- `docs/reports/final_report_status.md`

### Table 5-2. Image-only baseline retrieval results

`Placement`  
Evaluation 5.3 본문 바로 앞

`Recommended caption`  
`Table 5-2. Image-only retrieval baseline on the two evaluation splits used in the draft-stage experiments.`

`Suggested columns`
- Dataset
- Accuracy@1
- Accuracy@5

`Content source`
- `submission/reports/Draft.docx`

`Values`
- Random 1000 models / 0.287 / 0.791
- Category-sampled 500 models / 0.306 / 0.812

### Table 5-3. Objective status and critical assessment

`Placement`  
Evaluation 마지막 부분, critical discussion 직전 또는 직후

`Recommended caption`  
`Table 5-3. Objective-by-objective assessment based on the current implementation and available evidence.`

`Suggested columns`
- Objective
- Current status
- Evidence
- Remaining gap

`Content source`
- `docs/reports/final_report_status.md`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`

## 3. Recommended Figures

### Figure 3-1. Overall system architecture

`Placement`  
Design 3.1-3.2 설명 직후

`Recommended caption`  
`Figure 3-1. Overall architecture of the Smart Image Part Identifier, showing the web layer, API layer, hybrid-search orchestrator, multimodal processing components, and retrieval stores.`

`Source`
- `docs/architecture/ARCHITECTURE_MERMAID.md`

`Preparation note`
- Mermaid 다이어그램을 정리해서 export
- React + FastAPI가 메인 경로로 보이도록 수정 여부 확인

### Figure 3-2. Query-time hybrid search flow

`Placement`  
Design 3.4.2 설명 직후

`Recommended caption`  
`Figure 3-2. Query-time hybrid retrieval flow from user image and optional text through OCR, embedding generation, Milvus search, candidate fusion, and Top-K result construction.`

`Source`
- `docs/architecture/ARCHITECTURE_MERMAID.md`
- 필요 시 기존 Draft의 pipeline figure 재사용

`Key message`
- retrieval-first + fallback 구조를 보여줌

### Figure 3-3. Agent and catalog orchestration path

`Placement`  
Design 3.4.4 설명 직후

`Recommended caption`  
`Figure 3-3. Tool orchestration path showing how the agent can call hybrid search, catalog search, and external tools while preserving evidence sources.`

`Source`
- `docs/reports/final_report_status.md`의 tool summary
- 필요 시 Mermaid 추가 제작

`Key message`
- 단순 검색 엔진이 아니라 orchestrated workflow임을 보여줌

### Figure 4-1. Repository or module overview

`Placement`  
Implementation 4.1 직후

`Recommended caption`  
`Figure 4-1. High-level repository structure separating web, API, model, documentation, and submission evidence layers.`

`Source`
- `readme.md`
- `docs/architecture/PROJECT_STRUCTURE.md`

`Alternative`
- 코드 트리 그림 대신 architecture block diagram으로 대체 가능

### Figure 4-2. Web search UI

`Placement`  
Implementation 4.2 직후

`Recommended caption`  
`Figure 4-2. Search interface in the React web application, showing image upload, query input, and Top-K retrieval output.`

`Source`
- 현재 `apps/web` 실행 후 새 스크린샷 캡처

`Preparation note`
- base64/개발 로그가 아니라 사용자 화면 중심으로 캡처

### Figure 4-3. Hybrid score decomposition example

`Placement`  
Implementation 4.3-4.4 직후

`Recommended caption`  
`Figure 4-3. Example of hybrid result evidence, including decomposed scores such as image, OCR, caption, lexical, and specification-aware signals.`

`Source`
- 검색 결과 JSON 예시 또는 UI 카드
- `packages/model/smart_match/hybrid_search_pipeline/hybrid_pipeline_runner.py`

`Key message`
- 결과가 black box가 아니라 evidence-aware ranking이라는 점 강조

### Figure 4-4. Agent UI with writeback toggle

`Placement`  
Implementation 4.6 직후

`Recommended caption`  
`Figure 4-4. Agent chat interface showing source-backed answers and the explicit opt-in control for Milvus writeback.`

`Source`
- 현재 `apps/web/src/views/AgentChatPage.tsx` 반영 후 UI 스크린샷

`Key message`
- human review and safer writeback policy를 시각적으로 보여줌

### Figure 5-1. OCR and retrieval failure examples

`Placement`  
Evaluation 5.4 또는 5.4-5.5 사이

`Recommended caption`  
`Figure 5-1. Representative OCR and retrieval failure cases observed during prototype evaluation, including blur, glare, small text, stylised labels, and visually similar variants.`

`Source`
- 기존 Draft figure 재사용 권장
- 필요 시 Appendix E 자료 재조합

`Key message`
- 왜 OCR을 uncertain evidence로 다뤄야 하는지 설명

## 4. Optional Extra Visuals

필수는 아니지만 여유가 있으면 아래 중 하나를 추가할 수 있다.

### Optional Figure A. Latency instrumentation breakdown

`Recommended caption`  
`Figure A. Stages currently instrumented for latency measurement in the hybrid search path.`

`Use only if`
- p50/p90/p95 결과는 아직 없지만 instrumentation 자체를 시각적으로 보여주고 싶을 때

### Optional Figure B. Artifact evidence bundle structure

`Recommended caption`  
`Figure B. Structure of the validation artifact bundle used to support implementation and regression claims in the final report.`

`Use only if`
- Appendix 또는 reproducibility section을 강화하고 싶을 때

## 5. Caption Writing Rules

- 캡션은 단순 이름이 아니라 "무엇을 보여주고 왜 중요한지"를 한 문장으로 쓴다.
- 본문에서 반드시 각 표/그림을 직접 언급한다.
- React UI와 Gradio를 혼동하지 않도록 캡션에 역할을 명확히 적는다.
- 아직 측정되지 않은 수치는 캡션에서 암시하지 않는다.

## 6. Final Assembly Order

1. `final_report_docx_ready.md`를 본문 기준으로 사용
2. 이 문서의 Table/Figure 번호대로 배치
3. 기존 Draft 시각자료 중 재사용 가능한 것 선별
4. 부족한 UI 스크린샷만 새로 캡처
5. 본문의 `[Insert ... here]` 마커 제거
