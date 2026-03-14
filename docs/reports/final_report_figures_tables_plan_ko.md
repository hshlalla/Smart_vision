# Final Report Figures and Tables Plan (한국어 버전)

이 문서는 `docs/reports/final_report_figures_tables_plan.md`의 한국어 대응본이다.  
최종보고서 본문에 넣을 표와 그림의 추천 배치, 캡션, 근거 출처를 한국어로 빠르게 확인할 수 있도록 정리했다.  
기준 본문은 `docs/reports/final_report_docx_ready.md`이며, 한국어 참조본은 `docs/reports/final_report_docx_ready_ko.md`다.

## 1. Usage Rule

- 먼저 `final_report_docx_ready.md`의 `[Insert ... here]` 위치를 기준으로 배치한다.
- 가능하면 기존 제출본 시각자료를 재사용하되, 현재 코드와 맞지 않는 설명은 캡션에서 바로잡는다.
- 새로운 스크린샷이 필요하면 React 웹 UI 기준으로 다시 캡처한다.
- Gradio는 메인 사용자 인터페이스가 아니라 보조 디버그 인터페이스로만 다룬다.

## 2. Recommended Tables

### Table 1-1. 설문 결과와 요구사항 함의

- **배치 위치**: Introduction에서 domain/user requirements 설명 직후
- **권장 캡션**: 사용자 설문에서 도출된 주요 관찰과 시스템 설계에 대한 함의를 요약한 표
- **핵심 메시지**: 문제 정의가 단순 아이디어가 아니라 사용자·도메인 근거를 가진다는 점을 보여줌

### Table 3-1. 요구사항-컴포넌트 대응표

- **배치 위치**: Design 장 초반 architecture overview 직후
- **권장 캡션**: domain/user requirements가 어떤 architectural component로 대응되는지 보여주는 traceability 표
- **핵심 메시지**: 설계가 임의적이 아니라 requirement-driven임을 보여줌

### Table 5-1. 평가 개요와 근거 상태

- **배치 위치**: Evaluation 장 시작부
- **권장 캡션**: 각 평가 축별로 이미 확보된 근거, partial evidence, future work를 구분한 표
- **핵심 메시지**: 이미 수행된 것과 아직 진행 중인 것을 정직하게 분리함

### Table 5-2. Image-only baseline retrieval 결과

- **배치 위치**: retrieval 결과 설명 직후
- **권장 캡션**: image-only baseline의 정량 또는 반정량 결과를 요약한 표
- **핵심 메시지**: vision-only 접근의 가능성과 한계를 동시에 보여줌

### Table 5-3. Objective status and critical assessment

- **배치 위치**: Evaluation 마지막 부분
- **권장 캡션**: objective별 현재 상태, 근거, 남은 gap을 요약한 표
- **핵심 메시지**: implementation과 evaluation의 경계를 솔직하게 드러냄

## 3. Recommended Figures

### Figure 3-1. Overall system architecture

- **배치 위치**: Design 장의 system overview 직후
- **권장 캡션**: web layer, API layer, hybrid-search orchestrator, multimodal component, retrieval store를 보여주는 전체 구조도
- **소스**: `docs/architecture/ARCHITECTURE_MERMAID.md`

### Figure 3-2. Query-time hybrid search flow

- **배치 위치**: query-time workflow 설명 직후
- **권장 캡션**: user image와 optional text query가 OCR, embedding, candidate fusion, Top-K 결과로 이어지는 hybrid retrieval 흐름도
- **핵심 메시지**: retrieval-first + fallback 구조를 시각적으로 보여줌

### Figure 3-3. Agent and catalog orchestration path

- **배치 위치**: agent / catalog 설명 직후
- **권장 캡션**: agent가 hybrid search, catalog search, 기타 도구를 어떻게 orchestration하고 source evidence를 보존하는지 보여주는 그림
- **핵심 메시지**: 단순 검색이 아니라 orchestrated workflow라는 점을 강조함

### Figure 4-1. Repository or module overview

- **배치 위치**: Implementation 장 초반
- **권장 캡션**: web, API, model, docs, submission evidence가 분리된 high-level repository structure
- **핵심 메시지**: 시스템 구성이 계층적으로 분리되어 있음을 보여줌

### Figure 4-2. Web search UI

- **배치 위치**: frontend 설명 직후
- **권장 캡션**: 이미지 업로드, 텍스트 입력, Top-K 결과가 보이는 search interface 스크린샷
- **핵심 메시지**: 사용자 관점에서 실제 워크플로우를 보여줌

### Figure 4-3. Hybrid score decomposition example

- **배치 위치**: hybrid ranking 설명 직후
- **권장 캡션**: image, OCR, caption, lexical, spec-aware signal 등 복합 점수 근거를 보여주는 예시
- **핵심 메시지**: 결과가 black box가 아니라 evidence-aware ranking이라는 점을 전달함

### Figure 4-4. Agent UI with writeback toggle

- **배치 위치**: agent / safety 설명 직후
- **권장 캡션**: source-backed answer와 explicit writeback opt-in control이 보이는 agent UI 스크린샷
- **핵심 메시지**: human review와 safer writeback policy를 시각적으로 보여줌

### Figure 5-1. OCR and retrieval failure examples

- **배치 위치**: Evaluation 장의 failure analysis 파트
- **권장 캡션**: blur, glare, small text, stylised label, visually similar variant 등 대표적 실패 사례
- **핵심 메시지**: 왜 OCR을 uncertain evidence로 다뤄야 하는지 설명함

## 4. Optional Extra Visuals

### Optional Figure A. Latency instrumentation breakdown

- **언제 쓰는가**: p50/p90/p95는 아직 없어도 instrumentation 자체를 보여주고 싶을 때
- **권장 캡션**: hybrid search path에서 현재 계측되는 latency stage를 보여주는 그림

### Optional Figure B. Artifact evidence bundle structure

- **언제 쓰는가**: appendix나 reproducibility section을 강화하고 싶을 때
- **권장 캡션**: 구현·회귀 검증 주장을 지원하는 artifact bundle 구조도

## 5. Caption Writing Rules

- 캡션은 “무엇을 보여주는지”뿐 아니라 “왜 중요한지”가 드러나도록 쓴다.
- 현재 증거 수준을 넘는 표현은 피한다.
- `conducted`, `fully validated` 같은 표현은 실제 근거가 있을 때만 쓴다.
- 가능하면 figure/table이 본문 주장과 직접 연결되도록 만든다.

## 6. Final Assembly Order

1. `final_report_docx_ready.md`의 insertion marker 위치를 기준으로 figure/table을 고른다.
2. 기존 Draft 시각자료 중 재사용 가능한 것을 선별한다.
3. 현재 코드와 어긋나는 설명은 캡션에서 수정한다.
4. 부족한 UI 스크린샷은 새로 캡처한다.
5. Evaluation 표는 supported / partial / future work 구분이 분명하게 보이도록 조립한다.
