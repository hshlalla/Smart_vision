# Final Report Revision Checklist

이 문서는 최종보고서 작성 시 사용할 실무 체크리스트다.  
기준선은 `submission/reports/Draft.docx`이며, 두 차례 피드백과 현재 코드/아티팩트를 기준으로 수정 범위를 정리한다.

## 1. Canonical Inputs

최종보고서 정리 시 아래 순서를 기준으로 사용한다.

1. `submission/reports/Draft.docx`
2. `submission/feedback/preliminary_feedback.md`
3. `submission/feedback/draft_feedback.md`
4. `submission/guides/final_report_guide.md`
5. `docs/reports/final_report_status.md`
6. `submission/evidence/report_support_2026-03-10/`

## 2. What Must Be Fixed First

### 2.1 High-risk claim corrections

아래 표현은 현재 코드/증적 기준으로 그대로 두면 위험하다.

- `accept/edit` UI가 이미 구현되었다는 단정 표현
- `Gradio UI layer`가 메인 사용자 UI처럼 읽히는 표현
- `CER benchmarking (Conducted)`처럼 정량 결과가 이미 완료된 듯한 제목
- `Latency Evaluation (Conducted)`처럼 p50/p90/p95 결과가 이미 확보된 듯한 제목
- `Hybrid Retrieval Ablation (Conducted)`처럼 실험 자동화와 표가 이미 준비된 듯한 표현
- seeded split, reproducible eval scripts, usability evaluation이 완료되었다는 단정 표현

### 2.2 Safe replacement rule

근거가 아직 부분적이면 아래 표현으로 낮춘다.

- `implemented`
  실제 코드와 테스트가 있으면 유지
- `partially implemented`
  코드 경로는 있으나 정식 UI/실험/운영 플로우가 없으면 사용
- `protocol defined`
  측정 항목과 절차만 정해졌으면 사용
- `instrumentation added`
  로그/계측만 들어갔고 수치 집계는 없으면 사용
- `planned next step`
  아직 코드와 데이터가 없으면 사용

## 3. Feedback-to-Action Mapping

### 3.1 Preliminary feedback 반영 포인트

- 사용자/도메인 요구사항 전용 소절은 유지한다.
- 문헌/설문 근거를 더 직접적으로 시스템 요구사항과 연결한다.
- Design 장에서 bullet 위주 서술을 줄이고 문단 설명을 늘린다.
- Evaluation strategy를 Design 장 내부에서 먼저 정의하고, Evaluation 장에서 같은 기준으로 실제 평가한다.

### 3.2 Draft feedback 반영 포인트

- 도메인/사용자 정당화는 아직 더 강화할 수 있으므로, survey 결과를 기능 요구사항과 직접 매핑한다.
- 다이어그램은 반드시 본문에서 설명한다.
- originality는 단순 멀티모달 조합이 아니라 `retrieval-first + human review + catalog evidence + agent orchestration`의 시스템 수준 기여로 설명한다.
- Evaluation 장은 결과 나열보다 “왜 이 지표가 목표에 맞는가”를 먼저 설명한다.

## 4. Chapter-by-Chapter Revision Checklist

### 4.1 Introduction

- 템플릿명과 프로젝트 번호를 첫 부분에서 명확히 유지한다.
- secondhand parts가 일반 상품보다 어려운 이유를 한 문단으로 압축해 제시한다.
- Domain requirements와 User requirements를 너무 길게 반복하지 말고, 핵심 제약 3-4개로 정리한다.
- Aim/Objectives는 Evaluation에서 실제 측정 가능한 표현만 남긴다.
- `Top-5 shortlist + structured listing summary + evidence`를 핵심 산출물로 고정한다.

### 4.2 Literature Review

- 현재 강점이므로 구조는 유지하되, 각 절 마지막 문장을 `그래서 내 설계는 무엇을 채택했는가`로 통일한다.
- consumer visual search와 industrial parts literature를 대비시키는 문장을 강화한다.
- feedback, OCR uncertainty, hybrid retrieval의 필요성을 literature gap으로 다시 묶는다.
- 참고문헌 스타일은 마지막까지 동일하게 맞춘다.

### 4.3 Design

- bullet 나열 대신 시스템 흐름을 문단으로 설명한다.
- 아키텍처 다이어그램, 검색 파이프라인, agent 흐름도를 본문에서 각각 해설한다.
- Evaluation strategy를 Design 장 후반에 별도 소절로 넣는다.
- FR/NFR 또는 objective별로 어떤 컴포넌트가 대응하는지 표로 정리한다.
- `human-in-the-loop`는 현재 완전한 승인 UI가 아니라 “설계 원칙 + 일부 구현 + 남은 작업”으로 서술한다.

### 4.4 Implementation

- 실제 구현 축을 네 부분으로 압축한다.
- `web frontend`
- `api layer`
- `hybrid search core`
- `catalog RAG + agent orchestration`
- 코드 설명은 “무슨 파일을 만들었는지”보다 “어떤 알고리즘/정책이 동작하는지” 중심으로 쓴다.
- 이번 패치 내용은 반드시 반영한다.
- `update_milvus=false` 기본 안전화
- agent UI 토글
- hybrid search 단계별 latency instrumentation
- model package lazy import로 pytest collection 안정화
- `Gradio`는 디버그/보조 인터페이스로만 위치를 낮추고, 메인 UX는 `React + FastAPI` 기준으로 정리한다.

### 4.5 Evaluation

- 목표별로 평가를 나눈다.
- retrieval effectiveness
- OCR robustness
- latency / interactivity
- regression / implementation stability
- 현재 증적으로 바로 쓸 수 있는 결과:
- API pytest pass
- model pytest pass
- hybrid search에 단계별 latency instrumentation 구현
- 문서에 기록된 image-only baseline/qualitative observations
- 현재는 “방법과 계획만 있는 것”:
- CER aggregate benchmark
- fixed split retrieval benchmark automation
- p50/p90/p95 latency summary
- usability study rerun
- `Conducted` 대신 아래처럼 바꾼다.
- `Evaluation protocol defined`
- `Initial regression evidence`
- `Instrumentation implemented`
- `Quantitative benchmark in progress`
- 결과 표는 `evidence already collected`와 `planned quantitative study`를 분리한다.

### 4.6 Conclusion

- 시스템이 “automatic final identifier”가 아니라 “identification assistant”임을 분명히 한다.
- strongest contribution은 end-to-end orchestration과 uncertainty-aware retrieval임을 강조한다.
- limitations는 숨기지 말고 적는다.
- OCR noise
- open-world catalog coverage
- missing full accept/edit workflow
- incomplete quantitative evaluation automation
- future work는 `to_do_list`의 high-priority 항목과 맞춘다.

## 5. Claim-to-Evidence Map

| Report claim | Status | Evidence to cite |
|---|---|---|
| End-to-end prototype exists | supported | `docs/reports/final_report_status.md`, app/api/model code structure |
| Hybrid retrieval combines image/OCR/text/caption/spec signals | supported | `packages/model/smart_match/hybrid_search_pipeline/hybrid_pipeline_runner.py`, status doc |
| Catalog RAG is implemented | supported | catalog routes and `docs/reports/final_report_status.md` |
| Agent can orchestrate tools and expose sources | supported | agent API/UI, status doc |
| Unsafe automatic writeback was mitigated | supported | `apps/api/smart_vision_api/schemas/agent.py`, `apps/web/src/views/AgentChatPage.tsx`, evidence bundle |
| Regression tests pass | supported | `submission/evidence/report_support_2026-03-10/api_pytest_output.txt`, `submission/evidence/report_support_2026-03-10/model_pytest_output.txt` |
| Search latency is measured end-to-end with summary statistics | partial | instrumentation exists; summary metrics still need batch evaluation |
| OCR quality is benchmarked with CER/WER | partial | protocol exists; aggregate result evidence still needed |
| Accept/edit human review flow is implemented in production UI | not yet supported | describe as planned / partial only |
| Hybrid ablation is completed | not yet supported | describe as planned evaluation only |
| Usability evaluation is complete | not yet supported | only keep if new data is collected before submission |

## 6. Search Terms To Fix In Draft

`Draft.docx`에서 아래 표현을 우선 검색해서 수정한다.

- `accept/edit`
- `Gradio`
- `Conducted`
- `CER`
- `Latency Evaluation`
- `Hybrid Retrieval Ablation`
- `reproducible`
- `seeded split`
- `usability`

## 7. Recommended Final Report Positioning

최종보고서의 톤은 아래처럼 잡는 것이 안전하다.

- 본 시스템은 secondhand parts identification을 위한 `retrieval-first decision support prototype`이다.
- 강점은 멀티모달 검색, catalog evidence, tool orchestration, listing-oriented structured output이다.
- 완전 자동 확정 시스템이 아니라 `human-in-the-loop assistant`로 정의한다.
- 평가는 `already measured`, `instrumented`, `protocol defined`, `future work`를 명확히 구분한다.

## 8. Immediate Next Writing Order

1. `Draft.docx`에서 high-risk 표현 교정
2. Design 장에 evaluation strategy 소절 삽입
3. Implementation 장을 현재 코드 구조와 일치하게 재서술
4. Evaluation 장에서 실제 증적 있는 표만 남기고 과장 표현 제거
5. Conclusion을 assistant framing과 limitation 기반으로 재작성
