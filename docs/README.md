# Docs Guide

`docs/`는 현재 작업 중인 내부 문서만 두는 폴더다.  
제출했던 보고서 원본, 피드백, 루브릭, 제출 증적은 `submission/`에 둔다.

## Structure

- `architecture/`
  - 시스템 구조, 머메이드 다이어그램, 프로젝트 구조 설명
- `planning/`
  - 현재 실행 계획과 backlog
- `reports/`
  - 최종보고서 작성용 내부 메모와 과거 작업용 보고서 reference
- `release_notes/`
  - 앱/API/모델 릴리즈 노트

## Runtime Source Of Truth

현재 구현 동작을 설명할 때는 아래 문서를 우선 기준으로 본다.

- `README.md`
- `apps/api/README.md`
- `packages/model/README.md`
- `apps/web/README.md`

`docs/reports/*`는 제출용 작업 문서이므로 런타임 설명의 1차 기준으로 쓰지 않는다.
중복 재등록, preview-confirm 인덱싱, merge 정책처럼 실제 동작에 직접 영향을 주는 내용도 위 README 계층을 우선 기준으로 본다.

## Historical Reference Boundary

아래 문서들은 여전히 참고 가치가 있지만, 현재 런타임의 최신 동작을 강제하는 source of truth는 아니다.

- `docs/reports/report_working_reference.md`
- `docs/reports/report_working_reference_ko.md`
- 일부 `docs/planning/*`

이 문서들에 과거 설계 대안이나 넓은 범위의 아이디어가 남아 있더라도, 현재 동작이 다르게 구현되어 있다면 런타임 README와 최신 release note를 우선한다.

## Report Writing Rule

최종보고서를 쓸 때는 아래 순서를 기준으로 본다.

1. 제출 기준/루브릭:
   - `submission/guides/final_report_guide.md`
   - `submission/guides/final_report_rubric_v2 (1).pdf`
2. 이전 제출본과 피드백:
   - `submission/reports/preliminary.pdf`
   - `submission/reports/Draft.docx`
   - `submission/feedback/preliminary_feedback.md`
   - `submission/feedback/draft_feedback.md`
3. 현재 구현 상태와 반영 포인트:
   - `docs/reports/final_report_status.md`
   - `docs/planning/to_do_list.md`
4. 재현 가능한 증적:
   - `submission/evidence/report_support_2026-03-10/`

## Report Document Meaning

- `reports/report_working_reference.md`
  - 과거에 정리한 긴 작업용 보고서 reference.
  - 현재 제출 파일 자체는 아니며, 문장/구성 참고용이다.
- `reports/final_report_status.md`
  - 현재 코드베이스와 제출 피드백을 기준으로 최종보고서에 반영할 상태/근거를 정리한 문서.
  - 최종보고서 작성 시 가장 먼저 확인할 작업본이다.
- `reports/final_report_revision_checklist.md`
  - `Draft.docx`에서 무엇을 고쳐야 하는지, 어떤 주장을 낮춰야 하는지, 어떤 근거를 연결해야 하는지 정리한 최종 수정 체크리스트.
  - 실제 최종본 수정 작업에 바로 쓰는 실행 문서다.
- `reports/final_report_working_draft.md`
  - `Draft.docx`를 현재 코드/피드백/아티팩트 기준으로 다시 쓴 최종보고서 본문 초안.
  - 실제 문장 단위 수정이나 DOCX 반영 전에 보는 기준 원고다.
- `reports/final_report_docx_ready.md`
  - 제출용 DOCX에 옮기기 쉽게 문체와 구조를 다듬은 본문 원고.
  - 그림/표 삽입 위치 마커가 포함되어 있다.
- `reports/final_report_figures_tables_plan.md`
  - 최종보고서에 넣을 표/그림의 번호, 캡션, 배치 위치, 근거 출처를 정리한 조립 가이드다.
- `planning/to_do_list.md`
  - 구현 및 평가 backlog.
  - 보고서에 아직 쓰면 안 되는 항목과 앞으로 해야 할 항목을 구분하는 기준으로 본다.
