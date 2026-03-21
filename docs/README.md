# Docs Guide

`docs/`는 현재 유지 중인 내부 문서와 제출용 보고서 작업본을 두는 폴더다.  
제출했던 보고서 원본, 피드백, 루브릭, 제출 증적은 `submission/`에 둔다.

## Structure

- `architecture/`
  - 시스템 구조, 머메이드 다이어그램, 프로젝트 구조 설명
- `planning/`
  - 실행 계획과 backlog
- `reports/`
  - 최종보고서 원고, 장별 초안, 그림/표 조립 가이드
- `release_notes/`
  - 앱/API/모델 릴리즈 노트

실험 러너와 실험 보고서는 `experiments/`에 둔다.

- `experiments/`
  - 재현 가능한 평가 러너
  - 최종 실험 보고서 한글/영문본
  - 현재 실험 상태 요약

## Runtime Source Of Truth

현재 구현 동작을 설명할 때는 아래 문서를 우선 기준으로 본다.

- `README.md`
- `apps/api/README.md`
- `packages/model/README.md`
- `apps/web/README.md`

`docs/reports/*`는 제출용 원고이므로 런타임 설명의 1차 기준으로 쓰지 않는다.
중복 재등록, preview-confirm 인덱싱, merge 정책처럼 실제 동작에 직접 영향을 주는 내용도 위 README 계층을 우선 기준으로 본다.

## Report Writing Rule

최종보고서를 쓸 때는 아래 순서를 기준으로 본다.

1. 제출 기준/루브릭
   - `submission/guides/final_report_guide.md`
   - `submission/guides/final_report_rubric_v2 (1).pdf`
2. 이전 제출본과 피드백
   - `submission/reports/preliminary.pdf`
   - `submission/reports/Draft.docx`
   - `submission/feedback/preliminary_feedback.md`
   - `submission/feedback/draft_feedback.md`
3. 최종 실험 보고서와 반영 규칙
   - `experiments/qwen3_vl_1000_sample_final_report_ko.md`
   - `experiments/qwen3_vl_1000_sample_final_report_en.md`
   - `experiments/CURRENT_EXPERIMENT_STATUS.md`
4. 최종 원고와 조립 문서
   - `docs/reports/final_report_docx_ready.md`
   - `docs/reports/final_report_docx_ready_ko.md`
   - `docs/reports/final_report_status.md`
   - `docs/reports/final_report_figures_tables_plan.md`
   - `docs/reports/final_report_figures_tables_plan_ko.md`
5. 재현 가능한 증적
   - `submission/evidence/report_support_2026-03-10/`

## Report Document Meaning

- `reports/final_report_status.md`
  - 현재 남겨둔 보고서 문서 묶음과 실험 반영 규칙을 요약한 상태 문서다.
  - 장별 초안과 통합 원고를 어떤 순서로 쓸지 정리한다.
- `reports/final_report_docx_ready.md`
  - 제출용 DOCX에 옮기기 쉽게 정리한 영문 통합 원고다.
- `reports/final_report_docx_ready_ko.md`
  - 영문 원고의 한국어 대응본이다.
- `reports/final_report/`
  - 장별로 나눈 한국어 작업 원고다.
- `reports/final_report_figures_tables_plan.md`
  - 최종보고서에 넣을 표/그림의 번호, 캡션, 배치 위치, 근거 출처를 정리한 조립 가이드다.
- `reports/final_report_figures_tables_plan_ko.md`
  - 위 문서의 한국어 대응본이다.
- `reports/final_report_revision_checklist.md`
  - 문장 수위, claim 수준, DOCX 반영 전 점검 항목을 정리한 체크리스트다.
