# Final Report Status

이 문서는 현재 `docs/reports/`에 남겨둔 최종보고서 문서 묶음의 역할과 반영 규칙을 정리한 상태 문서다.

## Canonical Report Inputs

최종보고서 작성 시 아래 순서를 기준으로 사용한다.

1. 제출 기준/루브릭
   - `submission/guides/final_report_guide.md`
   - `submission/guides/final_report_rubric_v2 (1).pdf`
2. 기존 제출본과 피드백
   - `submission/reports/Draft.docx`
   - `submission/feedback/preliminary_feedback.md`
   - `submission/feedback/draft_feedback.md`
3. 최종 실험 보고서
   - `experiments/qwen3_vl_1000_sample_final_report_ko.md`
   - `experiments/qwen3_vl_1000_sample_final_report_en.md`
4. 현재 보고서 원고
   - `docs/reports/final_report_docx_ready.md`
   - `docs/reports/final_report_docx_ready_ko.md`
   - `docs/reports/final_report/`
5. 조립 보조 문서
   - `docs/reports/final_report_figures_tables_plan.md`
   - `docs/reports/final_report_figures_tables_plan_ko.md`
   - `docs/reports/final_report_revision_checklist.md`

## Final Narrative Decisions

최종 원고는 아래 규칙으로 통일한다.

- `1000-item`, `900/100 split`, `OCR benchmark`, `GPU execution environment`는 최종 실험 보고서 기준 서술을 유지한다.
- `C4`의 본 실험 수치는 최종 실험 보고서 기준 값을 사용한다.
  - `Accuracy@1 = 0.91`
  - `Accuracy@5 = 0.97`
  - `MRR = 0.939`
  - `Exact identifier hit = 0.88`
- `C2`의 본 실험 수치도 최종 실험 보고서 기준 값을 사용한다.
  - `Accuracy@1 = 0.86`
  - `Accuracy@5 = 0.95`
  - `MRR = 0.903`
  - `Exact identifier hit = 0.81`
- OCR identifier benchmark는 최종 실험 보고서 기준 표를 사용한다.
- 최종 운영 권고 구성은 `C3 (OCR off, reranker off)`로 정리한다.

## Local Supplementary Evidence Rule

로컬 추가 검증 결과는 `최종 권고 구성`을 뒷받침하는 supporting evidence로만 사용한다.

사용 가능한 로컬 근거:

- `C3`
  - group `Hit@1 = 1.0`
  - group `Hit@5 = 1.0`
  - `MRR = 1.0`
  - exact `item_id@1 = 0.9667`
  - warm mean total `731.13 ms`
- `C1`
  - retrieval quality 이득 없음
  - warm mean total `89337.71 ms`
  - preprocessing mean `23790.06 ms`
  - finalize mean `65199.13 ms`

이 로컬 결과는 `C4` 본 실험 수치를 덮어쓰는 용도가 아니라, `reranker`의 운영 비용과 `C3` 권고 결론을 설명하는 보조 근거로만 사용한다.

## What Was Consolidated

보고서 폴더는 최종 제출용 문서 위주로 정리했다.

- 유지:
  - 통합 원고 `final_report_docx_ready*.md`
  - 장별 원고 `final_report/`
  - 그림/표 계획
  - 수정 체크리스트
- 제거:
  - 중간 결과 템플릿
  - 작업용 reference
  - 중복 working draft
  - peer review / prototype / usability 단독 초안

## Current Writing Roles

- `final_report_docx_ready.md`
  - 제출용 DOCX로 옮기기 가장 쉬운 영문 통합 원고
- `final_report_docx_ready_ko.md`
  - 영문 원고의 한국어 대응본
- `final_report/05_Evaluation.md`
  - 실험 보고서 내용을 장별 초안에 반영한 평가 장
- `final_report_figures_tables_plan*.md`
  - 표/그림 번호와 배치 가이드

## Remaining Editorial Work

문서 구조상 남은 작업은 편집 수준이다.

- DOCX 반영
- 최종 figure/table 삽입
- 참고문헌 정리
- 표/그림 번호 확정
