# Final Report Revision Checklist

이 문서는 최종보고서 편집 직전 확인할 항목만 남긴 실행용 체크리스트다.

## 1. Canonical Inputs

최종 원고 편집 시 아래 문서를 기준으로 사용한다.

1. `submission/reports/Draft.docx`
2. `submission/feedback/preliminary_feedback.md`
3. `submission/feedback/draft_feedback.md`
4. `experiments/qwen3_vl_1000_sample_final_report_en.md`
5. `experiments/qwen3_vl_1000_sample_final_report_ko.md`
6. `docs/reports/final_report_docx_ready.md`
7. `docs/reports/final_report_docx_ready_ko.md`
8. `docs/reports/final_report_figures_tables_plan.md`
9. `docs/reports/final_report_figures_tables_plan_ko.md`

## 2. Claim Consistency Checks

- `1000-item`, `900/100 split`, `OCR benchmark`, `GPU runtime`는 최종 실험 보고서 기준으로 유지했는가
- `C4` 수치는 최종 실험 보고서 값으로 유지했는가
- 최종 운영 권고 구성은 `C3 (OCR off, reranker off)`로 통일했는가
- `C1`과 `C3`의 로컬 추가 검증은 supporting evidence로만 사용했는가
- OCR은 기본 retrieval engine이 아니라 verification/evidence signal로 설명했는가
- 시스템을 `retrieval-first, human-in-the-loop identification assistant`로 일관되게 설명했는가

## 3. Wording Checks

- `fully autonomous identifier`처럼 과한 표현을 쓰지 않았는가
- `implemented`, `evaluated`, `recommended`가 실제 문서 근거와 맞는가
- `in progress`, `pending` 표현이 최종 실험 보고서와 충돌하지 않는가
- `C4`가 main benchmark winner이고 `C3`가 final recommended operating configuration이라는 구분이 분명한가

## 4. Structure Checks

- Evaluation 장이 `end-to-end`, `retrieval`, `OCR`, `latency`, `reliability`, `usability readiness` 순서로 정리되었는가
- Figure/Table insertion marker가 최종 조립 계획과 맞는가
- 장별 원고 `final_report/`와 통합 원고 `final_report_docx_ready*.md`가 같은 수치와 결론을 바라보는가

## 5. Final Assembly Checks

- 그림/표 번호와 캡션을 모두 채웠는가
- 참고문헌 형식을 `Draft.docx` 기준으로 정리했는가
- 내부 메모, 작업 노트, 임시 annotation을 제거했는가
- 제출 직전 영문 원고와 한국어 대응본 사이의 수치 불일치가 없는가
