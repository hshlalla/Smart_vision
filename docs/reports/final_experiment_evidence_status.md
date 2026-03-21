# Final Experiment Evidence Status

이 문서는 최종보고서에서 **무엇을 실제 결과로 써도 되는지**와 **무엇이 아직 준비/계획 단계인지**를 분리하기 위한 기준 문서다.

## 1. Purpose

최종보고서에서는 `구현 완료`, `실험 준비 완료`, `계측 구현`, `프로토콜 정의`, `실험 완료`를 명확히 구분해야 한다.
이 문서의 목적은 실험 계획서와 실제 증적을 연결해, 보고서에서 과장 표현이 들어가지 않도록 하는 것이다.

## 2. Canonical Sources

- 실험 계획 기준: `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- 보고서 상태 기준: `docs/reports/final_report_status.md`
- 최종 원고 기준: `docs/reports/final_report_docx_ready.md`
- 실제 검증 artifact: `submission/evidence/report_support_2026-03-10/`
- 과거 retrieval baseline 근거: `submission/reports/Draft.docx`
- 결과 입력 템플릿: `docs/reports/final_experiment_results_fill_template_ko.md`

## 3. Status Labels

- `completed`: 실제 실행 결과와 근거 파일이 있음
- `completed earlier`: 이전 제출/실험에서 이미 확보된 결과가 있고 현재 보고서에서 재인용 가능함
- `prepared`: 데이터셋/manifest/환경은 준비됐지만 실험은 아직 실행하지 않음
- `instrumented`: 계측 코드는 들어갔지만 summary 결과는 아직 없음
- `protocol defined`: 절차와 문항은 정의됐지만 실제 수행 결과는 없음
- `planned`: 아이디어/계획만 있고 실행 증적은 아직 없음

## 4. Experiment Evidence Map

| ID | Experiment / Evidence | Current status | Safe to claim now | Evidence / note |
|---|---|---|---|---|
| B0 | Draft-stage image-only retrieval baseline | completed earlier | Earlier baseline results exist | `submission/reports/Draft.docx`의 기존 image-only 결과만 인용 가능 |
| E0 | End-to-end scenario validation | partial evidence | Working prototype exists | 웹/API/모델 동작과 UI 흐름은 구현됐지만 formal scenario run log는 별도 정리 필요 |
| E1 | 1000-item fixed-split retrieval ablation | prepared | Dataset and manifests prepared | `1491`개 통합 데이터셋, `train/test` split, eval input 생성 완료. 아직 ablation 수치는 없음 |
| E2 | OCR identifier benchmark | planned | Protocol defined only | CER / exact match용 subset, ground truth schema, runner가 아직 최종 실행되지 않음 |
| E3 | Latency and resource benchmark | instrumented | Timing instrumentation exists | 단계별 timing capture는 구현됐지만 p50/p90/p95 summary는 아직 없음 |
| E4 | Human-review pilot study | protocol defined | Pilot protocol defined only | task sheet / Google Form 문항은 있으나 aggregate result는 없음 |
| E5 | Safety and reliability validation | completed | Regression evidence available | `submission/evidence/report_support_2026-03-10/README.md`의 pytest 결과 사용 가능 |

## 5. What Is Safe to Write in the Final Report

현재 시점에서 안전하게 쓸 수 있는 것은 아래와 같다.

1. **구현 완료 / working prototype**
- end-to-end indexing / hybrid search / catalog / agent / safer writeback / regression test

2. **이미 확보된 정량 결과**
- Draft-stage image-only retrieval baseline
- API / model pytest 결과

3. **준비 또는 진행 상태로만 써야 하는 것**
- 1000-item ablation
- OCR CER aggregate benchmark
- latency percentile summary
- human-review pilot aggregate outcome

## 6. Writing Rule

최종보고서에서는 아래 규칙을 사용한다.

- 수치 표와 artifact가 있으면 `completed`
- 이전 제출본에만 있고 현재 raw artifact가 부족하면 `completed earlier`
- 데이터와 입력만 준비됐으면 `prepared`
- 로그 수집만 되면 `instrumented`
- 문항/절차만 있으면 `protocol defined`
- 아직 실행 전이면 `planned`

## 7. Immediate Cleanup Suggestion

보고서 문서에서 혼동될 수 있는 표현은 다음처럼 바꾼다.

- `Latency measured` -> `Latency instrumentation implemented; batch summary pending`
- `OCR benchmark completed` -> `OCR benchmark protocol defined; aggregate results pending`
- `Hybrid ablation completed` -> `Hybrid ablation prepared on fixed inputs; quantitative run pending`
- `Evaluation completed` -> `Evaluation evidence currently consists of earlier baseline results plus recent engineering validation`
