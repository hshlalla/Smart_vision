# Final Experiment Evidence Status

이 문서는 최종보고서에서 **무엇을 실제 결과로 써도 되는지**와 **무엇이 아직 준비/계획 단계인지**를 분리하기 위한 기준 문서다.

## 1. Purpose

최종보고서에서는 `구현 완료`, `실험 준비 완료`, `계측 구현`, `프로토콜 정의`, `실험 완료`를 명확히 구분해야 한다.
이 문서의 목적은 실험 계획서와 실제 증적을 연결해, 보고서에서 과장 표현이 들어가지 않도록 하는 것이다.

## 2. Canonical Sources

- 실험 계획 기준: `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- 보고서 상태 기준: `docs/reports/final_report_status.md`
- 최종 원고 기준: `docs/reports/final_report_docx_ready.md`
- 실제 실험 러너 개요: `experiments/README.md`
- 현재 실험 상태 기준: `experiments/CURRENT_EXPERIMENT_STATUS.md`
- 실제 검증 artifact: `submission/evidence/report_support_2026-03-10/`
- 과거 retrieval baseline 근거: `submission/reports/Draft.docx`
- 결과 입력 템플릿: `docs/reports/final_experiment_results_fill_template_ko.md`

## 3. Status Labels

- `completed`: 실제 실행 결과와 근거 파일이 있음
- `completed earlier`: 이전 제출/실험에서 이미 확보된 결과가 있고 현재 보고서에서 재인용 가능함
- `prepared`: 데이터셋/manifest/환경은 준비됐지만 실험은 아직 실행하지 않음
- `partial evidence`: 일부 실행 결과와 관찰값은 있으나, 최종 controlled study로 쓰기엔 범위가 제한적임
- `instrumented`: 계측 코드는 들어갔지만 summary 결과는 아직 없음
- `in progress`: 실행은 시작됐거나 pilot가 있었지만 aggregate 결과가 아직 미완성임
- `protocol defined`: 절차와 문항은 정의됐지만 실제 수행 결과는 없음
- `planned`: 아이디어/계획만 있고 실행 증적은 아직 없음

## 4. Experiment Evidence Map

| ID | Experiment / Evidence | Current status | Safe to claim now | Evidence / note |
|---|---|---|---|---|
| B0 | Draft-stage image-only retrieval baseline | completed earlier | Earlier baseline results exist | `submission/reports/Draft.docx`의 기존 image-only 결과만 인용 가능 |
| E0 | End-to-end scenario validation | completed | Current deployed path works operationally | `experiments/CURRENT_EXPERIMENT_STATUS.md` 기준 current-index suite에서 `8/8` scenario success |
| E1 | Retrieval effectiveness ablation | partial evidence | Operational sanity result와 sampled holdout baseline 사용 가능 | current-index sanity benchmark(`Accuracy@1=0.9732`, `Hit@5=1.0`, `MRR=0.9855`)와 sampled `C3` baseline(`item_id@1=0.9667`)은 사용 가능. 다만 full controlled `C1/C2/C3/C4` 비교는 미완료 |
| E2 | OCR identifier benchmark | in progress | Pilot attempt only | OCR/Qwen pilot runner는 있으나 aggregate CER / exact match 결과는 아직 미완성 |
| E3 | Latency and resource benchmark | partial evidence | Warm mean latency observation 사용 가능 | current-index suite warm total mean 약 `653.65 ms`, sampled `C3` warm total mean 약 `731.13 ms`는 사용 가능. 다만 `p50/p90/p95` 및 full resource summary는 미완료 |
| E4 | Human-review pilot study | protocol defined | Pilot protocol defined only | task sheet / Google Form 문항은 있으나 aggregate result는 없음 |
| E5 | Safety and reliability validation | completed | Regression evidence available | `submission/evidence/report_support_2026-03-10/`와 `experiments/CURRENT_EXPERIMENT_STATUS.md`의 reliability refresh(API/model tests, frontend build, eval input generation) 사용 가능 |

## 5. What Is Safe to Write in the Final Report

현재 시점에서 안전하게 쓸 수 있는 것은 아래와 같다.

1. **구현 완료 / working prototype**
- end-to-end indexing / hybrid search / catalog / agent / safer writeback / regression test
- current-index suite 기준 `E0` 시나리오 `8/8` 성공

2. **이미 확보된 정량/운영 결과**
- Draft-stage image-only retrieval baseline
- current-index sanity benchmark (`Accuracy@1=0.9732`, `Hit@5=1.0`, `MRR=0.9855`)
- sampled holdout `C3` baseline (`item_id@1=0.9667`, group `Hit@1=1.0`, `MRR=1.0`)
- warm mean latency observation (`~653.65 ms`, `~731.13 ms`)
- API / model pytest 결과, frontend build, retrieval-eval input generation

3. **진행 상태로만 써야 하는 것**
- full controlled `C1/C2/C3/C4` retrieval ablation
- OCR CER / exact-match aggregate benchmark
- latency percentile/resource summary
- human-review pilot aggregate outcome

주의:
- `experiments/runs/` raw output은 실행 머신의 local-output 성격이라 git에 항상 포함되지는 않는다.
- 따라서 현재 보고서 문구는 `experiments/CURRENT_EXPERIMENT_STATUS.md`에 요약된 수치와 결론을 기준으로 맞춘다.

## 6. Writing Rule

최종보고서에서는 아래 규칙을 사용한다.

- 수치 표와 artifact가 있으면 `completed`
- 이전 제출본에만 있고 현재 raw artifact가 부족하면 `completed earlier`
- 일부 실행 결과와 관찰값은 있으나 범위가 제한적이면 `partial evidence`
- 실행은 시작됐지만 aggregate 결과가 정리되지 않았으면 `in progress`
- 데이터와 입력만 준비됐으면 `prepared`
- 로그 수집만 되면 `instrumented`
- 문항/절차만 있으면 `protocol defined`
- 아직 실행 전이면 `planned`

## 7. Immediate Cleanup Suggestion

보고서 문서에서 혼동될 수 있는 표현은 다음처럼 바꾼다.

- `Latency measured` -> `Operational warm-latency observations are available; full percentile summary is still pending`
- `OCR benchmark completed` -> `OCR pilot has been attempted, but aggregate benchmark results are still pending`
- `Hybrid ablation completed` -> `Current-index sanity and sampled baseline runs exist, but the full controlled comparison remains in progress`
- `OCR on/off not implemented` -> `OCR toggles are implemented; the remaining issue is controlled comparison completeness`
- `Evaluation completed` -> `Evaluation evidence currently consists of earlier baseline results, operational sanity runs, sampled baseline runs, and recent engineering validation`
