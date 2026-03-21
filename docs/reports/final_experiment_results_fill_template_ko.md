# Final Experiment Results Fill Template

이 문서는 최종보고서용 실험 결과를 **숫자만 채워 넣을 수 있도록** 만든 작업 템플릿이다.
목적은 `실험 계획`, `실제 실행 결과`, `최종보고서 문장`을 한 곳에 모으는 것이다.

기준 문서:
- 실험 계획: `docs/planning/experiment_plan_qwen3_vl_1000_items.md`
- 현재 증적 상태: `docs/reports/final_experiment_evidence_status.md`
- 최종 원고: `docs/reports/final_report_docx_ready.md`
- submission 증적 폴더: `submission/evidence/`

---

## 0. 사용 규칙

이 문서에서는 각 실험마다 아래 5가지를 반드시 채운다.

1. **실행 여부**: completed / partial evidence / in progress / prepared / instrumented / protocol defined / planned
2. **입력 데이터**: 무엇을 대상으로 돌렸는지
3. **실행 조건**: 모델/설정/환경/반복 횟수
4. **결과 수치**: 표 또는 bullet
5. **보고서용 문장**: 그대로 옮길 수 있는 문장

중요:
- 숫자와 artifact가 없으면 `completed`로 쓰지 않는다.
- `partial evidence`는 일부 실행 결과가 있지만 full controlled study는 아직 아니라는 뜻이다.
- `in progress`는 pilot나 부분 실행은 있었지만 aggregate 결과가 아직 정리되지 않았다는 뜻이다.
- `prepared`는 입력 데이터와 manifest가 준비된 상태를 뜻한다.
- `instrumented`는 로그 수집은 되지만 summary 수치는 아직 없는 상태를 뜻한다.
- `protocol defined`는 문항/절차는 확정됐지만 실제 결과는 아직 없다는 뜻이다.
- 최종보고서에는 `completed result`와 `planned study`를 반드시 분리해서 쓴다.

---

## 1. Global Experiment Metadata

아래 블록은 모든 실험에 공통으로 채운다.

### 1.1 Run Metadata

- Experiment date:
- Report draft version:
- Git commit hash:
- Operator:
- Machine / host:
- OS:
- Python version:
- GPU:
- CUDA / driver:
- Milvus version:
- Main collection names:

### 1.2 Model Stack

- Image embedding:
- Text embedding:
- Reranker:
- OCR path:
- Caption / product info model:
- Query-time OCR on/off:
- Index-time OCR on/off:

### 1.3 Dataset Snapshot

- Dataset root:
- Total items:
- Train / index split:
- Test / query split:
- Domain mix:
- Ground-truth rule:
- Manifest path:

### 1.4 Evidence Save Paths

권장 저장 위치:

- `submission/evidence/final_experiments/E0/`
- `submission/evidence/final_experiments/E1/`
- `submission/evidence/final_experiments/E2/`
- `submission/evidence/final_experiments/E3/`
- `submission/evidence/final_experiments/E4/`
- `submission/evidence/final_experiments/E5/`

각 폴더에는 가능하면 아래를 저장한다.
- raw output txt / log
- metrics csv or json
- screenshots if needed
- short README explaining what was run

---
주의: `experiments/runs/...` 원본은 실행 머신의 local-output 성격이라 git에 항상 포함되지는 않을 수 있다. 보고서에는 raw path 대신 요약 표, JSON/CSV export, `submission/evidence/` 복사본을 남기는 것을 우선한다.

---

## 2. Final Report Evaluation Summary Table

이 표는 최종보고서 Evaluation 장의 요약표 초안으로 쓸 수 있다.

| Area | Experiment ID | Current status | Main metric | Result summary | Safe wording for report |
|---|---|---|---|---|---|
| End-to-end workflow | E0 | completed | scenario success | current-index suite `8/8` scenario success | completed |
| Retrieval effectiveness | E1 / B0 | partial evidence / completed earlier | Accuracy@1, Hit@5, MRR, item_id@1 | current-index sanity + sampled `C3` baseline available; full controlled comparison pending | completed earlier / partial evidence |
| OCR robustness | E2 | in progress | CER, exact match | OCR pilot attempted, aggregate result pending | in progress |
| Latency / interactivity | E3 | partial evidence | mean, p50/p90/p95 latency | warm mean observations available; percentile summary pending | partial evidence |
| User usefulness | E4 | protocol defined | task time, trust score | pilot form/protocol defined; aggregate result pending | pilot protocol defined |
| Engineering reliability | E5 | completed | pytest pass, safety checks | regression tests, frontend build, eval input generation available | completed |

---

## 3. E0. End-to-End Scenario Validation

### 3.1 Purpose

- `O1` 방어
- 실제 사용 흐름이 끊기지 않고 동작하는지 확인

### 3.2 Report Claim Supported

- `A working end-to-end prototype exists.`
- `The prototype supports indexing, retrieval, evidence display, and follow-up interaction.`

### 3.2.1 Current Known Evidence

현재 확인된 실행 근거:

- source: `experiments/CURRENT_EXPERIMENT_STATUS.md`
- current-index suite 기준 `8/8` scenario success
- 이 결과는 end-to-end operational sanity evidence로 바로 사용 가능하다

최종보고서에서는 아래처럼 쓰면 안전하다.

> Operational end-to-end scenario validation showed `8/8` successful flows on the current indexed stack, supporting the claim that the prototype works across upload, indexing, retrieval, and evidence display.

### 3.3 Input / Scenario Definition

- Scenario count:
- Image set used:
- Query type:
- Whether metadata preview/edit was used:
- Whether agent path was included:
- Whether catalog path was included:

### 3.4 Recommended Procedure

1. 이미지 업로드
2. metadata preview 생성 확인
3. 필요 시 metadata 수정
4. confirm indexing 실행
5. 검색 또는 재검색 수행
6. Top-K 결과와 evidence 노출 확인
7. 필요 시 agent / catalog path 호출
8. writeback default off 상태 확인
9. 실패 지점이 있으면 로그와 화면 캡처 저장

### 3.5 Data to Record

- Step success/fail
- Failure reason
- Number of operator actions
- Whether evidence was shown
- Whether shortlist was usable
- Whether result could be reused after save

### 3.6 Result Table Template

| Scenario ID | Upload | Preview | Confirm Index | Search | Evidence Visible | Reuse After Save | Result | Notes |
|---|---|---|---|---|---|---|---|---|
| S1 |  |  |  |  |  |  |  |  |
| S2 |  |  |  |  |  |  |  |  |
| S3 |  |  |  |  |  |  |  |  |

### 3.7 Final Summary to Fill

- Total scenarios run:
- Successful end-to-end runs:
- Partial failures:
- Critical blocking failures:
- Main failure pattern:
- Main positive observation:

### 3.8 Report-ready English Sentence Template

Use this if E0 was actually run:

> End-to-end scenario validation was performed on [N] representative workflows covering upload, metadata preview, confirmation-based indexing, retrieval, and evidence inspection. [X]/[N] scenarios completed successfully without blocking failure. The main observed limitation was [issue], but the results support the claim that the prototype functions as a working retrieval-first identification assistant.

Use this if only implementation evidence exists:

> A formal end-to-end scenario log is still being consolidated, but the current web, API, and model layers already support the full prototype path from upload to indexing, retrieval, and evidence-backed output.

### 3.9 Korean Fill Notes

- 이 실험은 “기능이 있다”를 넘어 “실제로 한 번에 흘러가는가”를 보여주는 용도다.
- formal log가 없으면 completed라고 쓰지 말고 implementation evidence로 낮춰 쓴다.

---

## 4. E1. Retrieval Effectiveness Ablation

### 4.1 Purpose

- `O2` 방어
- `C1`~`C4` 비교
- image-only / OCR on/off / reranker on/off / text contribution 비교

### 4.2 Report Claim Supported

- `Hybrid retrieval improves shortlist usefulness over weaker baselines.`
- `The chosen configuration is justified by measured retrieval quality rather than by model preference alone.`

### 4.2.1 Current Known Evidence

현재 이미 확보된 근거는 아래와 같다.

1. current-index sanity benchmark
- `Accuracy@1 = 0.9732`
- `Hit@5 = 1.0`
- `MRR = 0.9855`
- 주의: 이는 deployed collection 기준 operational sanity evidence이며 strict generalisation benchmark로 단정하면 안 된다.

2. sampled holdout baseline (`C3`)
- `30` sampled items
- `1` held-out query image per item
- `OCR off`, `reranker off`
- group `Hit@1 = 1.0`, group `Hit@5 = 1.0`, `MRR = 1.0`
- exact `item_id@1 = 0.9667`

따라서 현재 상태는 “실험 미실행”이 아니라 “baseline / sanity evidence는 있고, full controlled comparison이 남아 있음”으로 정리해야 한다.

### 4.3 Input Definition

- Dataset version:
- Index items:
- Query items:
- Correctness rule (`group_key` or exact `item_id`):
- Manifest path:
- Repeated runs per config:

### 4.4 Config Table

| Config ID | Image embedding | OCR | Text channel | Reranker | Collection set | Notes |
|---|---|---|---|---|---|---|
| C1 |  | off | on | on |  |  |
| C2 |  | on | on | on |  |  |
| C3 |  | off/on | on | off |  |  |
| C4 |  | off | minimal | on |  |  |

### 4.5 Recommended Procedure

1. index manifest와 query manifest 고정
2. 각 config에 대해 별도 컬렉션 또는 완전히 구분된 index 준비
3. 동일 query set에 대해 prediction JSONL 생성
4. metrics script로 Accuracy@1 / Accuracy@5 / Recall@5 / MRR / exact identifier hit rate 계산
5. domain별, hard-case subset별로 분리 가능하면 추가 계산
6. 대표 실패 사례 5~10개 저장

### 4.6 Metrics Table Template

| Config | Accuracy@1 | Accuracy@5 | Recall@5 | MRR | Exact identifier hit rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| C1 |  |  |  |  |  |  |
| C2 |  |  |  |  |  |  |
| C3 |  |  |  |  |  |  |
| C4 |  |  |  |  |  |  |

### 4.7 Hard-case Breakdown Template

| Case type | Config | Accuracy@5 | Exact identifier hit rate | Notes |
|---|---|---:|---:|---|
| visually similar |  |  |  |  |
| blur / glare |  |  |  |  |
| partial label |  |  |  |  |
| cluttered background |  |  |  |  |

### 4.8 Key Interpretation Prompts

- Best overall config:
- Best config for exact identifier discrimination:
- Did OCR help materially?
- Did reranker help materially?
- Which config had best quality/complexity trade-off?
- What failure remained unsolved?

### 4.9 Report-ready English Sentence Template

> Retrieval effectiveness was evaluated on a fixed split of [index N] indexed items and [query N] held-out queries. The main comparison tested [configs]. The best-performing configuration was [config], with Accuracy@1 = [ ], Accuracy@5 = [ ], and MRR = [ ]. These results suggest that [OCR / reranker / text evidence] contributed meaningfully to shortlist quality, especially in [hard-case type] cases.

If not completed:

> A fixed-split retrieval ablation has been fully specified and the evaluation inputs have been prepared, but the final quantitative comparison across configurations is still pending.

### 4.10 Korean Fill Notes

- 이 실험이 끝나야 “왜 이 파이프라인을 채택했는지”를 정량으로 말할 수 있다.
- 안 끝났으면 `prepared` 또는 `quantitative run pending`으로 쓴다.

---

## 5. E2. OCR Identifier Benchmark

### 5.1 Purpose

- `O3` 방어
- OCR을 유지할 가치가 있는지 정량 판단
- exact identifier sensitivity 확인

### 5.2 Report Claim Supported

- `OCR is useful but uncertain evidence.`
- `Character-level errors remain a meaningful failure source in part identification.`

### 5.3 Input Definition

- OCR benchmark subset size:
- Source dataset:
- Ground-truth file path:
- Scenario groups included:
- Whether strings were normalised before scoring:

### 5.4 Compared Methods

| Method | Description | Notes |
|---|---|---|
| M1 | PaddleOCR path |  |
| M2 | Qwen3-VL extracted identifier text |  |
| M3 | OCR + Qwen merged evidence |  |

### 5.5 Recommended Procedure

1. identifier가 실제 보이는 샘플 subset 고정
2. 각 샘플의 ground-truth identifier 문자열 수동 검증
3. 각 method별 predicted string 저장
4. 동일 normalisation 규칙 적용
5. CER, exact full-string match, token recall 계산
6. blur / glare / small text 등 scenario별 breakdown 계산

### 5.6 Metrics Table Template

| Method | CER | Exact full-string match | Part-number token recall | Maker token recall | Notes |
|---|---:|---:|---:|---:|---|
| M1 |  |  |  |  |  |
| M2 |  |  |  |  |  |
| M3 |  |  |  |  |  |

### 5.7 Scenario Breakdown Template

| Scenario group | Method | CER | Exact match | Notes |
|---|---|---:|---:|---|
| clean single-object |  |  |  |  |
| blur / glare |  |  |  |  |
| cluttered background |  |  |  |  |
| partial / multi-object |  |  |  |  |
| small label / far distance |  |  |  |  |

### 5.8 Interpretation Prompts

- Which method had lowest CER?
- Which method had best exact match?
- Did merged evidence outperform each single path?
- In which scenario did OCR fail most often?
- Is OCR still worth keeping in the final stack?

### 5.9 Report-ready English Sentence Template

> OCR robustness was evaluated on a benchmark subset of [N] samples with manually verified ground-truth identifier strings. The best method achieved CER = [ ] and exact full-string match = [ ]. Performance dropped most sharply under [scenario], confirming that OCR is useful but should still be treated as uncertain evidence in realistic secondhand imagery.

If not completed:

> An OCR benchmark protocol and labelled subset have been defined, but aggregate CER and exact-match results are still pending.

### 5.10 Korean Fill Notes

- CER만 쓰지 말고 exact match도 같이 보는 게 좋다.
- 실제 보고서에서는 “OCR이 완벽하다”가 아니라 “유용하지만 불확실하다”로 정리하는 게 안전하다.

---

## 6. E3. Latency and Resource Benchmark

### 6.1 Purpose

- `O4` 방어
- interactive feasibility 판단
- pipeline 병목 파악

### 6.2 Report Claim Supported

- `The system has latency instrumentation at the component level.`
- `Interactive feasibility can be analysed with percentile metrics.`

### 6.2.1 Current Known Evidence

현재 확인된 warm-latency observation은 아래와 같다.

- current-index suite warm total mean: 약 `653.65 ms`
- sampled `C3` baseline warm total mean: 약 `731.13 ms`
- sampled `C3` warm mean preprocessing: 약 `698.96 ms`
- sampled `C3` warm mean text search: 약 `20.39 ms`
- sampled `C3` warm mean image search: 약 `3.72 ms`

이 수치는 percentile study를 대체하지는 않지만, 현재 병목이 query-side preprocessing/embedding에 있다는 정성적 결론을 뒷받침한다.

### 6.3 Input Definition

- Index size:
- Query count:
- Run count per query:
- Cold run definition:
- Warm run definition:
- Hardware:
- Batch size:

### 6.4 Recommended Procedure

1. index build time separately collect
2. query set 100개를 최소 3회 반복
3. 첫 회는 cold-ish, 이후는 warm으로 구분
4. total 및 stage별 latency 수집
5. 평균, p50, p90, p95 계산
6. 가능하면 GPU memory peak도 함께 기록

### 6.5 Query Latency Table Template

| Mode | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| cold total |  |  |  |  |  |
| warm total |  |  |  |  |  |

### 6.6 Stage Breakdown Table Template

| Stage | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| preprocessing |  |  |  |  |  |
| image_search |  |  |  |  |  |
| ocr_search |  |  |  |  |  |
| caption_search |  |  |  |  |  |
| text_search |  |  |  |  |  |
| rerank |  |  |  |  |  |
| fetch_models |  |  |  |  |  |
| finalize |  |  |  |  |  |
| total |  |  |  |  |  |

### 6.7 Indexing / Resource Table Template

| Metric | Value | Notes |
|---|---:|---|
| indexing mean latency |  |  |
| indexing p95 latency |  |  |
| peak GPU memory |  |  |
| cold/warm gap |  |  |

### 6.8 Interpretation Prompts

- Is total p95 acceptable for practical use?
- Which stage dominates latency?
- Does OCR add unacceptable cost?
- Does reranker add acceptable cost relative to quality gain?
- Is cold-start behaviour a deployment risk?

### 6.9 Report-ready English Sentence Template

> Latency was measured at both total and component levels using structured timing logs collected over [N] repeated queries. The warm-query total latency was p50 = [ ] ms and p95 = [ ] ms, while the dominant bottleneck was [stage]. These results indicate that the current system is [acceptable / borderline / too slow] for interactive use under the tested hardware conditions.

If not completed:

> The hybrid-search path now captures structured timing information for the main processing stages, but a full percentile summary has not yet been generated.

### 6.10 Korean Fill Notes

- 이 실험이 없으면 latency는 “계측됨”까지만 말해야 한다.
- p50/p90/p95가 없으면 completed study로 쓰면 안 된다.

---

## 7. E4. Human-Review Pilot Study

### 7.1 Purpose

- `O5` 방어
- shortlist + evidence가 실제 사용자 effort를 줄이는지 확인
- trust / clarity / edit burden 확인

### 7.2 Report Claim Supported

- `The prototype supports a practical listing-assistance workflow.`
- `Users can use shortlist and evidence to narrow decisions.`

### 7.3 Study Setup

- Participant count:
- Participant profile:
- Facilitator:
- Recording allowed:
- Baseline condition:
- Assisted condition:
- Task set:

### 7.4 Recommended Tasks

- Task 1: image-based identification
- Task 2: shortlist judgement
- Task 3: metadata preview/edit/confirm

### 7.5 Measures

- task completion time
- plausible/correct final answer 여부
- manual edit count
- external search usage count
- Likert trust/usability scores
- open feedback

### 7.6 Quantitative Result Table Template

| Participant | Task 1 time | Task 2 time | Task 3 time | Correct / plausible | Manual edits | External searches | Notes |
|---|---:|---:|---:|---|---:|---:|---|
| P1 |  |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |  |

### 7.7 Questionnaire Summary Template

| Item | Mean score (1-5) | Notes |
|---|---:|---|
| Interface clarity |  |  |
| Shortlist usefulness |  |  |
| Evidence usefulness |  |  |
| Metadata preview usefulness |  |  |
| Preference over manual search |  |  |
| Decision confidence |  |  |

### 7.8 Qualitative Summary Template

- Most useful part:
- Most confusing part:
- First improvement requested by participants:
- Main trust concern:
- Main evidence-related positive:

### 7.9 Report-ready English Sentence Template

> A small pilot usability study was conducted with [N] participants using three listing-oriented tasks. The study recorded task completion time, manual edits, external search usage, and post-task trust/usability ratings. Participants rated [best aspect] most positively, while the main pain point was [issue]. These findings provide early, small-scale evidence that the shortlist-plus-evidence design can reduce effort, although the study is not large enough to support strong statistical claims.

If not completed:

> A pilot usability protocol has been defined using three short listing-oriented tasks and a post-task questionnaire, but aggregate participant results are still pending.

### 7.10 Korean Fill Notes

- 표본이 작으면 “early pilot evidence”로만 써야 한다.
- 강한 통계 주장보다 usability signal / pain point 정리에 초점을 둔다.

---

## 8. E5. Safety and Reliability Validation

### 8.1 Purpose

- engineering reliability 근거 확보
- safer writeback, regression stability, testability 방어

### 8.2 Report Claim Supported

- `Recent safety and testability changes were executed and validated.`
- `Writeback is safe by default.`

### 8.3 Inputs

- API pytest command:
- Model pytest command:
- Evidence folder:
- Tested commit hash:

### 8.4 Recommended Checks

1. API pytest pass
2. model pytest pass
3. `update_milvus=false` default 확인
4. explicit opt-in 없이 저장 안 되는지 확인
5. lazy import로 lightweight test collection 되는지 확인

### 8.5 Result Table Template

| Check | Result | Evidence path | Notes |
|---|---|---|---|
| API pytest |  |  |  |
| Model pytest |  |  |  |
| Writeback default off |  |  |  |
| Explicit opt-in required |  |  |  |
| Lightweight import / testability |  |  |  |

### 8.6 Current Known Evidence

현재 바로 인용 가능한 known evidence:

- API tests: `12 passed, 1 warning in 5.37s`
- Model tests: `4 passed in 0.09s`
- Evidence root: `submission/evidence/report_support_2026-03-10/`

### 8.7 Report-ready English Sentence Template

> Engineering reliability was supported by a focused validation pass recorded in the evidence bundle. The API test suite reported [ ], and the model test suite reported [ ]. These checks confirmed that writeback is now safe by default and that lightweight test collection no longer fails due to heavyweight eager imports.

### 8.8 Korean Fill Notes

- E5는 지금도 가장 방어 가능한 completed evidence다.
- retrieval benchmark를 대신하진 않지만, 시스템 프로젝트라는 점에서는 중요한 근거다.

---

## 9. Final Paste Targets in the Report

실험 결과를 채운 뒤 아래 위치에 옮긴다.

- Evaluation overview 표 -> `docs/reports/final_report_docx_ready.md:128`
- Retrieval baseline / ablation -> `docs/reports/final_report_docx_ready.md:136`
- OCR benchmark 결과 -> `docs/reports/final_report_docx_ready.md:149`
- Latency benchmark 결과 -> `docs/reports/final_report_docx_ready.md:151`
- Usability pilot 결과 -> `docs/reports/final_report_docx_ready.md:153`
- Engineering reliability 결과 -> `docs/reports/final_report_docx_ready.md:157`
- Objective status 표 -> `docs/reports/final_report_docx_ready.md:164`

---

## 10. Minimum Safe Final Report Version

만약 제출 전까지 E1~E4를 끝내지 못하면, 최소한 아래 조합으로 쓴다.

1. **completed earlier**
- Draft-stage image-only baseline

2. **completed**
- E5 safety / reliability validation

3. **prepared / instrumented / protocol defined**
- E1 fixed-split dataset and manifests
- E2 OCR benchmark protocol
- E3 latency instrumentation
- E4 usability pilot protocol

이 조합이면 “아직 안 한 실험을 한 것처럼 쓰는 오류”는 피할 수 있다.
