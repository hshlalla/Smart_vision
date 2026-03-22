# Draft 보강 가이드 (한국어)

이 문서는 **기존 draft 위에 내용을 덧대서 수정**하려는 작업 방식을 위한 가이드다.  
즉 [`01_Introduction.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/01_Introduction.md) 같은 파일을 처음부터 새로 쓰기보다, **기존 draft 문단을 유지하면서 어디에 무엇을 추가/교체할지**를 빠르게 찾을 수 있도록 정리했다.

## 1. 작업 원칙

지금부터는 장별 파일을 “새 원고”로 보기보다, 아래처럼 쓰는 것이 좋다.

- 기존 draft 문단은 최대한 유지
- `docs/reports/final_report/*.md`는 보강용 참고 문장으로 사용
- 실험 수치, 설계 rationale, 구현 변경점만 draft에 덧붙이기
- 문단 전체를 갈아엎기보다
  - 한두 문단 추가
  - 오래된 문장 교체
  - claim 수위 조정
로 해결

즉, **새로 쓰는 방식보다 patch 방식**으로 접근하는 것이 맞다.

---

## 2. 문서 사용 역할

### 그대로 복붙하지 말고 참고용으로 볼 파일

- [`01_Introduction.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/01_Introduction.md)
- [`02_Literature_Review.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/02_Literature_Review.md)
- [`03_Design.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/03_Design.md)
- [`04_Implementation.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/04_Implementation.md)
- [`05_Evaluation.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/05_Evaluation.md)
- [`06_Conclusion.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/06_Conclusion.md)

이 파일들은 “장별로 어떤 내용이 들어가야 하는지”를 보는 참고용이다.

### draft에 직접 반영할 때 먼저 볼 파일

- [`final_report_docx_ready_ko.md`](/Users/mac/project/Smart_vision/docs/reports/final_report_docx_ready_ko.md)
- [`final_report_status.md`](/Users/mac/project/Smart_vision/docs/reports/final_report_status.md)
- [`experiment_result_tables_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_result_tables_template_ko.md)
- [`experiment_results_writeup_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_results_writeup_template_ko.md)

이 파일들은 실제로 draft를 업데이트할 때 쓸 핵심 근거 문서다.

---

## 3. 장별 patch 방식

## 3.1 Introduction

기존 draft에서 유지해도 되는 것:

- 문제 중요성
- 산업적 배경
- 왜 part identification이 어려운지 설명

추가/교체해야 하는 것:

- 문제 framing을 `classification`보다 `retrieval-first, human-in-the-loop`로 더 분명히 정리
- 사용자 요구 조사에서 transparency / editability / evidence가 중요했다는 점 추가
- 목표 정의를 shortlist + evidence + listing support 중심으로 수정

추천 작업:

- 기존 introduction 앞부분은 유지
- 중간에 retrieval-first framing 문단 1개 추가
- 마지막 objective 문단은 최신 표현으로 교체

---

## 3.2 Literature Review

기존 draft에서 유지해도 되는 것:

- visual retrieval
- OCR
- multimodal retrieval
- vector DB 관련 배경

추가/교체해야 하는 것:

- OCR을 ground truth가 아닌 uncertain evidence로 보는 해석
- human-in-the-loop와 evidence-backed interaction의 중요성
- hybrid retrieval justification 강화

추천 작업:

- 기존 literature review는 대부분 유지
- 마지막 정리 문단만 새 방향으로 보강

---

## 3.3 Design

기존 draft에서 유지해도 되는 것:

- 전체 architecture
- web / api / model 분리
- Milvus multi-collection 구조

추가/교체해야 하는 것:

- `preview -> edit -> confirm` 인덱싱 구조
- multi-image indexing
- fallback-oriented hybrid design
- catalog + agent orchestration 경로

추천 작업:

- architecture 문단은 유지
- indexing path 설명은 최신 구조로 교체
- evaluation strategy 문단 추가

---

## 3.4 Implementation

기존 draft에서 유지해도 되는 것:

- React, FastAPI, Milvus 구성
- OCR / embedding / retrieval 파이프라인 개요

추가/교체해야 하는 것:

- GPT metadata preview 도입
- confirm 이후에만 저장되는 safer writeback
- multi-image indexing 지원
- catalog search와 agent path
- 한글 ranking fix
- unified dataset 구축과 eval manifest 생성

추천 작업:

- 기존 implementation 본문은 유지
- 인덱싱과 hybrid search 관련 문단만 최신 구조로 수정
- dataset preparation paragraph를 새로 1개 추가

---

## 3.5 Evaluation

이 장이 가장 많이 바뀌어야 한다.

기존 draft에서 유지해도 되는 것:

- image-only baseline
- failure analysis 방향

추가/교체해야 하는 것:

- main benchmark: `C2 vs C4`
- 운영 권고: `C3`
- OCR benchmark 결과
- latency 결과
- reliability 결과
- usability 결과 또는 현재 정리된 수준

추천 작업:

- 기존 evaluation 서론 유지
- 결과표는 새 템플릿 기준으로 교체
- 해석 문단은 최신 결과에 맞게 다시 작성

참고:

- [`05_Evaluation.md`](/Users/mac/project/Smart_vision/docs/reports/final_report/05_Evaluation.md)
- [`experiment_result_tables_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_result_tables_template_ko.md)
- [`experiment_results_writeup_template_ko.md`](/Users/mac/project/Smart_vision/docs/reports/experiment_results_writeup_template_ko.md)

---

## 3.6 Conclusion

기존 draft에서 유지해도 되는 것:

- 프로젝트 기여 요약
- orchestration project라는 점

추가/교체해야 하는 것:

- main benchmark에서는 `C4`가 strongest
- 운영 권고는 `C3`
- 시스템의 최종 표현은 `retrieval-first, human-in-the-loop identification assistant`
- future work 항목 최신화

추천 작업:

- 기존 conclusion은 살리고
- 마지막 두 문단만 최신 근거에 맞게 교체

---

## 4. 최소 수정 우선순위

시간이 부족하면 아래 순서로 patch한다.

1. Evaluation
2. Conclusion
3. Design
4. Implementation
5. Introduction
6. Literature Review

이 순서가 중요한 이유는, 최종 점수에 가장 큰 영향을 주는 것은

- 실험 결과를 정확히 반영했는지
- 주장 수위를 적절히 조정했는지
- 현재 시스템을 무엇으로 정의하는지

이 세 가지이기 때문이다.

---

## 5. 가장 쉬운 실제 작업 방식

추천 방식:

1. 기존 draft 문서를 연다
2. 이 가이드를 옆에 둔다
3. 장별로 다음만 한다
   - 유지할 문단 표시
   - 지울 문단 표시
   - 새로 넣을 문단만 추가
4. 모든 문단을 새로 쓰지 않는다

즉, 지금 `final_report/` 폴더는 “새 문서를 쓰는 틀”이 아니라,
**기존 draft를 어디서 어떻게 patch할지 알려주는 참고 세트**로 보면 된다.

---

## 6. 한 줄 요약

지금부터는 `final_report/`를 새 원고처럼 쓰지 말고,  
**기존 draft를 수정할 때 필요한 문장과 논리만 가져오는 patch reference 세트**로 사용하는 것이 맞다.
