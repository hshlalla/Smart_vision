# Final Submission Strategy

이 문서는 **최종 제출 직전**, 점수 효율이 가장 높은 작업만 추려서 정리한 마지막 전략 문서다.
목표는 새 기능 추가가 아니라 **final report / video / evidence의 완성도 극대화**다.

## 1. 기본 판단

- `90점` 수준은 기능 몇 개를 더 넣는다고 바로 나오지 않는다.
- 가장 큰 점수 레버리지는 아래 3가지다.
  1. `Evaluation` 장의 설득력
  2. `working project`가 분명히 보이는 video
  3. report claim과 evidence의 정합성
- 따라서 제출 직전에는 연구 backlog보다 제출물 품질을 우선한다.

## 2. 90점에 가까워지려면 필요한 것

### 2.1 Report

아래 항목이 빠지면 상위권 점수 방어가 어렵다.

- `docs/reports/final_report_docx_ready.md`의 insertion marker를 실제 표/그림으로 교체
- `Evaluation` 장에 실제 측정 결과와 해석을 반영
- `success / failure / limitation / future work`를 분리해서 비판적으로 서술
- public repository link 포함
- 참고문헌, 표/그림 캡션, 단어 수 제한 정리
- `Final Assembly Note`와 내부 메모 삭제

### 2.2 Evidence

- 다른 컴퓨터에서 실행한 실험 결과를 `submission/evidence/` 아래 제출 가능한 형태로 복사
- raw output 전체보다 아래 3가지를 우선 보존
  - `README.md` (무엇을 어떻게 실행했는지)
  - `metrics csv/json` 또는 summary markdown
  - 대표 스크린샷/표
- 보고서에는 raw run directory보다 요약된 evidence와 표를 인용

### 2.3 Video

- `3–5분`, 본인 음성, 배속 금지, working project가 보여야 함
- 반드시 보여줘야 할 것은 “무엇을 만들었는가”보다 “실제로 어떻게 동작하는가”다
- 추천 흐름:
  1. problem and why it is hard
  2. search demo
  3. indexing preview/confirm demo
  4. agent or catalog demo
  5. evaluation result summary
  6. limitations and contribution

## 3. 오늘 우선순위

### P0. Experiments (외부 컴퓨터)

- 실험 결과를 확정하고 요약 표를 만든다.
- `docs/reports/final_experiment_results_fill_template_ko.md`에 숫자를 채운다.
- `submission/evidence/final_experiments/` 아래로 옮길 제출용 복사본을 만든다.

### P1. Final Report

- `docs/reports/final_report_docx_ready.md`를 기준으로 final manuscript를 조립한다.
- 최소 반영 항목:
  - `Table 5-1`
  - `Table 5-2`
  - `Table 5-3`
  - 핵심 architecture / UI / failure example figure
- report 문구는 measured / partial / in progress를 구분한다.

### P2. Video

- 최종 report와 같은 메시지를 말하게 맞춘다.
- 실험 결과가 확정되면 script의 pending 표현을 결과 반영 표현으로 바꾼다.
- bilingual보다 단일 언어 버전이 시간 관리에 유리하면 단일 언어로 간다.

### P3. Submission Preflight

- public repo link 확인
- final DOCX/PDF 생성
- video file 생성
- 파일명 / 길이 / 음성 / 분량 확인

## 4. 실험과 비디오를 제외하고 남는 것

실험과 비디오를 다른 컴퓨터에서 진행하더라도, 아래는 별도 마감 작업으로 남는다.

1. final report에 결과 반영
2. 표/그림 삽입
3. final DOCX/PDF 생성
4. repo link 삽입
5. assembly note 제거
6. references / caption / word count 최종 점검

즉, 실험과 비디오가 끝나도 **final report 조립**이 남아 있으면 제출 완료가 아니다.

## 5. 고득점 관점의 실제 체크리스트

아래를 만족하면 점수 상단으로 갈 가능성이 커진다.

- 모든 주요 claim에 evidence가 연결됨
- `Evaluation` 장이 단순 결과 나열이 아니라 critical discussion까지 포함함
- limitation을 숨기지 않고 정확히 씀
- report의 그림/표가 본문 논리와 직접 연결됨
- video가 “작동하는 시스템 + 이해도”를 보여줌
- report와 video가 같은 메시지를 말함

## 6. 지금 하지 않아도 되는 것

제출 직전에는 아래 항목의 점수 효율이 낮다.

- 새로운 기능 추가
- 대규모 리팩터링
- 추가 모델 교체 실험
- backlog 전체 완료 시도
- accept/edit full workflow 같은 큰 기능 확장

이런 항목은 future work로 남겨도 된다.

## 7. 보수적 점수 해석

- 현재 저장소 상태만으로는 `90점`을 보장할 수 없다.
- 하지만 실험 결과 정리, report 완성, video 완성도가 높으면 `80대 상단` 이상은 충분히 현실적이다.
- `90점`은 특히 아래가 강해야 가능성이 생긴다.
  - evaluation quality
  - critical discussion quality
  - visual clarity
  - strong working video

점수는 새 기능 수보다 **제출물의 설득력**에 더 크게 좌우된다.
