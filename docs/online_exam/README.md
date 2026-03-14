# Online Exam Notes

이 폴더는 `submission/pastexam/`에 있는 CM3070 past exam을 바탕으로, 현재 프로젝트
`Smart Image Part Identifier for Secondhand Platforms`에 맞춰 정리한 온라인 시험 대비 자료를 담는다.

## Files

- `cm3070_past_exam_answer_bank.md`
  - 최근 past exam(2023-09, 2024-03, 2024-09, 2025-03)에 대한 프로젝트 맞춤형 답안 초안
  - 실제 시험에서는 모든 문항을 답하는 것이 아니라, 보통 그 해 지시사항에 맞춰 3문항만 선택해 답해야 하므로 여기서는 재사용 가능한 장문 답안 재료 중심으로 정리함

## Project framing used throughout

- Template: `CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"`
- Project: retrieval-first, human-in-the-loop assistant for identifying industrial/electronic parts from secondhand photos
- Core stack:
  - React web app
  - FastAPI backend
  - model package for hybrid retrieval
  - OCR + image/text embeddings + Milvus vector search + catalog retrieval + agent orchestration
- Evidence boundary:
  - end-to-end prototype exists
  - regression tests exist
  - latency instrumentation exists
  - full CER / latency percentile / usability evaluation is not yet complete and should be described honestly as partial or in progress

## Usage note

이 문서는 최종 암기본이 아니라, 답안 작성 시 바로 꺼내 쓸 수 있는 구조화된 초안이다.
실제 시험 답안에서는 다음을 지키는 것이 좋다.

- 질문에서 요구한 범위에 맞춰 3개 문항만 선택
- 현재 프로젝트에서 실제로 구현되거나 측정된 것만 주장
- `fully automated identification`처럼 과장된 표현은 피하고
  `retrieval-first, human-in-the-loop identification assistant`라는 framing 유지
- 평가 관련 문항에서는
  - 이미 있는 증거
  - instrumentation만 끝난 항목
  - future work
  를 분리해서 서술
