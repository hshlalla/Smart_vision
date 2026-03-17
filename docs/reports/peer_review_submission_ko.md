# Peer Review Submission (한국어 작업본)

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. 제출 목적

이 문서는 peer review 제출용 안내문 작업본이다.  
리뷰어가 프로젝트를 어떤 형태로 확인해야 하는지, 어떤 질문에 답해야 하는지, 그리고 실행 조건이 무엇인지 명확히 전달하는 것이 목적이다.

## 2. 프로젝트 개요

프로젝트명은 **Smart Image Part Identifier for Secondhand Platforms**이다.  
이 시스템은 사용자가 업로드한 부품 사진을 바탕으로 후보 부품을 검색하고, OCR, 메타데이터, 텍스트 질의를 함께 활용하여 listing workflow를 돕는 **retrieval-first identification assistant**를 목표로 한다.

핵심 기능은 다음과 같다.

1. 이미지 기반 부품 검색
2. OCR 및 텍스트 기반 보조 검색
3. 메타데이터 초안 자동 생성
4. Catalog PDF 검색
5. Agent 기반 보조 검색 및 근거 제시

## 3. Peer Review용 제출 형태

현재 peer review 제출 형태는 **영상 + 실행 안내문** 기준으로 제안한다.

이유는 다음과 같다.

- 전체 시스템은 web, API, model, Milvus, OCR, multimodal embedding runtime을 함께 필요로 한다.
- 실행 환경에 따라 모델 의존성과 하드웨어 차이가 있다.
- 리뷰어가 동일한 로컬 환경을 갖추지 않았을 수 있으므로, 영상 기반 시연이 가장 안정적이다.

따라서 reviewer 제출 패키지는 다음 구성을 권장한다.

1. 3~5분 데모 영상
2. 이 peer review 안내문 PDF
3. 필요 시 간단한 스크린샷 1~2장

영상은 가능하면 **voiceover가 포함된 시연**이 좋다.  
즉 어떤 기능을 보고 있는지, 왜 그 기능이 중요한지, 사용자 관점에서 어떤 도움이 되는지를 직접 설명해야 한다.

## 4. Reviewer가 확인해야 할 내용

리뷰어는 다음 흐름을 확인하면 된다.

1. 검색 화면에서 이미지 또는 텍스트 질의 입력
2. Top-K 후보와 근거가 반환되는지 확인
3. 메타데이터 자동 추출 또는 보조 식별 흐름 확인
4. 결과가 listing assistance 관점에서 유용한지 판단

여기서 핵심은 “앞으로 무엇을 더 할 수 있는가”보다, **현재 프로토타입이 원래 목표를 얼마나 잘 달성했는가**를 판단하는 것이다.

## 5. Reviewer Questions

리뷰어는 아래 세 질문에 대해 다음 척도로 답한다.

- agree
- partially agree
- neither agree nor disagree
- partially disagree
- disagree

### Question 1

**I could understand how to use the prototype interface without additional explanation.**

### Question 2

**The retrieval results and supporting evidence looked appropriate for the example query.**

### Question 3

**The prototype seemed technically challenging and more substantial than a simple single-model demo.**

추가로 플랫폼의 generic criterion인 아래 항목도 함께 평가된다.

- **Is the final product of high quality?**

## 6. Software and Hardware Requirements

### Recommended review mode

가장 권장되는 리뷰 방식은 **데모 영상 시청**이다.

### If a reviewer wants to run the system

필요 조건은 다음과 같다.

- Python 3.12+
- Node.js 20+
- Docker / Docker Compose
- Milvus runtime
- OpenAI API key for metadata preview or agent features
- 충분한 메모리와 저장공간

실행 난이도가 있는 이유:

- OCR, embedding, reranking, vector database가 함께 동작해야 한다.
- 일부 최신 모델 조합은 하드웨어와 운영체제에 영향을 받는다.
- 따라서 peer review에서는 실행형보다 영상형 제출이 더 현실적이다.

## 7. Reviewer Instructions

리뷰어는 다음 순서로 확인하면 된다.

1. 데모 영상을 본다.
2. 검색 입력과 결과 화면을 확인한다.
3. 결과 근거가 납득 가능한지 판단한다.
4. 위 세 질문에 대해 척도로 답한다.

## 8. 제출용 요약 문구

최종 peer review 제출에서는 다음과 같이 정리할 수 있다.

> This submission is a video-based demonstration of a retrieval-first AI prototype for identifying industrial and electronic parts from secondhand listing images. Reviewers are asked to evaluate interface clarity, result appropriateness, and technical challenge using the three provided Likert-scale questions.
