# Peer Review Round 2 Submission (한국어 작업본)

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. 문서 목적

이 문서는 두 번째 peer review 제출용 작업본이다.  
문제 지침에 맞춰 다음 두 가지를 포함한다.

1. 프로젝트 실행 및 테스트 안내
2. 리뷰어가 답해야 하는 세 가지 평가 질문

또한 peer review에서 공통으로 적용되는 generic criterion인 **“Is the final product of high quality?”**도 함께 고려한다.

## 2. Project Template

이 프로젝트는 CM3020 템플릿 **“Orchestrating AI models to achieve a goal”**를 사용한다.

## 3. Project Overview

프로젝트명은 **Smart Image Part Identifier for Secondhand Platforms**이다.  
이 시스템은 사용자가 업로드한 사진을 바탕으로 산업용·전자 부품을 식별할 수 있도록 돕는 retrieval-first AI prototype이다.

핵심 동작은 다음과 같다.

1. 사용자가 이미지 또는 텍스트 질의를 입력한다.
2. 시스템은 OCR, 멀티모달 임베딩, 벡터 검색, 텍스트 검색을 결합한다.
3. 후보 결과를 Top-K shortlist로 반환한다.
4. 필요한 경우 메타데이터 초안을 제안하고, 사람이 수정 후 저장할 수 있다.

## 4. Instructions for Running and Testing the Project

### Recommended review mode

가장 권장되는 방식은 **데모 영상 시청**이다.

이유는 다음과 같다.

- 프로젝트는 web frontend, FastAPI backend, Milvus, OCR runtime, multimodal model stack에 의존한다.
- 운영체제와 하드웨어에 따라 모델 실행 가능 여부가 달라질 수 있다.
- 따라서 모든 리뷰어가 동일하게 재현 가능한 제출 형태로는 영상이 가장 안정적이다.

### If a reviewer wants to run the project

필요 조건은 다음과 같다.

- Python 3.12+
- Node.js 20+
- Docker / Docker Compose
- Milvus
- 충분한 RAM과 저장공간
- 일부 assisted feature를 위한 OpenAI API key

### Practical note

현재 프로젝트는 강한 runtime dependency를 가진다.  
따라서 peer review 목적에서는 “직접 실행”보다 “영상 기반 기능 검토”가 더 적절하다.

또한 영상은 단순 화면 녹화보다 **3~5분 voiceover demo** 형태가 더 적합하다.  
리뷰어가 무엇을 보고 있는지, 그리고 그 기능이 사용자 요구와 어떻게 연결되는지를 이해할 수 있게 해야 한다.

### Suggested review procedure

1. 데모 영상을 본다.
2. 검색 흐름과 결과 화면을 확인한다.
3. 결과가 납득 가능한 shortlist인지 판단한다.
4. 아래 세 질문에 응답한다.

여기서 중요한 것은 “이 시스템이 앞으로 얼마나 더 확장될 수 있는가”보다, **현재 상태에서 원래 목표를 얼마나 성공적으로 달성했는가**를 평가하는 것이다.

## 5. Questions for Reviewers

리뷰어는 아래 세 문항에 대해 다음 척도로 응답한다.

- Disagree
- Partially disagree
- Neither agree nor disagree
- Partially agree
- Agree

### Question 1

**I could understand how to use the prototype interface without additional explanation.**

### Question 2

**The retrieval results and supporting evidence looked appropriate for the demonstrated query.**

### Question 3

**The prototype appeared technically challenging and more substantial than a simple single-model demo.**

## 6. Generic Criterion Used in Peer Review

리뷰어는 위 세 질문 외에도 다음 generic criterion을 함께 고려한다.

- **Is the final product of high quality?**

## 7. What Reviewers Will Be Asked to Provide

리뷰어는 실제 peer review 시스템 안에서 다음 형태의 평가를 하게 된다.

1. 위 세 질문에 대한 척도 응답
2. generic quality criterion에 대한 응답
3. 프로젝트의 장점
4. 개선 제안

즉 제출자는 별도의 답변 양식을 작성하는 것이 아니라,  
리뷰어가 이러한 항목으로 평가할 수 있도록 **명확한 실행 안내와 좋은 질문 세 개**를 제공하면 된다.

## 8. Submission Summary

이 peer review 제출은 다음을 제공한다.

- 실행 및 테스트 안내
- 소프트웨어/하드웨어 요구사항
- 사용한 프로젝트 템플릿 명시
- 리뷰어가 평가할 세 가지 질문

따라서 이번 peer review 과제 요구사항에는 맞는 형태로 정리되었다.
