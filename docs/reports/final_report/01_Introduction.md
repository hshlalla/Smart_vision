# 1. Introduction

## 1.1 Background and Motivation

이 프로젝트는 CM3020 템플릿 **“Orchestrating AI models to achieve a goal”**를 따른다. 핵심 목표는 중고 플랫폼의 등록 워크플로우에서 사용자가 업로드한 사진만으로 산업용·전자 부품의 제조사, 모델명, 부품 번호를 더 쉽게 식별할 수 있도록 돕는 것이다.

이 문제는 중요하다. 일반 소비재는 대략적인 외형만으로도 검색이 가능한 경우가 있지만, 산업용 부품은 잘못 식별될 경우 가격 책정 오류, 검색 실패, 거래 실패로 이어질 수 있다. 또한 실제 구분 단서는 전체 외형이 아니라 작은 모델 코드, 시리얼 번호, 명판(nameplate)인 경우가 많다. 그런데 이러한 단서는 사용자가 촬영한 이미지에서 glare, blur, 저해상도, 마모, 부분 가림 등의 이유로 잘 보이지 않는 경우가 많다. 그 결과 비전문 판매자는 매뉴얼, 카탈로그, 검색엔진을 수동으로 대조하며 시간을 많이 쓰게 된다.

산업 동향을 보면 사진 기반 보조 기능은 등록 마찰을 줄이는 데 도움이 된다. eBay의 이미지 검색은 사용자가 적절한 키워드를 모르더라도 사진으로 후보를 찾을 수 있음을 보여주었고, 한국의 일부 중고 플랫폼도 업로드 이미지를 바탕으로 상품 정보를 추정하는 AI 보조 기능을 도입했다. 그러나 이런 사례들은 주로 일반 소비재를 대상으로 한다. 부품 식별은 시각적으로 매우 비슷한 변형들이 많고, 실제로 중요한 정보가 작고 노이즈가 많은 텍스트인 경우가 많기 때문에 더 까다롭다. 따라서 이 문제는 단순한 이미지 유사도 검색 이상으로 다뤄져야 한다.

## 1.2 Problem Statement

이 프로젝트의 중심 주장은, 중고 부품 식별은 **closed-set classification**이 아니라 **retrieval-first, human-in-the-loop decision-support** 문제로 보는 것이 더 적절하다는 것이다. 유용한 시스템이라면 이미지와 텍스트 증거를 함께 활용할 수 있어야 하고, OCR이 실패해도 어느 정도 작동해야 하며, listing workflow에 맞는 구조화된 결과를 반환해야 하고, 사용자가 결과를 검증할 수 있도록 근거를 제시해야 한다.

## 1.3 Domain and User Requirements

### 1.3.1 Domain Requirements

이러한 framing은 도메인 제약을 직접 반영한다. 도메인 측면에서 중고 부품 인벤토리는 open-world이며 long-tail 특성을 가진다. 희귀 모델, 단종품, 신규 품목이 계속 등장하므로 고정된 클래스 집합에 기반한 분류기는 적합하지 않다. 또한 이 도메인은 매우 fine-grained하다. 외형은 비슷하지만 작은 표기 차이로 구분되는 경우가 많다. 이미지 품질 역시 일정하지 않기 때문에 OCR은 유용하지만 ground truth로 취급할 수는 없다. 마지막으로 사용자는 단순히 “비슷한 이미지”가 아니라 maker, part number, category, supporting evidence처럼 실제 등록에 활용할 수 있는 필드를 원한다.

### 1.3.2 User Requirements

사용자 측면의 요구도 같은 방향을 가리킨다. 중고 플랫폼 경험이 있는 참여자 6명을 대상으로 한 간단한 요구사항 조사에서, 상세 스펙을 텍스트로 작성하는 것이 어렵고 수동 검색 의존도가 높으며, 사진 기반 보조 기능에 대한 수요가 있음을 확인했다. 동시에 참여자들은 AI 결과를 무조건 신뢰하지 않았다. 반복적으로 강조된 것은 **투명성**, **검증 가능성**, **수정 가능성**이었다. 따라서 시스템은 최종 답을 독단적으로 제시하기보다 후보군을 좁혀주고 근거를 제공하는 shortlist 기반 워크플로우를 지향해야 한다.

[Insert Table 1-1 here: survey 결과와 requirement implication 요약 표 삽입]

## 1.4 Implications for System Design

이러한 요구는 retrieval-first, human-in-the-loop 설계를 직접적으로 요구한다. 즉 시스템은 단일 정답을 제시하기보다 Top-K 후보군을 제공하고, supporting evidence를 함께 제시하며, 사용자가 최종적으로 확인하고 수정할 수 있도록 구성되어야 한다. 이 요구는 이후 설계 파트의 hybrid retrieval 구조, metadata preview-confirm 인덱싱 경로, evidence-backed result presentation으로 이어진다.

## 1.5 Aim and Objectives

이러한 관찰을 바탕으로 본 프로젝트의 목표는, 사용자가 업로드한 부품 사진을 입력으로 받아 Top-5 후보 shortlist와 listing-oriented summary, 그리고 supporting evidence를 함께 제공하는 end-to-end prototype을 구현·평가하는 것이다. 세부 목표는 end-to-end 워크플로우 구현, retrieval effectiveness 평가, OCR robustness 분석, interactive feasibility를 위한 latency instrumentation 추가, 그리고 실제 listing assistance 관점에서의 유용성 평가로 정리할 수 있다.

## 1.6 Report Structure

이 보고서의 나머지 구성은 다음과 같다. 2장은 visual retrieval, OCR, multimodal embeddings, vector databases, interactive feedback 관련 선행연구를 검토한다. 3장은 시스템 설계를 설명하며 evaluation strategy를 설계 안에 포함한다. 4장은 web, API, model 계층에 걸친 구현 내용을 설명한다. 5장은 수행 완료된 실험과 평가 결과를 바탕으로 시스템의 성능과 한계를 분석한다. 6장은 기여, 한계, 향후 과제를 정리한다.
