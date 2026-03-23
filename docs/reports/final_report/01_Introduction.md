# 1. Introduction

이 프로젝트는 CM3020 템플릿 **“Orchestrating AI models to achieve a goal”**를 따른다. 핵심 목표는 중고 플랫폼에서 산업용 및 전자 부품을 등록하는 과정의 마찰을 줄이고, 비전문 판매자가 업로드한 사진을 바탕으로 제조사, 모델명, 부품 번호를 더 쉽게 식별할 수 있도록 돕는 것이다.

## 1.1 Background and Motivation

일반 소비재는 전체 외형만으로도 검색이 가능한 경우가 많지만, 산업용 부품은 성격이 다르다. 중고 거래 맥락에서 부품을 잘못 식별하면 가격 책정 오류, 검색 실패, 거래 실패, 그리고 구매자 신뢰 저하로 이어질 수 있다. 실제로 결정적인 식별 단서는 물체의 전체 형상보다 작은 모델 코드, 시리얼 번호, 명판(nameplate)인 경우가 많다.

그러나 이러한 단서는 사용자가 직접 촬영한 사진에서 잘 드러나지 않는 경우가 많다. glare, blur, 저해상도, 마모, 부분 가림 때문에 중요한 식별 정보가 누락되거나 읽기 어려워진다. 그 결과 비전문 판매자는 매뉴얼, 카탈로그, 검색엔진을 일일이 대조하며 listing을 수동으로 작성해야 한다.

산업 동향은 사진 기반 보조 기능이 등록 마찰을 줄일 수 있음을 시사한다. 예를 들어 eBay의 이미지 검색은 사용자가 정확한 키워드를 모를 때도 사진으로 검색할 수 있음을 보여주었고, 최근의 AI listing 보조 기능은 업로드 이미지를 바탕으로 상품 정보를 추정하는 방향으로 발전하고 있다. 하지만 이러한 도구는 대체로 일반 소비재에 최적화되어 있다. 산업용 부품은 외형이 거의 동일한 변형이 많기 때문에, 단순한 이미지 유사도만으로는 충분하지 않다.

## 1.2 Problem Statement

이 프로젝트의 핵심 전제는 중고 산업 부품 식별을 **closed-set classification** 문제가 아니라, **open-world, retrieval-first, human-in-the-loop decision-support** 문제로 보아야 한다는 것이다.

초기에는 일반적인 시각 유사도와 OCR을 적극적으로 결합하면 가장 좋은 결과를 얻을 수 있다고 가정할 수 있었다. 그러나 산업 도메인의 제약을 고려하면, OCR은 전기 사양, 배경 텍스트, 포장재 정보와 같은 불필요한 문자열을 과도하게 추출해 retrieval 품질을 오히려 떨어뜨릴 수 있다. 따라서 실용적인 시스템이라면 다음 조건을 만족해야 한다.

1. 단순한 raw text extraction에 의존하지 않고, vision-language model을 통해 레이아웃과 시각적 문맥을 함께 이해할 수 있어야 한다.
2. OCR은 기본 검색 엔진이 아니라 선택적 보조 검증 신호로 다뤄져야 한다.
3. 텍스트가 없거나 읽기 어려운 경우에도 graceful degradation이 가능해야 한다.
4. maker, part number와 같은 구조화된 결과를 listing workflow에 맞게 제공해야 한다.
5. 사용자가 결과를 검토하고 수정할 수 있도록 투명한 evidence를 함께 제시해야 한다.

## 1.3 Domain and User Requirements

이러한 framing은 도메인 제약과 사용자 요구를 동시에 반영한다. 도메인 측면에서 중고 부품 인벤토리는 open-world이며 long-tail 특성을 가진다. 희귀 모델, 단종품, 신규 품목이 계속 등장하므로 순수한 closed-set 분류기는 이 도메인에 적합하지 않다. 또한 이 도메인은 매우 fine-grained하다. 겉보기 형상은 비슷하지만, 작은 식별 표기 차이만으로 다른 부품이 되는 경우가 많다.

사용자 요구를 확인하기 위해 중고 플랫폼 사용 경험이 있는 참여자 `n = 6`을 대상으로 간단한 요구사항 elicitation survey를 수행했다. 응답 결과, 상세 스펙을 텍스트로 작성하는 것이 어렵고 수동 검색 의존도가 여전히 높다는 점이 확인되었다. 동시에 사진 기반 식별 보조 기능에 대한 수요도 분명했다. 그러나 사용자는 AI 결과를 무조건 신뢰하지 않았고, **투명성**, **근거 확인 가능성**, **수정 가능성**을 반복적으로 강조했다. 따라서 본 시스템은 단일 정답을 강하게 제시하기보다, supporting evidence와 함께 Top-K 후보군을 제공하고 최종 판단을 사용자에게 남기는 방향으로 설계되었다.

주요 survey 결과와 requirement implication은 **표 1**에 요약했다.

**Table 1. Survey findings and requirement implications (`n = 6`)**

| Survey finding | Responses | Requirement implication |
| --- | --- | --- |
| 상세 스펙을 작성하는 것이 어렵다 | 3/6 respondents (50%) | 구조화된 metadata output 필요 (FR6: Part Card) |
| 사용자는 식별을 위해 수동 검색에 의존한다 | 5/6 use search engines; 4/6 check labels/manuals | 사진 기반 identification support 필요 (FR1-FR4) |
| 사용자는 AI 보조 기능에 개방적이다 | 5/6 would probably use a photo-based system | listing assistance 기능 필요 |
| AI 결과를 무조건 신뢰하지 않는다 | 0/6 trust without review; 4/6 would still review | human-in-the-loop confirmation 필요 (FR7) |
| 투명성과 수정 가능성이 신뢰를 높인다 | 2/6 cited transparency; 2/6 editability | explainability 및 edit 기능 필요 (NFR7) |
| 워크플로우 속도가 중요하다 | 1/6 cited speed explicitly | latency target 필요 (NFR3) |

*Table note:* 이 표는 초기 설계 단계에서 수행한 소규모 요구사항 조사 결과를 요약한 것이다. 표본 수가 제한적이므로 (`n = 6`), 통계적 일반화보다 설계 방향을 뒷받침하는 정성적 근거로 해석하는 것이 적절하다.

survey 질문지와 응답 요약은 Appendix A에 별도로 수록한다.

## 1.4 Aim and Objectives

**Aim**  
사용자가 업로드한 부품 사진을 입력으로 받아, structured listing summary와 supporting evidence를 동반한 Top-5 candidate shortlist를 제공하는 end-to-end prototype을 구현·오케스트레이션·평가함으로써 listing creation process를 단순화하는 것이다.

**Objectives**

**O1. Working MVP Workflow**  
catalogue indexing, multimodal retrieval, human-in-the-loop confirmation step을 포함하는 end-to-end 흐름을 구현한다.

**O2. Retrieval Effectiveness**  
open-world 데이터셋에서 Accuracy@1, Accuracy@5, MRR 등의 지표를 사용해 retrieval 성능을 평가하고, visual signal과 OCR-derived text 간의 균형을 분석한다.

**O3. Latency and Interactive Feasibility**  
component-level latency와 percentile 요약을 계측하여 interactive listing workflow에서 실용적인 응답성이 확보되는지 평가한다.

**O4. OCR Robustness Analysis**  
character-level error metric을 포함한 targeted benchmark를 통해 산업 이미지에서 전통적 OCR의 한계를 분석한다.

**O5. User-Centred Evaluation**  
evidence transparency, editability, pilot usability feedback을 통해 시스템의 실제 유용성을 평가한다.

## 1.5 Report Structure

이 보고서의 나머지 구성은 다음과 같다. 2장은 visual retrieval, OCR, multimodal embeddings, interactive system 관련 선행연구를 비판적으로 검토한다. 3장은 시스템 설계와 evaluation-oriented architecture를 설명한다. 4장은 web, API, model 계층에 걸친 구현 내용을 다룬다. 5장은 평가 결과를 분석하며, empirical evidence를 바탕으로 vision-language-dominant architecture로의 설계 전환을 설명한다. 6장은 프로젝트의 기여, 한계, 향후 과제를 정리한다.
