# Prototype Write-Up (한국어 작업본)

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. Template Statement

본 프로토타입은 CM3020 템플릿 **"Orchestrating AI models to achieve a goal"**를 사용한다.

## 2. Project Overview

프로젝트의 전체 목표는 중고 플랫폼에서 사용자가 업로드한 사진만으로 산업용·전자 부품을 더 쉽게 식별하고 등록할 수 있도록 돕는 것이다. 이 문제는 단순 분류보다 검색 문제에 가깝다. 실제 현장에서는 희귀 부품, 단종 모델, 외형이 거의 비슷한 변형품이 섞여 있고, 결정적인 단서는 작은 라벨 텍스트나 part number인 경우가 많다.

이 프로토타입은 전체 프로젝트 중 핵심 식별 워크플로우를 구현한다. 사용자가 이미지를 업로드하면 시스템은 OCR, 멀티모달 임베딩, 벡터 검색, 텍스트 검색, 점수 융합을 통해 Top-K 후보를 반환한다. 또한 catalog 검색과 agent 보조 경로를 통해 내부 문서와 외부 근거를 함께 사용할 수 있게 했다.

## 3. Features Implemented

현재 프로토타입에 구현된 주요 기능은 다음과 같다.

1. 하이브리드 검색: 이미지, OCR 텍스트, 사용자 텍스트 질의를 함께 사용해 후보를 찾는다.
2. 메타데이터 기반 랭킹 보정: maker, part number, description exact/partial match를 점수에 반영한다.
3. 멀티이미지 인덱싱 흐름: 여러 장의 제품 사진을 하나의 모델로 연결해 인덱싱할 수 있다.
4. GPT 기반 메타 초안 생성: 사용자가 이미지를 올리면 저장 전에 메타데이터 초안을 자동 생성한다.
5. Catalog 검색: PDF 문서를 인덱싱하고 페이지 단위로 검색할 수 있다.
6. Agent 보조 흐름: hybrid search, catalog search, web search를 조합해 근거를 제시한다.
7. Safe write-back: 자동 저장이 아니라 사용자 확인 이후에만 저장되도록 했다.

## 4. Algorithms, Techniques, and Methods

핵심 알고리즘은 retrieval-first hybrid pipeline이다. 단일 모델이 정답을 하나 예측하는 구조가 아니라, 여러 신호를 모아 shortlist를 만드는 구조다.

### 4.1 Query-Time Flow

```text
User image/text
 -> OCR / caption / embedding generation
 -> Milvus search over image/text/model channels
 -> candidate merge by model_id
 -> lexical/spec-aware score boost
 -> reranking
 -> Top-K shortlist
```

이 구조를 선택한 이유는 다음과 같다.

- 이미지 유사도만으로는 세부 부품 변형을 구분하기 어렵다.
- OCR은 유용하지만 노이즈가 많기 때문에 단독 정답 근거가 되기 어렵다.
- 사용자에게는 단일 예측보다 후보 목록과 근거가 더 실용적이다.

### 4.2 Main Techniques

- OCR: PaddleOCRVL 우선, 필요 시 PaddleOCR fallback
- Image embedding: Qwen3-VL-Embedding-2B
- Text embedding: BGE-M3
- Reranking: Qwen3-VL-Reranker-2B
- Caption generation / metadata preview: Qwen3-VL 또는 GPT 기반 경로
- Retrieval store: Milvus multi-collection design

### 4.3 Why This Is Technically Challenging

이 프로토타입의 난점은 단순 이미지 검색이 아니라, **여러 불확실한 신호를 동시에 처리해야 한다는 점**이다.

- OCR은 실패할 수 있다.
- 텍스트가 없는 이미지도 있다.
- 비슷하게 생긴 부품이 많다.
- 사용자 이미지 품질이 일정하지 않다.
- 검색 결과는 실제 listing field와 연결되어야 한다.

따라서 이 시스템은 단일 모델 데모보다 더 복잡한 orchestration 문제를 다룬다.

## 5. Code Explanation

코드 전체를 붙이지 않고, 중요한 부분만 설명한다.

### 5.1 Hybrid Search Orchestrator

프로토타입의 중심은 `HybridSearchOrchestrator`이다. 이 클래스는 전처리, OCR, 임베딩 생성, Milvus 검색, 후보 병합, 최종 랭킹을 조정한다.

핵심 아이디어는 `model_id` 기준으로 여러 검색 채널의 결과를 합치는 것이다. 예를 들어 어떤 후보는 이미지 유사도는 높지만 OCR 점수는 낮을 수 있고, 다른 후보는 반대로 텍스트 일치도가 높을 수 있다. Orchestrator는 이를 합쳐 최종 점수를 계산한다.

### 5.2 Ranking Logic

랭킹 로직에서는 dense similarity만 보지 않는다. exact lexical evidence를 함께 반영한다.

```python
final_score = min(1.0, similarity * 0.65 + lexical_score * 0.20 + exact_field_boost * 0.15)
```

이 로직의 의미는 다음과 같다.

- `similarity`: 임베딩 기반 유사도
- `lexical_score`: 텍스트 토큰 단위 일치도
- `exact_field_boost`: maker, part number, description의 exact/partial match 보정

즉 “비슷해 보이기만 하는 결과”보다 “실제로 부품명이나 part number가 맞는 결과”를 더 위로 올리도록 설계했다.

### 5.3 Metadata Preview Before Indexing

저장 전 GPT 기반 메타 초안 생성도 중요한 구현이다. 이 기능은 사용자가 이미지를 먼저 올리고, 시스템이 maker, part number, category, description 초안을 생성한 뒤, 사용자가 수정·확인하고 저장하도록 한다. 이 흐름은 완전 자동 write-back보다 더 안전하고, 실제 플랫폼 workflow와도 맞다.

## 6. Visual Representation

최종 PDF에는 다음 시각 자료를 넣는 것이 적절하다.

1. 전체 시스템 아키텍처 다이어그램
2. 검색 화면 스크린샷
3. GPT 메타 초안 생성 화면 스크린샷
4. hybrid score evidence 예시
5. OCR/검색 failure 사례 이미지

## 7. Evaluation of Prototype Success

현재 프로토타입은 end-to-end 식별 보조 시스템으로서 의미 있는 수준까지 구현되었다.

성공한 점:

- web, API, model 계층이 실제로 연결된 working prototype이 있다.
- retrieval-first 설계가 코드 수준에서 구현되었다.
- image-only보다 hybrid retrieval이 필요하다는 설계 근거가 명확하다.
- catalog retrieval, agent orchestration, safer write-back까지 포함해 시스템 수준의 완성도가 올라갔다.

아직 부족한 점:

- OCR CER 정량평가는 아직 완료되지 않았다.
- hybrid ablation 실험은 계획되어 있으나 최종 수치가 아직 없다.
- latency percentile 보고도 준비 단계다.
- full accept/edit/reject workflow는 아직 부분 구현 상태다.

따라서 이 프로토타입은 “완전 자동 식별 시스템”이라기보다 **retrieval-first, human-in-the-loop identification assistant**로 평가하는 것이 가장 정확하다.

## 8. Planned Improvements

다음 개선이 적절하다.

1. fixed split 기준 retrieval benchmark 자동화
2. OCR on/off, reranker on/off ablation 수행
3. latency percentile batch report 생성
4. full reviewed write-back workflow 추가
5. 더 큰 GPU 환경에서 최신 모델 스택 전체 검증

## 9. Final Note

이 문서는 프로토타입 보고서용 작업본이다. 제출용 PDF를 만들 때는 2000단어 제한에 맞게 문장을 조금 더 압축하고, figure/table을 실제 스크린샷과 표로 교체하면 된다.
