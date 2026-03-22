# 4. Implementation

## 4.1 Technology Stack and Runtime Environment

구현은 크게 web, API, model 세 계층으로 나뉜다. Web 계층은 React와 Vite를 사용해 검색, 인덱싱, agent interaction, catalog upload와 같은 사용자 인터페이스를 제공한다. API 계층은 FastAPI를 기반으로 요청 검증, 세션 처리, hybrid orchestration 호출, logging과 safety policy를 담당한다. Model 계층은 OCR, embedding, reranking, Milvus interaction, hybrid score fusion을 담당한다.

[Insert Figure 4-1 here: repository or module overview]

## 4.2 Code Structure and Core Modules

검색 UI는 텍스트 질의와 이미지 업로드를 함께 지원한다. 사용자가 검색을 실행하면 결과 카드에는 단순 제목뿐 아니라 score와 evidence를 함께 보여주도록 구성되어 있다. 이를 통해 사용자는 black-box prediction을 받는 것이 아니라, 왜 이 결과가 상위에 왔는지를 일부라도 판단할 수 있다.

![Figure 4-2: Web search UI](../../images/fig_4_2_web_search_ui.png)

## 4.3 Indexing Pipeline Implementation

인덱싱 경로는 과거의 direct-save 흐름보다 `preview -> edit -> confirm` 구조를 중심으로 재설계되었다. Preview 단계에서는 GPT 기반 metadata draft를 생성하고, 사용자가 이를 수정할 수 있다. Confirm 이후에만 실제 image/text embedding과 Milvus insert가 수행된다. 이는 human review가 필수인 listing workflow에 더 적합하다.

또한 실제 인덱싱은 단일 이미지가 아니라 multi-image를 반영할 수 있다. 이는 노트북, 산업용 장비 부품, 자동차 부품처럼 앞면과 라벨면 정보가 다를 수 있는 항목에서 중요하다. Preview는 여러 이미지를 함께 보고 메타데이터 초안을 만들고, confirm 단계에서는 여러 이미지가 동일 item 아래 인덱싱된다.

## 4.4 Search Pipeline Implementation

하이브리드 검색 구현의 핵심은 `hybrid_pipeline_runner`와 그 하위 모듈들이다. 이 부분은 query preprocessing, image embedding, text embedding, multi-collection retrieval, lexical boosting, exact identifier boosting, reranking을 단계적으로 수행한다. 최근에는 한글 질의에 대한 lexical matching 개선, exact substring boost 강화, 낮은 score 후보 컷오프 같은 ranking fix도 반영되었다. 이 변경은 실제로 `"홍수훈"` 같은 한글 exact-match 질의에서 잘못된 결과가 1순위로 오는 문제를 줄이기 위해 도입되었다.

[Insert Figure 4-3 here: hybrid score decomposition example]

### 4.4.1 Core Retrieval Algorithm

검색 파이프라인은 image, text, caption, lexical signal을 따로 계산한 뒤 fusion ranking을 수행한다. 이 구조는 open-world retrieval과 partial evidence robustness를 모두 고려한 것이다. 특히 exact identifier boost와 lexical score는 domain-specific signal을 반영하기 위한 핵심 구현 요소다.

## 4.5 Human-in-the-Loop Feedback and Knowledge Accumulation

API 계층에서는 safer writeback과 testability가 강화되었다. 예를 들어 writeback은 opt-in 또는 confirm 경로로 제한되며, preview만으로는 Milvus에 쓰지 않는다. 또한 regression tests를 통해 API endpoint와 일부 model utility path에 대한 회귀 검증이 추가되었다.

## 4.6 Evaluation Tooling and Profiling Hooks

Catalog 기능은 PDF 문서를 벡터화해 searchable knowledge source로 만드는 경로를 제공한다. Agent 기능은 hybrid retrieval 결과, catalog evidence, external search evidence를 조합해 더 풍부한 답변을 생성한다. 이는 단순 검색 데모를 넘어, source-backed decision support 흐름을 만든다는 점에서 의미가 있다.

![Figure 4-4: Agent UI with writeback toggle](../../images/fig_4_4_agent_chat_ui.png)

추가로 데이터셋 구축과 평가 입력 생성도 구현 범위에 포함된다. GParts 기반 자동차 부품 데이터와 1090481 반도체 장비용 부품 데이터를 통합해 `unified_v1` 데이터셋을 만들었고, 이를 바탕으로 train/test split과 retrieval evaluation manifest를 생성했다. 이 작업은 단순한 데이터 정리가 아니라, 이후 실험의 재현성과 정당성을 확보하는 데 중요한 구현 단계였다.

## 4.7 Implementation Status Summary

현재 구현 상태는 prototype 수준에서 end-to-end workflow를 충족한다. 검색, 인덱싱, catalog, agent, bilingual UI, evaluation tooling까지 연결된 상태이며, 최종 실험 결과를 통해 주요 설계 선택의 타당성을 확인할 수 있는 수준에 도달했다.
