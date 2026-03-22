# 3. Design

## 3.1 Design Goals and Constraints

본 시스템은 React 기반 웹 인터페이스, FastAPI 기반 API 계층, 그리고 hybrid retrieval orchestration을 담당하는 model 계층으로 구성된다. 전체 목표는 사용자가 업로드한 부품 이미지를 입력으로 넣었을 때, image evidence, OCR evidence, metadata text, caption text를 함께 사용해 관련 후보를 수집하고, 최종적으로 evidence-backed shortlist를 반환하는 것이다.

[Insert Figure 3-1 here: overall system architecture]

## 3.2 System Overview and Component Responsibilities

상위 수준에서 보면 시스템은 세 개의 상호작용 루프를 가진다. 첫째는 **query-time retrieval loop**다. 사용자는 이미지만 올리거나, 이미지와 텍스트를 함께 입력할 수 있다. 둘째는 **indexing loop**다. 새로운 아이템을 등록할 때 메타데이터 초안을 생성하고, 사용자가 수정한 뒤 확정 저장한다. 셋째는 **evidence expansion loop**다. 필요하면 catalog search나 외부 검색을 통해 추가 근거를 제공한다.

## 3.3 Data Model and Storage Schema

Query-time retrieval path의 핵심은 여러 신호를 따로 계산한 뒤 최종적으로 fusion하는 구조다. 이미지는 image embedding collection으로 검색되고, OCR이나 metadata 기반 텍스트는 text collection과 caption collection을 통해 검색된다. 이후 각 후보의 image score, text score, caption score, lexical signal, exact identifier boost를 조합해 최종 점수를 계산한다. 이 방식은 visually similar but textually distinct 항목을 구분하는 데 유리하다.

[Insert Figure 3-2 here: query-time hybrid retrieval flow]

## 3.4 Core Workflows

### 3.4.1 Indexing Workflow (Catalogue Ingestion)

인덱싱 경로는 human-in-the-loop를 전제로 설계되었다. 사용자가 이미지를 업로드하면 시스템은 곧바로 확정 저장하지 않고, 먼저 GPT 기반 metadata preview를 생성한다. 사용자는 여기서 maker, part number, category, description 등을 확인하고 수정할 수 있다. 실제 저장은 confirm 이후에만 수행된다. 이 구조는 자동 writeback보다 안전하며, 실제 listing workflow에 더 자연스럽다.

또한 본 시스템은 multi-image 입력을 고려한다. 단일 이미지로는 제품의 앞면, 옆면, 라벨면, 포트면을 모두 알 수 없기 때문에, metadata preview는 여러 장을 함께 보고 생성하고, 실제 인덱싱도 여러 장을 같은 item 단위로 반영하도록 설계하였다. 이 점은 fine-grained part identification에서 특히 중요하다.

### 3.4.2 Search Workflow (Query Photo to Top-K + Listing Summary)

설계상 중요한 또 다른 요소는 **graceful fallback**이다. OCR은 유용하지만 불안정하다. 따라서 OCR이 실패해도 image retrieval과 metadata-aware ranking이 어느 정도 작동해야 한다. 반대로 visual similarity만으로 부족한 경우에는 exact identifier boost나 text retrieval이 보강 역할을 한다. 이러한 fallback-oriented hybrid design은 실제 noisy image 조건에서 robustness를 높이기 위한 선택이다.

### 3.4.3 Accept/Edit Feedback Workflow (Human-in-the-Loop)

사용자는 retrieval shortlist와 metadata preview를 확인한 뒤 accept, edit, or confirm에 해당하는 행동을 수행한다. 현재 구현은 production-grade audit workflow까지는 가지 않지만, preview-confirm path와 safer writeback policy를 통해 human-in-the-loop control을 실질적으로 반영하고 있다.

## 3.5 Information Extraction and Identifier Parsing

이 시스템은 OCR, metadata preview, caption, exact substring matching을 함께 사용해 identifier signal을 다룬다. part number와 maker는 retrieval score뿐 아니라 evidence interpretation에서도 중요한 역할을 한다. 다만 본 프로젝트는 OCR을 완전한 truth source로 두기보다, retrieval and verification evidence의 일부로 설계했다.

## 3.6 Evaluation-Oriented Design (Instrumentation and Traceability)

추가적으로 시스템은 catalog retrieval과 agent orchestration도 포함한다. Catalog 경로는 PDF나 reference document를 벡터화하여 내부 문서 검색을 수행할 수 있게 하며, agent 경로는 hybrid search, catalog search, 그리고 외부 web search를 단계적으로 호출해 더 풍부한 근거를 제공한다.

[Insert Figure 3-3 here: agent and catalog orchestration path]

평가 전략도 설계 안에 포함된다. 본 프로젝트는 단순 accuracy 하나로 성능을 주장하지 않는다. 대신 retrieval effectiveness, OCR robustness, latency/interactivity, engineering reliability, usability를 서로 다른 실험 트랙으로 나누어 검증하도록 설계되었다. 이는 시스템이 실제로는 검색 도구이자 listing assistance workflow라는 점을 반영한 것이다.

[Insert Table 3-1 here: requirement-to-component traceability]

## 3.7 Design Limitations and Time-Boxed Extensions

현재 설계는 prototype의 핵심 흐름을 우선 구현하는 방향으로 time-boxed 되어 있다. 따라서 metadata-only draft registration, audited review workflow, selective OCR verification policy 같은 기능은 의도적으로 future work로 남겨 두었다.
