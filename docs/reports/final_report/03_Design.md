# 3. Design

이 장은 시스템 아키텍처, AI orchestration pipeline, human-in-the-loop(HITL) 워크플로우를 설명한다. 또한 retrieval quality, latency, OCR robustness, user-facing usefulness를 체계적으로 검증할 수 있도록 시스템이 어떻게 evaluation-oriented하게 설계되었는지도 함께 제시한다.

## 3.1 High-Level System Architecture

모듈성과 반복적 실험을 가능하게 하기 위해, 시스템은 monorepo 구조 안에 분리된 application layer와 재사용 가능한 model package를 두는 형태로 설계되었다. 이러한 separation of concerns는 AI orchestration logic이 사용자 인터페이스와 독립적으로 발전할 수 있게 하면서도, 전체 워크플로우는 하나의 통합 시스템으로 유지되도록 한다.

- **Frontend (`apps/web`)**  
  React 기반 Single Page Application으로, 최종 사용자 인터페이스 역할을 한다. 이미지 업로드, metadata preview 및 수정, Top-K 결과 확인, catalog search, agent-assisted interaction을 지원한다.

- **Backend API (`apps/api`)**  
  FastAPI 기반 REST 서비스로, 인증, preview-confirm 인덱싱 요청 처리, 비동기 indexing task 추적, multimodal search routing, 그리고 웹 클라이언트와 모델 계층 간의 통신을 담당한다.

- **AI and Search Engine (`packages/model` + Milvus)**  
  핵심 intelligence layer로, metadata normalisation, OCR, image embedding, text embedding, caption generation, signal fusion, result ranking을 수행한다. Milvus는 multimodal retrieval을 위한 vector database로 사용되며, model-level metadata와 image-level attribute는 dedicated collection에 저장되어 hybrid search를 지원한다.

[Insert Figure 3.1: End-to-End System Architecture Pipeline]  
*(작성 가이드: `docs/architecture/ARCHITECTURE_MERMAID.md`의 “1) 전체 Pipeline (End-to-End)” 다이어그램 캡처본 삽입)*

**Figure 3.1. 웹 클라이언트, FastAPI 백엔드, 모델 오케스트레이션 계층, Milvus 기반 retrieval 저장소를 연결하는 end-to-end 시스템 구조.**

이 구조는 “Orchestrating AI models to achieve a goal” 템플릿을 직접적으로 반영한다. 무거운 multimodal processing은 model package에 격리되고, frontend와 API는 명시적인 JSON 또는 multipart interface를 통해 이를 호출한다.

## 3.2 AI Orchestration Pipeline

이 프로젝트의 핵심 기술적 과제는 global visual similarity와 maker, part number 같은 fine-grained textual identifier를 어떻게 균형 있게 다루는가에 있다. 따라서 시스템은 처음부터 단일 가설에 고정된 파이프라인으로 설계되지 않았고, OCR on/off, reranker on/off, lighter text channel 등 여러 구성을 실험할 수 있도록 설계되었다. 이는 competing design choice를 개념 수준이 아니라 empirical evidence 수준에서 비교할 수 있게 하기 위한 것이다.

최종 orchestrated pipeline은 다음 요소를 중심으로 구성된다.

- **Primary Vision Engine (Qwen3-VL)**  
  Qwen3-VL은 기본 image embedding backbone으로 사용된다. 이 모델의 장점은 단순한 visual representation뿐 아니라 mixed layout, logo-like manufacturer mark, embedded text pattern을 문맥적으로 다룰 수 있다는 점이다.

- **Text Embedding (BGE-M3)**  
  BGE-M3는 metadata, OCR-derived text, caption, user query text를 공통 searchable space에 표현하는 데 사용된다. 이를 통해 짧고 밀도 높은 alphanumeric identifier와 listing field를 pure image similarity 이상으로 다룰 수 있다.

- **Optional OCR (PaddleOCR)**  
  OCR은 기본 first-stage retrieval engine이 아니라 configurable component로 포함된다. 산업 이미지에서는 모든 visible text를 추출하는 것이 오히려 irrelevant signal과 추가 latency를 낳을 수 있기 때문에, OCR은 secondary evidence source로 취급된다.

- **Captioning Layer (GPT or Qwen Captioner)**  
  caption generation은 visual content를 상위 semantic text로 변환해 embedding 및 retrieval에 활용하기 위해 사용된다. runtime mode에 따라 GPT 기반 captioner 또는 Qwen 기반 captioner가 선택된다.

- **Fusion and Ranking Layer**  
  orchestrator는 image similarity, OCR-derived text similarity, caption similarity, lexical overlap, exact identifier boost를 결합한다. optional reranker path도 지원하지만, 품질과 latency를 both with and without reranker 조건에서 비교할 수 있도록 설계되었다.

[Insert Figure 3.2: AI Orchestration and Multimodal Fusion Pipeline]  
*(작성 가이드: `docs/architecture/ARCHITECTURE_MERMAID.md`의 “3) Model 상세” 다이어그램 캡처본 삽입)*

**Figure 3.2. image embedding, text evidence, caption signal, ranking logic가 결합되어 retrieval-first workflow를 구성하는 multimodal orchestration pipeline.**

이처럼 여러 신호를 조합하는 방식은 open-world part identification에서 closed-set classifier보다 더 적절한 retrieval-first 설계를 뒷받침한다.

## 3.3 Human-in-the-Loop (HITL) Workflow

1장의 도메인 분석과 survey 결과가 보여주듯, 사용자는 중고 플랫폼 listing에서 AI 출력을 검토하고 수정할 기회 없이 그대로 신뢰하지 않는다. 따라서 본 시스템은 autonomous classifier가 아니라, interactive decision-support tool로 설계되었다.

이 철학은 preview-confirm workflow로 구현된다.

- **Preview Phase (`/api/v1/hybrid/index/preview`)**  
  사용자는 한 장 이상의 이미지를 업로드한다. 시스템은 metadata draft를 생성하고, 업로드 항목이 기존 indexed model과 유사해 보일 경우 possible duplicate candidate도 함께 반환할 수 있다.

[Insert Figure 3.3: Login page]  
*(작성 가이드: 서비스 진입용 로그인 화면 단독 캡처 삽입)*

**Figure 3.3. 인덱싱, 검색, catalog, agent-assisted workflow에 들어가기 전의 서비스 진입 로그인 화면.**

- **Review and Edit**  
  frontend는 생성된 metadata와 visual evidence를 제시한다. 사용자는 candidate 정보를 확인하고, 이미지 비교를 통해 maker, part number, category, description 같은 필드를 수동으로 수정할 수 있다.

- **Confirm Phase (`/api/v1/hybrid/index/confirm`)**  
  사용자가 승인하면 시스템은 indexing job을 비동기 큐에 넣는다. 이후 업로드 이미지는 처리되어 Milvus에 저장되고, task state는 polling으로 추적된다.

[Insert Figure 3.4: Indexing upload and preview-confirm screens]  
*(작성 가이드: 인덱싱 업로드 화면과 preview/confirm 결과 화면을 2장 패널 또는 3장 패널로 삽입)*

**Figure 3.4. 이미지 업로드, metadata draft 생성, duplicate review, user confirmation으로 이어지는 human-in-the-loop 인덱싱 워크플로우.**

이 설계는 두 가지 역할을 한다. 첫째, 불확실한 AI 출력이 blind write-back 되는 위험을 줄인다. 둘째, 최종 human confirmation을 요구함으로써 indexed dataset의 품질을 더 안정적으로 유지한다.

## 3.4 Duplicate-Aware Ingestion and Multi-Image Design

산업 부품 식별은 단일 이미지로 충분하지 않은 경우가 많다. 앞면은 대략적인 object class를 보여주지만, 옆면, label view, port view가 결정적인 identifier를 담고 있는 경우가 흔하다. 이를 반영해 indexing workflow는 item 단위 multi-image input을 지원하도록 설계되었다.

또한 duplicate-aware ingestion도 포함된다. 새로운 업로드가 기존 part와 일치할 가능성이 높으면, 시스템은 기존 `model_id`를 재사용하고 genuinely new image만 추가할 수 있다. richer metadata 역시 버리지 않고 merge할 수 있다. 이는 불필요한 중복을 줄이고, 여러 사진이 동일한 physical item 또는 model family에 속할 수 있다는 현실을 더 잘 반영한다.

## 3.5 Search Workflow and Result Construction

query time에는 multimodal path와 lightweight path를 모두 지원한다.

- **Multimodal path**에서는 query image와 optional user text를 함께 사용하며, image similarity, optional OCR-based text similarity, caption-based similarity, lexical overlap, exact identifier boost를 계산한다.
- **Text-only path**에서는 이미지 없이 텍스트 검색이 필요할 때 BGE-M3 기반의 lighter model-level retrieval route를 사용한다.

핵심 설계 원리는 graceful degradation이다. OCR이 실패하거나 비활성화되어도 image retrieval과 metadata-aware matching은 여전히 유용한 후보를 반환해야 한다. 반대로 pure visual similarity가 부족한 경우에는 lexical 및 identifier signal이 순위를 보강한다. 이러한 hybrid design은 noisy real-world image 조건에서 robustness를 높이기 위한 선택이다.

[Insert Figure 3.5: Search page]  
*(작성 가이드: 검색 입력 화면 전체를 단독 캡처로 삽입)*

**Figure 3.5. 이미지 입력과 structured text input을 함께 사용하는 multimodal 검색 페이지.**

[Insert Figure 3.6: Search results and evidence-backed shortlist]  
*(작성 가이드: Top-K 결과, thumbnail, evidence section이 함께 보이는 결과 화면을 단독 또는 2장 패널로 삽입)*

**Figure 3.6. multimodal query가 ranked candidate shortlist와 inspectable evidence로 이어지는 검색 결과 화면.**

## 3.6 Extended Services: Catalog and Agent Support

핵심 retrieval pipeline은 두 가지 확장 서비스와 연결된다.

- **Catalog Retrieval**  
  reference PDF와 internal document를 index하고 검색함으로써, manual이나 catalog-like source에서 supporting material을 가져올 수 있게 한다.

- **Agent-Orchestrated Assistance**  
  agent layer는 hybrid search, catalog search, web search를 결합해 단순 retrieval만으로는 부족한 경우 더 넓은 evidence를 제공한다.

[Insert Figure 3.7: Catalog interface]  
*(작성 가이드: catalog 페이지 단독 캡처 삽입)*

**Figure 3.7. manual, PDF reference, supporting technical material에 대한 document-grounded lookup을 수행하는 catalog 화면.**

[Insert Figure 3.8: Agent support interface]  
*(작성 가이드: agent 페이지 단독 캡처 삽입. 필요하면 `docs/architecture/ARCHITECTURE_MERMAID.md`의 “5) Catalog + Agent Orchestration Path” 다이어그램을 보조 자료로 함께 사용)*

**Figure 3.8. catalog retrieval, hybrid search, broader evidence gathering이 core retrieval workflow를 어떻게 확장하는지 보여주는 agent support 화면.**

이 구성요소들은 retrieval benchmark의 유일한 초점은 아니지만, 시스템을 좁은 retrieval engine이 아니라 더 실용적인 listing-assistance workflow로 확장하는 역할을 한다.

## 3.7 Evaluation-Oriented Design

이 아키텍처는 ad hoc demo를 위한 구조가 아니라, 체계적인 평가를 지원하도록 의도적으로 설계되었다. profiling hook, ablation-friendly configuration switch, benchmark script를 준비함으로써 다음 질문들을 empirical하게 검증할 수 있도록 했다.

- OCR과 reranking이 retrieval quality에 얼마나 기여하는가?
- 각 구성의 latency cost는 얼마인가?
- 산업 이미지에서 OCR은 identifier extraction에 얼마나 robust한가?
- 시스템은 user-facing listing support에 충분한 transparency와 editability를 제공하는가?

이를 위해 설계에는 다음이 포함된다.

- configurable OCR-on / OCR-off path
- optional reranker usage
- image, text, caption, metadata signal의 분리 처리
- asynchronous indexing task status reporting
- offline evaluation input을 위한 index/query split 생성 경로

**Table 3.1. Traceability between domain requirements, system components, and evaluation strategy**

[Insert Table 3.1: Requirement-to-component traceability]

| Domain Requirement (from Ch.1) | Architectural Component (Design) | Evaluation Metric / Strategy (to be Ch.5) |
| :--- | :--- | :--- |
| 정확한 fine-grained part identification | Hybrid Fusion Pipeline (Qwen3-VL + BGE-M3) | Retrieval Effectiveness (Accuracy@1, Accuracy@5) |
| 산업 텍스트 노이즈에 대한 강인성 | OCR을 분리하고 secondary fallback으로 사용 | C2 vs C4 offline benchmark; Character Error Rate |
| 사용자 신뢰와 투명성 | Human-in-the-Loop UI (Preview/Confirm workflow) | User-centred qualitative assessment |
| listing workflow에 필요한 속도 | FastAPI async task 및 분리된 model package | Component-level Latency Profiling (p50/p90) |

이 evaluation-oriented 구조는 “어떤 하나의 구성이 보편적으로 최고”라고 주장하기보다, trade-off를 실제로 관찰하고 측정할 수 있게 만든다는 점에서 중요하다.

## 3.8 Design Limitations and Time-Boxed Extensions

현재 설계는 제한된 구현 기간 안에서 핵심 prototype workflow를 우선 완성하는 방향으로 time-boxed 되어 있다. 따라서 metadata-only draft registration, fuller audited review workflow, more selective OCR trigger policy, broader deployment-oriented optimisation 같은 기능은 의도적으로 future work로 남겨 두었다.

이러한 한계는 현재 prototype의 가치를 부정하는 것이 아니라, 구현된 research prototype과 future productisation 사이의 경계를 명확히 해 주는 역할을 한다.
