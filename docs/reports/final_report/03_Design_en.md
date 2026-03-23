# 3. Design

This chapter explains the system architecture, the AI orchestration pipeline, and the human-in-the-loop (HITL) workflow designed to address the challenges identified in the literature review. It also shows how the system was structured to support empirical evaluation of retrieval quality, latency, OCR robustness, and user-facing usefulness.

## 3.1 High-Level System Architecture

To maintain modularity and enable iterative experimentation, the system is organised as a monorepo with separate application layers and a reusable model package. This separation of concerns allows the AI orchestration logic to evolve independently from the user interface while keeping the overall workflow integrated.

- **Frontend (`apps/web`)**  
  A React-based Single Page Application that serves as the primary user interface. It supports image upload, metadata preview and editing, Top-K result inspection, catalog search, and agent-assisted interaction.

- **Backend API (`apps/api`)**  
  A FastAPI-based REST service that acts as the application orchestrator. It manages authentication, preview-confirm indexing requests, asynchronous indexing task tracking, multimodal search routing, and communication between the web client and the model layer.

- **AI and Search Engine (`packages/model` + Milvus)**  
  The core intelligence layer performs metadata normalisation, OCR, image embedding, text embedding, caption generation, signal fusion, and result ranking. Milvus is used as the vector database for multimodal retrieval, while model-level metadata and image-level attributes are stored in dedicated collections to support hybrid search [8], [11].

[Insert Figure 3.1: End-to-End System Architecture Pipeline]  
*(작성 가이드: `docs/architecture/ARCHITECTURE_MERMAID.md`의 “1) 전체 Pipeline (End-to-End)” 다이어그램 캡처본 삽입)*

**Figure 3.1. End-to-end system architecture linking the web client, FastAPI backend, model orchestration layer, and Milvus-based retrieval storage.**


This architecture directly supports the “Orchestrating AI models to achieve a goal” template. Heavy multimodal processing is isolated in the model package, while the frontend and API layers communicate with it through explicit JSON or multipart interfaces.

## 3.2 AI Orchestration Pipeline

The main technical challenge is to balance global visual similarity with fine-grained textual identifiers such as maker names and part numbers. Rather than hard-coding a single pipeline assumption from the start, the system was intentionally designed to support multiple retrieval configurations, including OCR-on and OCR-off conditions, reranker-on and reranker-off conditions, and lighter text channels. This made it possible to evaluate competing design choices empirically rather than only conceptually.

The final orchestrated pipeline is centred on the following components:

- **Primary Vision Engine (Qwen3-VL)**  
  Qwen3-VL is used as the main image embedding backbone. Its advantage in this project is not only strong visual representation, but also context-aware handling of mixed layout, logo-like manufacturer marks, and embedded text patterns that are difficult to isolate through naive OCR alone [1].

- **Text Embedding (BGE-M3)**  
  BGE-M3 is used to represent metadata, OCR-derived text, captions, and user query text in a common searchable space. This strengthens retrieval over short, dense alphanumeric identifiers and allows metadata-aware matching beyond pure image similarity [2].

- **Optional OCR (PaddleOCR)**  
  OCR is included as a configurable component rather than a compulsory first-stage retrieval engine. In industrial imagery, extracting all visible text often introduces irrelevant signals and additional latency. For that reason, OCR is treated as a secondary evidence source that can support verification and ranking, rather than being assumed to be universally beneficial [9].

- **Captioning Layer (GPT or Qwen Captioner)**  
  Caption generation is used to convert visual content into higher-level semantic text that can be embedded and searched. Depending on runtime mode, either a GPT-based captioner or a Qwen-based captioner is used.

- **Fusion and Ranking Layer**  
  The orchestrator combines image similarity, OCR-derived text similarity, caption similarity, lexical overlap, and exact identifier boosts. An optional reranker path can be enabled, but the system is designed so that retrieval quality and latency can be evaluated both with and without it.

[Insert Figure 3.2: AI Orchestration and Multimodal Fusion Pipeline]  
*(작성 가이드: `docs/architecture/ARCHITECTURE_MERMAID.md`의 “3) Model 상세” 다이어그램 캡처본 삽입)*

**Figure 3.2. Multimodal orchestration pipeline showing how image embeddings, text evidence, caption signals, and ranking logic are combined into a retrieval-first workflow.**

By combining these signals rather than relying on a single model output, the system adopts a retrieval-first design that is more appropriate for open-world part identification than a closed-set classifier.

## 3.3 Human-in-the-Loop (HITL) Workflow

As shown by the domain and survey findings in Chapter 1, users do not fully trust AI-generated outputs for secondhand marketplace listings without the opportunity to inspect and correct them [3]. For this reason, the system is designed not as an autonomous classifier, but as an interactive decision-support tool.

This is implemented through a preview-confirm workflow:

- **Preview Phase (`/api/v1/hybrid/index/preview`)**  
  The user uploads one or more images. The system generates a metadata draft and may also surface a likely duplicate candidate if the uploaded item appears similar to an already indexed model.

- **Review and Edit**  
  The frontend presents the generated metadata and visual evidence. The user can inspect the candidate information, compare images, and manually revise fields such as maker, part number, category, and description.

- **Confirm Phase (`/api/v1/hybrid/index/confirm`)**  
  After user confirmation, the system places the indexing job into an asynchronous queue. The uploaded images are then processed and stored in Milvus, and the task state is tracked through polling.

[Insert Figure 3.3: User Interface demonstrating the Human-in-the-Loop preview and verification workflow]  
*(작성 가이드: 업로드 화면과 preview/결과 확인 화면 스크린샷 2장 삽입)*

**Figure 3.3. Human-in-the-loop indexing workflow, showing metadata preview, duplicate review, and user confirmation before final write-back.**

This design serves two purposes. First, it reduces the risk of blind write-back from uncertain AI output. Second, it keeps the indexed dataset cleaner by requiring a final human confirmation step before persistence.

## 3.4 Duplicate-Aware Ingestion and Multi-Image Design

Industrial-part identification often depends on details that are not visible in a single photo. A front view may reveal the general object class, while a side view, label view, or port view may reveal the decisive identifier. For that reason, the indexing workflow supports multi-image input at the item level.

The design also includes duplicate-aware ingestion. If a new upload appears to match an existing part, the system can reuse the existing `model_id` and append genuinely new images rather than blindly creating a new record. Richer metadata can also be merged instead of discarded. This avoids unnecessary duplication and better reflects the fact that multiple photographs may belong to the same physical item or model family.

## 3.5 Search Workflow and Result Construction

At query time, the system supports both multimodal and lighter-weight retrieval paths.

- In the **multimodal path**, a query image may be combined with user text. The system computes image similarity, optional OCR-based text similarity, caption-based similarity, lexical overlap, and exact identifier boosts.
- In the **text-only path**, a lighter BGE-M3-based model-level retrieval route is available when only textual search is needed.

The key design principle is graceful degradation. If OCR fails or is disabled, image retrieval and metadata-aware matching should still produce useful candidates. If pure visual similarity is insufficient, lexical and identifier signals can reinforce ranking. This makes the retrieval layer more robust under noisy real-world image conditions.

## 3.6 Extended Services: Catalog and Agent Support

The core retrieval pipeline is complemented by two extended service paths:

- **Catalog Retrieval**  
  Reference PDFs and internal documents can be indexed and searched, allowing the system to retrieve supporting material from manuals or catalog-like sources.

- **Agent-Orchestrated Assistance**  
  The agent layer can combine hybrid search, catalog search, and web search to provide broader evidence when simple retrieval alone is not sufficient.

[Insert Figure 3.4: Agent and catalog orchestration path]  
*(작성 가이드: `docs/architecture/ARCHITECTURE_MERMAID.md`의 “5) Catalog + Agent Orchestration Path” 다이어그램 캡처본 또는 대응 UI 스크린샷 삽입)*

**Figure 3.4. Extended orchestration path for catalog retrieval and agent-assisted evidence expansion beyond the core hybrid search workflow.**

These components are not the sole focus of the retrieval benchmark, but they are part of the overall design because they extend the system from a narrow retrieval engine into a more practical listing-assistance workflow.

## 3.7 Evaluation-Oriented Design

The architecture was deliberately designed to support structured evaluation rather than only ad hoc demonstration. Profiling hooks, ablation-friendly configuration switches, and benchmark scripts were prepared so that the following questions could be tested empirically:

- How much do OCR and reranking contribute to retrieval quality?
- What is the latency cost of each configuration?
- How robust is OCR for identifier extraction in industrial imagery?
- Does the system provide enough transparency and editability for user-facing listing support?

To support these questions, the design includes:

- configurable OCR-on/OCR-off paths,
- optional reranker usage,
- separate handling of image, text, caption, and metadata signals,
- asynchronous indexing task status reporting,
- and offline evaluation inputs for index/query splits.

**Table 3.1. Traceability between domain requirements, system components, and evaluation strategy**

[Insert Table 3.1: Requirement-to-component traceability]  

| Domain Requirement (from Ch.1) | Architectural Component (Design) | Evaluation Metric / Strategy (to be Ch.5) |
| :--- | :--- | :--- |
| **Accurate identification of fine-grained parts** | Hybrid Fusion Pipeline (Qwen3-VL + BGE-M3) | Retrieval Effectiveness (Accuracy@1, Accuracy@5) |
| **Robustness against industrial text noise** | Decoupling OCR; using it as a secondary fallback | C2 vs C4 offline benchmark; Character Error Rate |
| **User trust and transparency** | Human-in-the-Loop UI (Preview/Confirm workflow) | User-centred qualitative assessment |
| **Workflow speed for seamless listing** | FastAPI async tasks & isolated model packages | Component-level Latency Profiling (p50/p90) |

*(작성 가이드: 요구사항과 시스템 구성요소, 평가항목을 연결하는 traceability table 삽입)*

This evaluation-oriented structure was important because the project does not claim that one configuration is universally best. Instead, the architecture was built to make those trade-offs observable and measurable.

## 3.8 Design Limitations and Time-Boxed Extensions

The current design prioritises the core prototype workflow within a limited implementation window. As a result, several useful extensions were intentionally deferred. These include metadata-only draft registration, a fuller audited review workflow, more selective OCR-trigger policies, and broader deployment-oriented optimisation.

These limitations do not invalidate the current prototype. Rather, they define the boundary between the implemented research prototype and future productisation work.
