# Smart Image Part Identifier for Secondhand Platforms

University of London  
Bachelor in Computer Science

Final Project  
CM3020 Artificial Intelligence, "Orchestrating AI models to achieve a goal"

Name: SuHun Hong  
Email: hshlalla@naver.com

## Draft Status

This file is the current working draft for the final report. It does not replace the submitted second-stage file at `submission/reports/Draft.docx`. Instead, it rewrites that draft using:

- the current codebase,
- the preliminary and draft feedback,
- the final report guide, and
- the validation artifacts stored under `submission/evidence/report_support_2026-03-10/`.

Figure and appendix references below are intentional placeholders for the final document assembly.

## 1. Introduction

### 1.1 Project Context and Motivation

This project follows the CM3020 template "Orchestrating AI models to achieve a goal". The goal is to build an AI-assisted system that helps identify industrial and electronic parts from user photos in secondhand listing workflows.

This problem matters because secondhand platforms increasingly contain items whose value depends on correct identification at the maker, model, and part-number level. For ordinary consumer products, users can often rely on brand recognition or manual keyword search. For industrial and electronic parts, that is much harder. Many parts look almost identical across variants, while the decisive information may be a small printed or engraved identifier that is difficult to read in user-generated images. Sellers are often non-experts, so listing creation becomes slow, frustrating, and error-prone.

Industry trends suggest that photo-based assistance can reduce friction in marketplace workflows. eBay has shown that image-based retrieval can support large-scale product discovery [2], and Korean secondhand platforms have introduced AI-assisted listing features that infer listing fields from uploaded photos [3]. However, these examples mostly target broad consumer goods. Small parts create a more demanding problem because visual similarity alone is often insufficient and OCR evidence is often noisy.

### 1.2 Problem Statement

The core challenge is that no single modality is reliable enough on its own.

- Visual similarity is useful, but fine-grained part variants can remain ambiguous.
- OCR can recover model numbers or nameplate text, but real images often contain glare, blur, low resolution, stylised fonts, or partial occlusion.
- Marketplace workflows need structured outputs such as maker, part number, and category rather than a generic list of similar images.

For that reason, the project treats part identification as a retrieval-first decision-support problem rather than a closed-set classification problem. The system should:

1. combine image and text evidence when available,
2. degrade gracefully when OCR fails or text is missing,
3. present a shortlist rather than over-claiming a single perfect answer, and
4. return listing-oriented structured outputs with supporting evidence.

### 1.3 Domain and User Requirements

Earlier feedback highlighted that the project concept needed a clearer justification from both the domain and the user perspective. The final design therefore starts from explicit requirements.

#### Domain Requirements

`R1. Open-world inventory and long-tail coverage.`  
Secondhand inventories are not fixed taxonomies. Rare, discontinued, or newly seen parts appear continuously, so a retrieval-based architecture is more appropriate than a classifier that requires retraining whenever categories change [10,21].

`R2. Fine-grained ambiguity.`  
Industrial and electronic parts often differ by small identifiers rather than gross shape. This means image-only matching is often insufficient, especially for visually similar variants [4,10].

`R3. Noisy user images and uncertain OCR.`  
Unlike curated industrial datasets, marketplace photos are taken under uncontrolled conditions. OCR should therefore be treated as uncertain evidence rather than ground truth.

`R4. Listing-ready structured outputs with traceable evidence.`  
A useful marketplace assistant must return fields that can be copied into a listing, while also showing enough evidence to let the user judge whether the suggestion is reliable.

#### User Requirements

A short requirements elicitation survey was conducted with six users familiar with secondhand platforms. Although the sample is small, it provided useful direction for system design.

Key findings from the survey were:

- writing detailed specifications was repeatedly identified as difficult,
- most respondents relied on manual web search or checking labels and manuals,
- most respondents said they would use a photo-based assistant,
- no respondent expressed unconditional trust in AI outputs, and
- transparency and editability were explicitly valued.

These observations motivate the following user requirements.

`U1. Reduce listing friction for non-expert sellers.`  
The system should reduce time spent searching for correct names and specifications.

`U2. Support shortlist-based decision making.`  
Users are more likely to trust a Top-K candidate list than a single unqualified prediction.

`U3. Preserve user control.`  
Users want evidence and the ability to verify or correct results before those results are treated as authoritative.

### 1.4 Implications for System Design

The requirements above motivate a retrieval-first, human-in-the-loop architecture. Instead of predicting a single fixed class, the system orchestrates OCR, multimodal embeddings, vector search, catalog retrieval, and structured result synthesis. It returns a shortlist of candidate parts together with evidence such as OCR text, similarity-based ranking signals, and catalog references when available.

This framing is important because it matches the practical workflow. In secondhand listing creation, success does not require fully autonomous perfect identification. A more realistic goal is to reduce search effort, narrow the candidate space, and give the user enough evidence to make a final decision.

### 1.5 Aim and Objectives

`Aim.`  
To build and evaluate an end-to-end prototype that takes a user photo of a part and produces a Top-5 shortlist of candidate matches together with a concise listing-oriented summary and supporting evidence.

`Objectives.`

`O1. Working MVP workflow.`  
Deliver an end-to-end flow from upload to hybrid retrieval and listing-oriented output.

`O2. Retrieval effectiveness.`  
Measure whether the correct item appears within the shortlist using Accuracy@1 and Accuracy@5, and compare image-only retrieval with hybrid retrieval where suitable.

`O3. OCR robustness.`  
Analyse the quality and limitations of OCR on identifiers such as model numbers or nameplates, using character-level metrics when ground truth is available.

`O4. Interactive feasibility.`  
Measure component-level timing so that retrieval latency can later be summarised with percentile statistics.

`O5. User usefulness and trust.`  
Evaluate, where possible, whether the system reduces listing effort and supports evidence-based verification rather than blind automation.

### 1.6 Report Structure

The remainder of the report is organised as follows. Chapter 2 reviews related work in visual retrieval, OCR, multimodal embeddings, vector databases, and interactive feedback. Chapter 3 presents the system design and explains how the evaluation strategy is embedded into that design. Chapter 4 describes the implementation, focusing on the major modules and algorithms rather than a file-by-file description. Chapter 5 reports the available evaluation evidence, distinguishes completed results from partially completed evaluation work, and critically analyses limitations. Chapter 6 concludes the report and outlines future work.

## 2. Literature Review

### 2.1 Vision-Based Product and Part Retrieval

Content-based image retrieval has evolved from handcrafted features toward deep feature embeddings that capture higher-level semantics. In the consumer domain, large platforms such as eBay have shown that image-based search can operate at marketplace scale and can help users search when they do not know the right keywords [2]. Google Lens and similar systems extend this idea to broader visual search scenarios [8]. These systems demonstrate that image-based retrieval is viable and useful, but they primarily target whole consumer products rather than fine-grained industrial parts.

In industrial contexts, the problem is closer to the present project. Li and Chen show that machine-part recognition can achieve strong accuracy using transfer learning on curated datasets [10]. However, such approaches assume a closed set of known categories. That assumption is poorly aligned with secondhand platforms, where inventory is open-ended and categories can change over time. Retrieval-oriented industrial work is more relevant because it avoids committing the system to a fixed taxonomy, but even there the literature often assumes more homogeneous datasets than those found in real marketplace images [11].

The literature therefore supports two conclusions. First, deep visual embeddings are useful for part retrieval. Second, vision alone is not enough for open-world, fine-grained identification when critical distinctions depend on textual identifiers. This motivates a design that uses vision as one signal among several, not as the entire solution.

### 2.2 Object Detection and Region Focus

Real marketplace photos often contain clutter, multiple objects, packaging, or distracting backgrounds. Practical visual-search systems therefore often insert an object detection stage before embedding generation. VOVA's search-by-image pipeline is a useful example: YOLO is used to crop the relevant object, and feature extraction is then performed on that region before the vectors are stored in Milvus [5].

For secondhand parts, region focus is attractive for two reasons. It may improve the image embedding by reducing background noise, and it may improve OCR by locating a label or nameplate more precisely. However, off-the-shelf detectors rarely include the specific industrial part classes needed in this domain. That means a detector would either need further annotation and training, or would need to operate in a more class-agnostic way. Both introduce complexity. For this reason, the present project treats detection and cropping as an important time-boxed extension rather than as a hard dependency of the MVP.

### 2.3 OCR as a Complement to Visual Retrieval

Prior work on spare-part and manufacturing-part search repeatedly shows that image similarity alone is often insufficient. Grid Dynamics describes a hybrid solution for manufacturing parts in which OCR, image similarity, and keyword search are combined because small parts can differ only in details that are difficult to see visually [12]. This pattern is directly relevant to secondhand parts: a seller may photograph the body of the item, a label, or both, and the system must exploit whichever evidence is available.

However, OCR is inherently noisy. Low resolution, motion blur, glare, curved surfaces, and engraved text can all cause character-level errors. In secondhand listing workflows, a single incorrect character can be enough to identify the wrong part. The literature therefore supports a key design principle of this project: OCR should be treated as probabilistic evidence that can strengthen retrieval, not as an unquestioned source of truth.

### 2.4 Multimodal Embeddings and Text Retrieval

Multimodal embedding models provide a natural basis for hybrid retrieval. CLIP-style models allow images and text to be compared in related embedding spaces [13,21], and more recent open models such as BGE-VL provide practical vision-language embeddings suitable for retrieval pipelines. For the text side, BGE-M3 is particularly relevant because it supports multilingual retrieval and multiple retrieval modes in a single framework [6]. This matters in secondhand platforms where listings, OCR text, and brand terminology may mix Korean, English, and shorthand identifiers.

The main limitation is domain mismatch. These models are trained on broad internet-scale data, not specifically on low-quality close-ups of industrial parts and worn labels. This means that generic embeddings are useful baselines but not complete solutions. The implication is that strong retrieval performance depends not only on the embedding model but also on system-level orchestration, metadata handling, lexical matching, and post-retrieval reasoning.

### 2.5 Vector Databases and Hybrid Search

Milvus is relevant not simply because it stores vectors, but because it supports the kind of multi-channel retrieval required for this problem [15,16]. A single item can be represented by an image vector, OCR-derived text vectors, attribute text, caption text, and model-level aggregated metadata. Hybrid retrieval then becomes a process of querying one or more of these channels, merging candidates, and fusing evidence.

This is an important design distinction. The challenge is not merely nearest-neighbour search over one embedding space. The real challenge is deciding how to combine signals when some are missing, noisy, or contradictory. The literature provides the infrastructure pattern for hybrid search, but leaves open the question of how modality weighting, lexical signals, and feedback should be handled in uncertain real-world settings. That gap is directly addressed by the present project's retrieval-first orchestration logic.

### 2.6 Interactive Feedback and Continuous Improvement

Interactive retrieval literature shows that user feedback can improve system usefulness over time. Relevance-feedback ideas from classical CBIR have been revisited in CLIP-based interactive retrieval, where user responses help refine ranking without retraining the encoder [18]. In secondhand marketplaces, the practical equivalent is that user confirmation or correction can be logged and reused later.

This literature does not imply that a full online learning loop must be implemented in the MVP. Instead, it supports the idea that the system should preserve traces of decisions and be designed so that confirmation data can later improve ranking, reranking, or metadata quality. In this project, that idea appears in the design for model-level metadata accumulation and in the decision to keep the system human-centred rather than fully autonomous.

### 2.7 Summary of Gaps and Design Implications

The literature suggests a useful recipe: deep visual retrieval, OCR, multimodal embeddings, scalable vector search, and user feedback. But important gaps remain for secondhand parts:

- most prior systems assume either closed-set classification or more curated datasets than real marketplaces provide,
- published industry examples often focus on whole products rather than fine-grained parts,
- OCR is recognised as useful but rarely analysed as uncertain evidence in uncontrolled user imagery,
- multilingual and heterogeneous text is often under-discussed, and
- listing-oriented structured output is usually secondary to search itself.

The project therefore positions its contribution at the system level. It is not a new foundation model. It is an orchestrated pipeline designed for an open-world, noisy, evidence-sensitive workflow in which the user remains responsible for the final decision.

## 3. Design

### 3.1 Design Goals and Constraints

The system is designed around five goals that directly follow from the domain and user requirements.

`G1. Open-world identification.`  
The problem is framed as retrieval rather than classification so that new items can be supported by indexing instead of retraining.

`G2. Fine-grained discrimination through multimodal evidence.`  
The design must combine image similarity with text-derived evidence because small identifiers may carry the most important information.

`G3. Robustness to missing or noisy evidence.`  
OCR may fail, captions may be unavailable, and metadata may be incomplete. The system therefore needs graceful fallbacks instead of assuming all modalities are present.

`G4. Listing-oriented structured output and transparency.`  
The output must be actionable for a seller, not just technically correct for a benchmark. That means returning a shortlist, key listing fields, and enough evidence to support verification.

`G5. Evaluation traceability.`  
The system must be instrumented so that later evaluation can measure retrieval quality, OCR behaviour, latency, and engineering stability in a repeatable way.

### 3.2 System Overview

The final architecture is organised into four main runtime layers and one supporting debug interface.

`1. Web interaction layer.`  
The primary user-facing interface is a React web application backed by a FastAPI service. The web application supports search, indexing, catalog operations, and an agent chat interface.

`2. Orchestration layer.`  
The central coordinator is the hybrid-search orchestrator. It is responsible for running OCR, generating embeddings, querying Milvus, merging candidates, and computing final ranking scores.

`3. Perception and representation layer.`  
This layer includes OCR, image embeddings, text embeddings, and optional caption generation. It transforms raw inputs into queryable representations.

`4. Retrieval and knowledge layer.`  
Milvus stores vectors and metadata for several modalities. A separate catalog index supports retrieval over internal PDF documents and technical references.

`5. Debug and development interface.`  
Gradio is retained as a developer-facing demo and debugging tool. It is useful for isolating model-pipeline behaviour from the full web stack, but it is not the primary user interface in the current system.

This structure addresses an issue identified during the report revision process: the system should not be described as if Gradio were the main final interface. In the current architecture, the main end-to-end path is React plus FastAPI, while Gradio is a supplementary diagnostic UI.

### 3.3 Data Model and Retrieval Schema

The design separates model-level information from image-level evidence.

At the conceptual level, the important entities are:

- `Model / Part`: `model_id`, maker, part number, category, description,
- `ImageAsset`: an image associated with a model,
- `OCRText`: raw and normalised OCR output,
- `Embeddings`: image, text, and optional caption vectors.

At the storage level, the current system uses multiple Milvus collections rather than a single monolithic table. In practice this supports several search channels:

- `image_parts` for image vectors,
- `text_parts` for OCR-derived or attribute-derived text vectors,
- `attrs_parts` and `caption_parts` for additional textual channels,
- `model_texts` for model-level aggregated metadata,
- `catalog_chunks` for internal document retrieval.

This separation is useful because the retrieval problem is not homogeneous. Image evidence, OCR evidence, metadata evidence, and catalog evidence have different characteristics and may be available in different situations.

### 3.4 Core Workflows

#### 3.4.1 Indexing Workflow

The indexing workflow ingests one or more images plus model metadata.

The steps are:

1. validate required fields such as `model_id`,
2. run OCR where appropriate and normalise the resulting text,
3. generate image and text embeddings,
4. upsert vectors and metadata into the relevant Milvus collections, and
5. update model-level aggregated text where structured metadata is available.

The workflow is intentionally repeatable. Additional images or corrected metadata can be indexed later without retraining the core embedding models.

#### 3.4.2 Search Workflow

The main search workflow is query-time identification for listing assistance.

The user may provide:

- an image,
- optional text,
- an optional part-number hint,
- and a requested `top_k`.

The orchestrator then:

1. preprocesses the image,
2. runs OCR if an image exists,
3. generates image and text embeddings as available,
4. issues search requests against the relevant Milvus collections,
5. merges candidates by `model_id`, and
6. produces a ranked shortlist with score decomposition and metadata.

If OCR is missing or unreliable, the system falls back to image-oriented retrieval instead of failing. If only text is provided, text-only search remains possible.

#### 3.4.3 Ranking and Evidence Fusion

The ranking logic is deliberately two-stage.

First, dense similarity is computed over whichever modalities are actually available. This includes image similarity, OCR-text similarity, caption-text similarity, and text-query similarity where appropriate.

Second, the dense score is blended with lexical and specification-oriented signals. In the current implementation, the final result includes component fields such as:

- `image_score`,
- `ocr_score`,
- `caption_score`,
- `lexical_score`,
- `spec_match_score`.

This is not only useful for ranking. It is also useful for transparency. A shortlist result can be explained in terms of which channels contributed to the match, which aligns with the user requirement that the system should not behave like a black box.

#### 3.4.4 Catalog Retrieval and Agent-Oriented Reasoning

The system also includes a catalog retrieval path for internal PDFs and technical documentation. Documents are chunked, embedded, and stored in a separate vector collection so that the system can retrieve internal textual evidence when users ask documentation-style questions or when additional context is needed.

An agent layer can then orchestrate tools such as hybrid search, catalog search, web search, and price extraction. This extends the system from a single retrieval pipeline into a broader decision-support workflow. Importantly, tool orchestration does not replace the hybrid search engine; it builds on it.

### 3.5 Human-in-the-Loop Boundary

The project has always aimed to be human-in-the-loop, but the final report must state clearly what is implemented and what remains planned.

The current system supports human review in three ways:

1. it returns Top-K candidates rather than only one final answer,
2. it exposes supporting evidence such as OCR text and modality scores, and
3. it now keeps agent-driven Milvus updates disabled by default unless the operator explicitly opts in.

The third point is important. During the final validation pass, the default agent option `update_milvus` was changed from `true` to `false`, and the web UI was updated to require an explicit operator decision before writeback is allowed. This reflects a safer interpretation of the human-review requirement: uncertain predictions should not automatically become new knowledge.

What is not yet fully implemented is a complete production-grade `accept / edit / reject` workflow with audit logging across the main search UI. That remains a high-priority next step, and the final report should describe it as partial or planned rather than fully complete.

### 3.6 Evaluation-Oriented Design

Feedback on earlier submissions emphasised that the evaluation strategy should be embedded into the design itself. The system was therefore designed with traceability in mind.

The design supports four evaluation dimensions:

`E1. Retrieval effectiveness.`  
The shortlist design is evaluated primarily with Accuracy@1 and Accuracy@5 because the intended outcome is shortlist usefulness rather than autonomous top-1 perfection.

`E2. OCR robustness.`  
When ground-truth identifier text exists, OCR quality should be evaluated at character level, because a single wrong character can imply a different part.

`E3. Latency and interactivity.`  
The system captures component-level timings so that later p50, p90, and p95 summaries can be produced.

`E4. Engineering reliability.`  
Regression testing is also relevant because a system-level project can fail in practice even when the model idea is sound. API behaviour, schema defaults, and lightweight model-package imports all affect whether the prototype is usable and reproducible.

### 3.7 Design Limitations

The design also has explicit limits.

- OCR remains brittle on reflective, blurred, or distant labels.
- Generic multimodal embeddings may miss domain-specific distinctions.
- The current system does not yet include a full object detector or cross-encoder reranker.
- Full quantitative automation for hybrid ablation, OCR CER, and latency percentiles is not yet complete.

These are not accidental omissions. They are the main boundaries that separate the implemented MVP from the planned next stage of refinement.

## 4. Implementation

### 4.1 Repository Structure and Runtime Setup

The implementation is organised as a small monorepo with separate application and model packages. This structure supports modular development and helps explain the system at the level of deployable components rather than isolated scripts.

The main folders are:

- `apps/web`: React + Mantine frontend,
- `apps/api`: FastAPI service,
- `apps/demo`: Gradio developer demo,
- `packages/model`: hybrid retrieval and indexing core,
- `docs`: internal documentation and report-writing material,
- `submission`: prior submissions, feedback, guides, and report evidence.

This layout evolved from an earlier prototype structure into a cleaner separation between product-facing code, model logic, and documentation. It also supports the report's focus on orchestration, because the system is now clearly divided into frontend, API, model core, and evidence layers.

### 4.2 Frontend and API Layer

The web frontend contains four main views.

`Search.`  
Allows the user to submit text, an image, or both. Images are encoded in the browser and sent to the API for hybrid search.

`Index.`  
Allows new catalog items to be indexed with images and metadata.

`Catalog.`  
Allows PDF documents to be indexed and searched as internal textual evidence.

`Agent chat.`  
Allows the user to ask higher-level questions while optionally attaching an image.

On the API side, these map to routes under:

- `/api/v1/hybrid/*`,
- `/api/v1/catalog/*`,
- `/api/v1/agent/*`,
- `/api/v1/auth/*`.

This split is important because it keeps the user-facing workflow separate from the model implementation. The API is the contract boundary. The model package can evolve internally while the frontend continues to call stable endpoints.

### 4.3 Hybrid Search Core

The core retrieval behaviour is implemented in the hybrid-search pipeline within `packages/model/smart_match`. The orchestrator collects candidates from multiple channels and merges them at model level rather than treating each image independently.

At query time, the implementation can use several channels:

- image embeddings,
- OCR-derived text embeddings,
- caption-derived text embeddings,
- direct user text,
- lexical matches,
- specification-token matches such as part numbers or electrical ratings.

The search path is retrieval-first and model-centric. That means the system first gathers candidate evidence from available channels and only then performs score fusion and re-ranking.

In simplified form, the logic is:

1. preprocess query image if present,
2. search image vectors if image embeddings are available,
3. search OCR/caption/text vectors where text evidence exists,
4. merge candidates by model identifier,
5. compute dense and lexical/specification-aware scores,
6. produce Top-K with score decomposition and attached metadata.

This is technically significant for the project because it operationalises the report's central idea: the prototype is not a single model predicting a label, but a coordinator that fuses multiple uncertain signals into a shortlist.

### 4.4 OCR, Embeddings, and Optional Captioning

The perception and representation stages use practical off-the-shelf models:

- PaddleOCR-based pipelines for OCR,
- BGE-VL for image embeddings,
- BGE-M3 for text embeddings,
- optional caption generation using GPT or Qwen-based backends.

The design choice here is pragmatic. The project does not claim to outperform specialised models through novel training. Instead, it uses strong available components and focuses on how they are orchestrated, how their outputs are normalised, and how their uncertainty is managed.

The OCR pipeline includes fallback behaviour and text normalisation. Captioning is optional and environment-dependent rather than mandatory. This matters because one of the practical lessons from implementation was that some channels should be treated as conditional helpers rather than assumptions.

### 4.5 Catalog RAG and Tool-Oriented Agent Integration

The project moved beyond the initial retrieval-only prototype by adding catalog retrieval and an agent layer.

For catalog retrieval:

1. PDF text is extracted page by page,
2. OCR fallback is used when pages are image-based,
3. text is chunked and embedded,
4. dense, lexical, and specification-aware retrieval is performed over those chunks.

For the agent layer, the system can call tools such as:

- `hybrid_search`,
- `catalog_search`,
- `web_search`,
- `extract_prices`,
- `gparts_search_prices`,
- and, when explicitly permitted, metadata update tools.

This is an important originality point for the report. The project is not only a visual search tool. It is an orchestrated AI workflow in which retrieval, internal documents, and external tools can be combined to support a user decision.

### 4.6 Reliability, Safety, and Observability Improvements

During the final validation pass, several implementation changes were made specifically to improve safety, reliability, and reportable evidence quality.

`1. Agent writeback safety.`  
The request schema for agent chat was changed so that `update_milvus` defaults to `false` rather than `true`. This prevents uncertain tool-assisted outputs from automatically polluting the knowledge base.

`2. Explicit writeback control in the UI.`  
The agent chat page now includes a switch so that Milvus updates happen only when the operator explicitly enables them.

`3. Hybrid search timing instrumentation.`  
Structured timing capture was added for major search phases including preprocessing, image search, OCR search, caption search, text search, model fetch, finalisation, and total time.

`4. Lightweight testability improvements.`  
The model package previously had eager imports that made simple pytest collection fragile because heavyweight runtime dependencies such as `torch` were pulled in unnecessarily. Package-level imports were changed to lazy imports so that lightweight tests can import the needed submodules without requiring the full runtime stack.

`5. Logging and validation improvements.`  
The broader codebase also includes improved request logging, upload validation, payload-size protection, and reduced duplicate logging in the API layer.

These changes are important for the final report because they show that the system was not only extended functionally but also hardened operationally.

### 4.7 Implementation Stability Evidence

The validation artifacts stored under `submission/evidence/report_support_2026-03-10/` record a reproducible engineering validation pass carried out on March 10, 2026.

The recorded results are:

- API tests: `12 passed, 1 warning in 5.37s`
- model tests: `4 passed in 0.09s`

These tests are not a substitute for the retrieval benchmarks, but they are useful evidence that key regression points are now automatically checked. In particular, the API tests verify the new safe behaviour for agent writeback defaults, and the model-side tests confirm that the package can be imported and tested in a lightweight environment after the lazy-import changes.

### 4.8 What Is Implemented Versus Partial

To keep the implementation chapter aligned with the actual codebase, the current status should be stated explicitly.

`Implemented now:`

- end-to-end indexing and hybrid search,
- React web UI backed by FastAPI,
- developer-facing Gradio demo,
- OCR and embedding integration,
- hybrid score fusion with transparent score decomposition,
- catalog PDF indexing and retrieval,
- agent tool orchestration with source exposure,
- safe-by-default agent writeback control,
- latency instrumentation on the hybrid search path,
- pytest-based regression coverage for selected API and model behaviours.

`Partially implemented or not yet complete:`

- full search-UI accept/edit/reject workflow with audit logging,
- complete OCR CER benchmark automation,
- full hybrid ablation automation across fixed splits,
- latency percentile reporting over a benchmark batch,
- cross-encoder reranking,
- object detection or focused cropping for labels and parts,
- a completed usability study with measured user outcomes.

This distinction is essential because one of the main risks in the earlier draft was that some planned capabilities could be misread as completed implementation.

## 5. Evaluation

### 5.1 Evaluation Strategy

The evaluation strategy follows directly from the project goals. Because the system is intended as an assistive identification tool rather than a fully autonomous classifier, the central question is whether it can reliably place the correct item into a small shortlist and provide supporting evidence that helps the user make the final choice.

The evaluation is therefore divided into four parts:

`1. Retrieval effectiveness.`  
Does the correct model appear in the Top-K results?

`2. OCR robustness.`  
When identifier text is visible, how reliable is the OCR signal and what failure modes matter most?

`3. Interactive feasibility.`  
Can the system be measured at component level so that latency bottlenecks are visible?

`4. Engineering reliability.`  
Do recent code changes behave as intended and remain testable?

The primary retrieval metrics are Accuracy@1 and Accuracy@5. Accuracy@5 is especially important because it matches the system's shortlist-based design. If the correct part appears within Top-5, the user can still complete the workflow successfully. This is a better reflection of the intended use case than top-1 accuracy alone.

For OCR, character error rate is the most relevant metric because a single character mistake can completely change the identity of a part number. For latency, percentile statistics are more useful than simple means because long delays matter most in interactive workflows.

### 5.2 Evaluation Inputs and Evidence Types

The final report uses two kinds of evidence:

`A. Retrieval evidence already available from the earlier draft-stage experiments.`  
This includes the image-only baseline on two dataset splits.

`B. Engineering validation evidence collected during the March 10, 2026 validation pass.`  
This includes regression test results and the newly added search-timing instrumentation.

This distinction matters because the final report should not overstate what has already been fully benchmarked. Retrieval baseline results are available. OCR aggregate benchmarking, hybrid ablation tables, and latency percentile summaries are not yet fully collected in the current artifact bundle.

### 5.3 Retrieval Results: Image-Only Baseline

The strongest quantitative retrieval evidence currently available is the image-only baseline already established during the draft-stage evaluation.

| Dataset | Accuracy@1 | Accuracy@5 |
|---|---:|---:|
| Random 1000 models | 0.287 | 0.791 |
| Category-sampled 500 models | 0.306 | 0.812 |

These numbers are meaningful for two reasons.

First, they show that retrieval-first identification is viable. Even with image evidence alone, the correct item appears within the Top-5 for a large proportion of queries.

Second, they also show the limit of vision-only retrieval. The random split falls slightly below the MVP threshold of 0.80 at Top-5, which supports the design decision to add OCR, text, metadata fusion, and human verification rather than relying on image similarity alone.

The category-sampled result is slightly stronger than the random split, suggesting that performance is sensitive to dataset composition and to how strongly the evaluated categories align with visible appearance. This again supports the decision to treat the system as a shortlist assistant rather than as a single-answer classifier.

### 5.4 Observed Failure Patterns

Initial inspection of failure cases suggests three recurring patterns.

`Visually similar variants.`  
Different parts may have near-identical shape while differing only by a model number that is not visible in the image.

`Background clutter or poor framing.`  
If the part occupies only a small part of the image, the embedding can capture irrelevant scene information.

`Catalog absence or incomplete coverage.`  
If the correct item is not in the index, the system can only return approximate neighbours.

These patterns are consistent with both the literature review and the implemented roadmap. They justify OCR fusion, future region focus, and the need for explicit uncertainty-aware user review.

### 5.5 OCR Robustness: Current Evidence and Missing Quantitative Benchmark

OCR remains an important but incomplete part of the evaluation story.

The qualitative evidence already collected in the draft stage highlights several real failure modes:

- small serial numbers,
- blur,
- glare,
- mixed vertical and horizontal text,
- stylised logos and fonts,
- engraved labels,
- distant nameplates.

This qualitative analysis is valuable because it shows why OCR should not be treated as a hard filter. In a part-identification task, OCR errors do not merely reduce text quality; they can actively mislead the retrieval process if given too much weight.

What is not yet complete is the aggregate CER benchmark over a controlled identifier set. The protocol for that benchmark is defined, but the final report should describe it as pending rather than conducted. This is one of the clearest places where the earlier draft needed correction.

### 5.6 Latency Evaluation: Instrumented but Not Yet Fully Summarised

The current codebase now captures timing information for the hybrid search path at multiple stages:

- preprocessing,
- image search,
- OCR search,
- caption search,
- text search,
- model fetch,
- finalisation,
- total time.

This is a useful implementation step because it makes later p50, p90, and p95 analysis possible. However, the final report should not claim that a complete latency benchmark has already been conducted unless those batch summaries are actually produced before submission.

The correct interpretation at present is:

- latency instrumentation is implemented,
- the measurement protocol is defined,
- bottleneck-oriented profiling is now feasible,
- but the quantitative percentile summary remains future work.

### 5.7 Engineering Validation Results

Although engineering validation is not the same as retrieval benchmarking, it is still relevant to evaluating a system-level AI project.

The March 10, 2026 validation pass recorded:

| Validation area | Result |
|---|---|
| API pytest run | `12 passed, 1 warning in 5.37s` |
| Model pytest run | `4 passed in 0.09s` |

The most relevant verified behaviours were:

- `update_milvus` defaults to `false`,
- `update_milvus=true` is still accepted when explicitly requested,
- model-package import behaviour is stable enough for lightweight pytest collection after lazy-import changes.

This evidence is modest compared with full benchmark automation, but it is still important because it verifies that the latest safety and testability changes behave as intended.

### 5.8 Objective-by-Objective Assessment

| Objective | Current status | Assessment |
|---|---|---|
| `O1` Working MVP workflow | Supported | The end-to-end workflow exists across web, API, and model layers. |
| `O2` Retrieval effectiveness | Partially supported | Image-only baseline is available; full hybrid ablation remains pending. |
| `O3` OCR robustness | Partially supported | Qualitative failure analysis exists; CER benchmark still needed. |
| `O4` Interactive feasibility | Partially supported | Timing instrumentation is implemented; percentile summaries are not yet reported. |
| `O5` User usefulness and trust | Partially supported | The design is human-centred, but a completed usability study is still missing. |

This table gives a more honest final assessment than the earlier wording that implied more evaluation work had already been completed.

### 5.9 Critical Discussion

The evaluation evidence supports three main conclusions.

First, the retrieval-first architecture is justified. The image-only baseline is already strong enough to show that shortlist-based identification is feasible, especially on the category-sampled split.

Second, image-only retrieval is not sufficient for the full problem. The random split result, the observed OCR failure patterns, and the known difficulty of near-duplicate parts all show why multimodal evidence and human verification are necessary.

Third, the project is stronger as an identification assistant than as an autonomous identifier. The current system is well positioned to reduce effort and narrow the search space, but it should not be described as if it can safely and consistently produce final authoritative answers without review.

The main limitations that remain are:

- OCR brittleness under realistic image noise,
- incomplete automation of hybrid benchmarking,
- lack of a complete accept/edit workflow in the main UI,
- and lack of a completed usability study.

These are important because they define the difference between a strong prototype and a mature deployed system.

## 6. Conclusion and Future Work

### 6.1 Summary of Contributions

This project presented a Smart Image Part Identifier for secondhand platforms under the CM3020 theme of orchestrating AI models to achieve a goal. Its main contribution is not a new standalone model, but a coherent system that combines several AI components into a practical identification workflow.

The most important contributions are:

`1. Retrieval-first system design for an open-world problem.`  
The project treats part identification as an open-world retrieval task rather than a fixed classifier problem.

`2. Multimodal evidence fusion.`  
The system combines image retrieval, OCR-derived text, optional captions, metadata matching, and catalog evidence rather than relying on one signal.

`3. Listing-oriented output.`  
The result is designed around seller workflows, returning a shortlist and listing-ready fields rather than only a visual similarity score.

`4. Tool orchestration beyond a single search call.`  
The architecture now includes catalog retrieval and agent-level tool use, making the system a broader decision-support pipeline.

`5. Safer and more testable implementation.`  
The final validation pass improved writeback safety, timing observability, and pytest-level regression coverage.

### 6.2 Final Positioning of the Prototype

The most accurate final description of the system is that it is a `retrieval-first, human-in-the-loop identification assistant`.

That wording matters. It reflects what the current evidence actually supports:

- the prototype can help users narrow the candidate space,
- it can surface relevant metadata and evidence,
- it can integrate internal and external tools when needed,
- but it should not yet be framed as a fully autonomous identification engine.

### 6.3 Limitations

Several limitations remain.

First, OCR remains fragile for precisely the kinds of images that matter in secondhand listing scenarios. Second, visually similar parts still create ambiguity when decisive identifiers are missing or unreadable. Third, the project does not yet have complete quantitative automation for hybrid ablation, OCR CER, and latency percentile reporting. Fourth, the current human-review implementation is safer than before, but still does not provide a complete production-grade accept/edit workflow in the main UI.

These limitations do not invalidate the project. Instead, they identify the exact boundary between what has been demonstrated and what still needs to be built.

### 6.4 Future Work

The highest-priority next steps are clear.

`1. Complete evaluation automation.`  
Build fixed-split retrieval evaluation scripts, OCR benchmarking, and latency aggregation so that all major claims can be reproduced directly from code.

`2. Add a full human-review writeback workflow.`  
Introduce explicit accept/edit/reject actions with audit logging and review-queue behaviour for low-confidence cases.

`3. Improve ranking quality for hard cases.`  
Explore cross-encoder reranking, confidence calibration, and specification-aware post-processing.

`4. Improve evidence extraction from images.`  
Investigate label detection, nameplate cropping, and image-quality assessment to make both OCR and visual retrieval more robust.

`5. Complete user-centred validation.`  
Run the planned usability study to measure effort reduction, user trust, and editing behaviour directly.

Taken together, these next steps would move the system from a strong prototype toward a more defensible and deployable listing assistant.

## References and Appendices Note

For final assembly, this draft should reuse and clean up the bibliography from `submission/reports/Draft.docx`, with consistent citation formatting and updated wording in the evaluation and implementation chapters. Figure references should also be aligned with the final selected screenshots, diagrams, and tables.
