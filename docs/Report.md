University of london
Bachelor in Computer Science

Final Project
Smart Image Part Identifier for Secondhand Platforms
CM3020 Artificial Intelligence, “Orchestrating AI models to achieve a goal”






Name: SuHun Hong
Email: hshlalla@naver.com









1	Introduction (988/1000)
This project follows template CM3020 Artificial Intelligence “Orchestrating AI models to achieve a goal”,
1.1	Background and Motivation
Secondhand marketplaces increasingly include industrial and electronic parts where accurate identification (maker, model, part number/serial number) is essential for searchability, pricing, and buyer trust. However, sellers often lack domain knowledge, and correct identifiers are frequently difficult to capture in real photos due to glare, blur, small text, wear, and partial occlusion. Traditional approaches to matching parts are often slow and frustrating, requiring manual comparison against catalogues or reference materials [1].

Industry trends show that photo-based assistance can reduce friction in online marketplaces. For example, eBay has demonstrated image-based search at marketplace scale, enabling users to search with photos rather than keywords [2]. In Korea, secondhand platforms have also introduced AI-assisted listing tools that analyse uploaded images to help generate listing fields, indicating strong user interest in reducing listing effort [3]. Despite these advances, identifying parts is typically harder than identifying complete consumer products because many components are visually near-identical across variants, while the decisive signal may be a small identifier (model code, part number) that is difficult to read reliably from user-generated images.

1.2	Problem Statement
Generic image similarity alone is often insufficient for robust part identification. Academic and industrial studies report that industrial components can be difficult to distinguish purely by appearance, particularly under domain constraints such as limited labelled data and fine-grained variations [4,10]. In addition, OCR is valuable but inherently noisy in the secondhand setting (blur, glare, stylised fonts, engraved text), so OCR output should be treated as uncertain evidence rather than ground truth.

Therefore, an effective solution should:
(i) combine visual similarity with textual evidence when available,
(ii) degrade gracefully when OCR fails or text is missing, and
(iii) produce structured outputs suitable for listing workflows rather than returning only visually similar images.

1.3	Domain and User Requirements
To address earlier feedback, this section makes explicit the requirements that justify the project concept and drive the design.

1.3.1	Domain requirements
R1. Open-world inventory and long-tail coverage.
Secondhand inventories evolve continuously; new and rare parts appear frequently. A closed-set classifier would require retraining to add new classes, making it impractical for an evolving catalogue. A retrieval-first approach is better suited because new items can be supported by indexing (not retraining), and retrieval naturally supports Top‑K candidate outputs.

R2. Fine-grained ambiguity and need for identifiers.
Industrial and electronic parts often have subtle differences or near-duplicates where appearance alone is not decisive. Prior work indicates part recognition is challenging under such fine-grained conditions, motivating the use of multiple signals beyond vision-only features [4,10].

R3. Noisy, user-generated images and uncertain OCR.
Marketplace photos are uncontrolled. OCR can provide decisive evidence when nameplates or labels are readable, but it can fail frequently under blur, glare, occlusion, and low resolution. The system must therefore fuse OCR with vision and provide a robust fallback path when OCR is empty or low confidence.

R4. Structured listing outputs and traceable evidence.
A listing assistant must output structured fields that map to listing form inputs (e.g., maker, model/part number, category). It should also present lightweight evidence (e.g., OCR text preview) to support trust and reduce incorrect listings, rather than operating as a black box.

1.3.2	User requirements

A short requirements elicitation survey (n=6) was conducted with users familiar with secondhand platforms (Appendix C). The responses indicate that identifying correct specifications and writing detailed listing information are among the most difficult parts of listing creation. Most respondents rely on manual search or checking labels/manuals to identify items, and while the majority indicated they would likely use a photo-based identification tool, they did not express unconditional trust in AI-generated results. Instead, users consistently emphasised the need for transparency and the ability to edit AI outputs.

These findings motivate the following user-centred requirements:

U1. Reduce listing friction for non-expert sellers.
The workflow should help sellers who do not know the correct technical name/model to produce listing-ready information with minimal effort.

U2. Provide controllable uncertainty management via Top‑K.
Because part identification can be ambiguous, the system should return a shortlist (Top‑5) rather than a single definitive prediction, enabling users to confirm the correct candidate.

U3. Support transparency and correction (human-in-the-loop).
Users must be able to accept or edit suggested fields before finalising the listing. This aligns with interactive retrieval and relevance feedback approaches, where user feedback can improve ranking and system usefulness over time [18].

1.4	Aim and Objectives
Aim.
To build and evaluate an end-to-end prototype that, given a user photo of a part, produces a Top‑5 shortlist of candidate matches together with a concise structured “Listing Summary” containing listing-ready metadata and evidence.

Objectives (MVP-driven).
O1. Working MVP workflow.
Deliver an end-to-end flow: upload → OCR (when applicable) → hybrid retrieval → Top‑5 candidates + Listing Summary → user confirm/edit.

O2. Identification performance (retrieval).
Measure retrieval performance using Accuracy@{1,5} on predefined evaluation splits, and assess improvement of hybrid retrieval over an image-only baseline when OCR/text evidence is available.

O3. OCR quality for identifiers.
Measure OCR quality for identifier text using a character-level metric, and document key failure modes (blur, glare, occlusion, low resolution).

O4. Workflow latency.
Measure component-level timings (OCR, retrieval) and report percentile latencies to assess interactive feasibility.

O5. Practical usefulness for listing creation.
Conduct a small usability evaluation to assess whether the workflow reduces effort and supports user trust, translating findings into a prioritised improvement backlog.

1.5	Report Structure
The remainder of this report is organised as follows. 
Chapter 2 reviews related academic and industry work on vision-based retrieval, OCR, multimodal embeddings, vector databases, and interactive feedback. Chapter 3 presents the system design, including architecture, data model, and user interaction flow. 
Chapter 4 describes the implementation, focusing on the main algorithms and code structure and providing visual evidence (screenshots/graphs). 
Chapter 5 reports initial evaluation results and critically analyses limitations and improvements. 
Chapter 6 concludes with a summary of contributions and future work directions.
2 Literature Review
2.1 Vision based product and part retrieval
Content based image retrieval (CBIR) has a long history in computer vision, starting from hand crafted features such as color histograms, textures and shapes, and evolving towards deep feature embeddings that capture higher level semantics. In the consumer domain, large e commerce platforms have successfully deployed visual search to reduce the friction of product discovery. eBay’s “Image Search” and “Find It on eBay” features allow users to upload a photo and retrieve visually similar listings from more than a billion items by encoding images with a convolutional neural network and comparing them in an embedding space.[2] These systems demonstrate that large scale visual product retrieval is technically feasible and that image based queries can complement or even replace textual descriptions when users do not know the right keywords.
More recently, Google Lens has generalized this paradigm into a multi domain visual search assistant. Users can point their camera at objects, text, or scenes and obtain visually similar items or related web content, powered by CNN based models trained on massive labeled datasets.[8] Commercial visual search has also been integrated into other shopping experiences, such as Amazon’s product search via smartphone cameras or within social apps, where a photo is used to locate the corresponding product on Amazon.[9] These systems show clear user value, but they mainly target whole consumer products (clothes, furniture, decor, etc.), not fine grained parts with subtle differences.
In industrial contexts, the problem is closer to the proposed project. Li and Chen investigate recognition of industrial machine parts using transfer learning on an Inception v3–based CNN, addressing the challenge of small sample sizes by fine tuning a pretrained model on limited part images.[10] Their model achieves strong classification accuracy on a curated dataset of machine parts, illustrating that domain specific fine tuning can work even when collecting labeled images is costly. However, their approach assumes a closed set of known part categories. Adding a new part type requires collecting labeled data and retraining or extending the classifier, which is difficult to scale to the long tail and ever changing inventory of a secondhand marketplace.
Shim et al. focus on content based image retrieval for industrial material images rather than classification.[11] They highlight that industrial datasets are typically small, expensive to acquire, and differ significantly from everyday photographic images. Their CBIR system uses deep features plus encoded physical properties, enabling experts to infer material characteristics from visually similar samples without re running physical experiments. This work is important as it shows that retrieval based methods are well suited to industrial imagery and can support expert workflows. Nevertheless, the retrieval domain is still relatively homogeneous (materials of a similar type) and does not integrate textual cues such as labels or part numbers, which are crucial in secondhand parts identification.
Taken together, these works suggest that (1) deep visual embeddings are effective for both consumer and industrial image retrieval, but (2) pure vision approaches struggle with open world, fine grained part identification on heterogeneous marketplaces. This motivates a hybrid design that augments visual retrieval with additional signals such as text and metadata, and that treats the problem as open set retrieval rather than closed set classification.
2.2 Object detection for region focus
Real world user images often contain cluttered backgrounds, multiple objects, or irrelevant regions. Many practical visual search systems therefore incorporate an object detection stage before feature extraction. VOVA, an e commerce platform, built a “shop by image” feature using a two stage pipeline: YOLO (You Only Look Once) is used on user uploaded photos to detect the product region, and ResNet then extracts feature vectors from the detected crop; Milvus serves as the vector database for similarity search.[5] This architecture significantly improves retrieval quality, because the embedding model sees a tightly cropped object rather than the whole scene.
For secondhand parts, the same issue arises: users might upload a photo of a workbench with many parts, or a broader shot that includes background textures. Using a detector such as YOLO to isolate the candidate part region before embedding is therefore an attractive strategy. However, off the shelf detectors are usually trained on generic classes (e.g., COCO: “bottle”, “chair”, “car”) and do not include specific industrial or electronic parts. This creates a gap: either the system must treat detection as class agnostic (“find any salient object”) or invest in further fine tuning with annotated part bounding boxes—both add complexity and data requirements. The proposed project adopts the detection first architecture demonstrated by VOVA but must cope with the absence of ready made classes for arbitrary secondhand goods, making robustness to detection failures an important design concern.
2.3 Vision + OCR hybrid systems for parts
A recurring insight in prior work on spare part identification is that visual appearance alone is often not enough. Grid Dynamics describe a “visual search for manufacturing parts” solution where engineers can find small replacement parts using a combination of image similarity, OCR, and extended keyword search.[12] They note that micro parts may differ only in details that are hard to see, and in many cases the only reliable identifier is a model or serial number printed on the part or its packaging. Their system therefore supports searches based on a photo of either the part itself or its label/box; a CNN produces visual embeddings, while OCR extracts text that can be matched to part catalogs or used as search keywords.
This hybrid pattern—visual retrieval plus text extraction—directly informs the present project. In a secondhand setting, a seller might photograph only the label, only the object, or both. A robust system must therefore (1) detect relevant text regions, (2) extract part numbers or brand names using OCR, and (3) fuse textual and visual evidence. Grid Dynamics show that such an approach can reduce manual catalog browsing for engineers, but their description focuses on a specific industrial client and does not discuss open world marketplaces, multilingual text, or integration with large scale vector search.
Furthermore, OCR is inherently noisy: low light photos, motion blur, or curved surfaces can cause misread characters. Existing industrial solutions often mitigate this with strict image quality requirements or user guidance. A secondhand marketplace cannot enforce such strict conditions, so any OCR driven system must treat OCR output as uncertain evidence rather than ground truth. This motivates the use of hybrid retrieval and re ranking strategies that cross check consistency between image and text features instead of relying on exact matches.
2.4 Multimodal embeddings and text models
Multimodal representation learning offers a unified way to handle images and text within a single embedding space. CLIP, for example, learns joint image–text embeddings from large scale image–caption datasets and enables cross modal retrieval where an image and a text description can be directly compared as vectors.[13] Building on this idea, recent open source models such as BGE VL specialize CLIP like architectures for efficient vision–language embeddings. These models are attractive for secondhand marketplaces because they allow both image queries (user uploads a photo) and text queries (user types a description) to operate over the same vector index.
On the text side, BGE M3 is a state of the art embedding model designed for multi lingual, multi function, multi granularity retrieval. It supports over 100 languages and can perform dense, sparse, and multi vector retrieval within one framework, making it well suited to real world information retrieval systems.[6] For a global secondhand platform where listing titles, descriptions, and labels may appear in various languages, this kind of model is crucial. It can embed OCR results, seller written descriptions, and external web text into a common space, enabling semantic matching even when exact keywords differ.
However, a limitation of generic multimodal models is domain mismatch. They are typically trained on internet scale, natural images and captions, not on close up shots of industrial parts, obscure labels, or worn secondhand goods. As a result, they may struggle with fine grained distinctions between visually similar components or with non standard typography on labels. Prior work in industrial CBIR emphasizes that domain specific data and features are often required for good performance.[14] The proposed project therefore uses BGE VL and BGE M3 as strong baselines but anticipates the need for domain adaptation or re ranking on top, rather than assuming that generic embeddings are sufficient.
2.5 Vector databases and hybrid search
Efficient storage and search over millions of image and text embeddings is a non trivial system component. Milvus is an open source vector database designed for billion scale similarity search with support for approximate nearest neighbor indexes and hybrid queries combining multiple vector fields and structured filters.[15] Recent documentation and tutorials show how Milvus can support multimodal hybrid search, where multiple embeddings (e.g., an image vector and a text vector) are stored in the same collection and queries operate across them with flexible scoring and re ranking.[16] 
E commerce case studies again provide concrete patterns. VOVA’s shop by image system indexes product image embeddings in Milvus and queries them based on user photos, achieving interactive response times despite a catalog of tens of millions of products.[5] Zilliz’s examples of combining multiple computer vision models via ONNX and Milvus demonstrate that multi model embeddings (e.g., from ResNet and VGG) can be stored side by side and fused during retrieval to improve robustness.[17] These patterns are directly applicable to a multi stage pipeline where object detection, OCR, and multimodal embeddings all generate different features for the same item.
For the secondhand part identification problem, hybrid search is particularly important because the system must gracefully handle queries with or without text. When OCR successfully extracts a high confidence part number, the search can bias heavily towards text similarity; when OCR is missing or low quality, the system can fall back to visual similarity and metadata. Milvus’s support for multiple vector fields and hybrid scoring provides the infrastructure to implement this behavior, but prior work stops short of fully exploring feedback driven or adaptive hybrid weighting, leaving room for research on how best to fuse modalities under uncertainty.
2.6 Interactive feedback and continuous learning
Beyond one shot retrieval, interactive systems can improve over time by incorporating user feedback. Classical CBIR literature proposed relevance feedback loops where users label some results as relevant or irrelevant, and the system adjusts the query representation accordingly. Recent work revisits this idea in the context of CLIP based retrieval. Nara et al. propose an interactive retrieval system that uses CLIP embeddings as a base and updates ranking based on binary feedback, without retraining the encoder.[18] They show that this approach can match or surpass state of the art metric learning models in category based image retrieval while adapting to individual users’ preferences.
For secondhand marketplaces, a similar mechanism could be used to refine part identification over time: when the system suggests a candidate part, the seller could confirm or correct it. These confirmations form labeled pairs of (query image, correct part) that can be used to update re ranking models or similarity thresholds. Industrial CBIR work has also highlighted the role of expert feedback for improving retrieval in domains with scarce labeled data.[14] While implementing an online feedback loop may be beyond the scope of the initial prototype, designing the system to log decisions and support future adaptation aligns with these research directions and would help the system remain robust as new types of items appear.
2.7 Summary and identified gaps
The reviewed literature and systems collectively suggest a promising recipe for image based identification of goods and parts:
•	deep visual embeddings for CBIR in both consumer and industrial settings;[10] 
•	object detection to focus on the relevant region;[5] 
•	OCR to exploit text such as model numbers;[12] 
•	multimodal embeddings that unify images and text;[19]
•	vector databases for scalable hybrid search;[16] 
•	and feedback mechanisms for continued improvement.[18] 
However, important gaps remain when I focus specifically on secondhand platforms and small parts:
1.	Open world, fine grained parts: industrial CNN classifiers achieve high accuracy but assume a fixed taxonomy and sufficient training images per class.[10] They do not directly address long tail, one off items common in secondhand marketplaces.
2.	Noisy, user generated images: most systems are evaluated on curated datasets or controlled industrial settings. Marketplace photos often have poor lighting, clutter, or occlusions, which degrade both detection and OCR performance. Existing industry case studies rarely quantify robustness under such conditions.[12] 
3.	Multilingual and heterogeneous text: while BGE M3 and similar models support many languages,[6] published industrial systems usually assume a single language and structured catalogs. Secondhand listings mix local language descriptions, manufacturer English, and informal abbreviations.
4.	Tight integration with listing workflows: eBay and others have begun to use image based tools to auto fill listing metadata for sellers, but publicly documented systems focus on broad products rather than specialized parts and often treat the visual component separately from OCR or hybrid search.[2] 
5.	Closed feedback loops in proprietary systems: interactive improvement and feedback mechanisms are present in the literature but are seldom exposed or documented in commercial visual search APIs, limiting their use as design blueprints for open implementations.[18] 
The proposed project aims to address these gaps by orchestrating an end to end pipeline that combines object detection, OCR, multimodal embedding, hybrid vector search, and LLM based reasoning. Unlike isolated academic prototypes or proprietary black box APIs, the goal is to design and document a system specifically tailored to secondhand items and parts, with an emphasis on (a) handling missing or noisy text, (b) supporting multilingual descriptions, and (c) enabling the system to accumulate knowledge over time by storing resolved queries for future reuse. This orchestration aligns with the “Orchestrating AI Models to Achieve a Goal” template and positions the project as a bridge between industrial CBIR research and real world marketplace workflows.
3. Design
This chapter presents the design of the Smart Image Part Identifier for secondhand platforms. The design follows the CM3020 template “Orchestrating AI models to achieve a goal” by combining OCR, multimodal embeddings, and hybrid retrieval into a single user-facing workflow, rather than relying on one model to solve the entire task. The guiding principle is retrieval-first assistance: the system should produce a small, evidence-backed shortlist that users can verify and edit, rather than a single opaque prediction.

 
Figure 3-1. High-level system architecture diagram (Gradio UI → Orchestrator → OCR/Embeddings → Milvus Hybrid Search → Fusion Ranking → Top-K + Listing Summary)

3.1 Design Goals and Constraints

The system is designed to satisfy the following goals under realistic secondhand photo conditions:

G1. Open-world identification (retrieval, not classification).
Secondhand inventories evolve continuously and include long-tail items. The system therefore treats identification as open-set retrieval, where new items can be supported by indexing (not retraining).

G2. Fine-grained discrimination using multimodal evidence.
Parts can be visually near-identical across variants, while the decisive signal may be a small printed/engraved identifier. The system must combine visual similarity with textual evidence (OCR and optional user text).

G3. Robustness to noisy, user-generated images.
Photos often contain glare, blur, cluttered backgrounds, and partial occlusion. OCR is therefore valuable but uncertain. The design must degrade gracefully when OCR fails.

G4. Listing-ready structured output and transparency.
The output must be useful for listing creation, not just “similar images”. The system returns Top-K candidates together with a structured Listing Summary (e.g., maker, part number, category, short description) and supporting evidence (e.g., OCR text preview and modality scores).

G5. Interactive performance and reproducibility.
The system should remain usable in an interactive workflow. It must measure and report component-level latency and produce reproducible evaluation results via fixed dataset splits and stable retrieval settings.

3.2 System Overview and Component Responsibilities

The architecture is organised into five layers:

(1) UI layer (Gradio).
The UI supports two main flows: indexing (adding catalogue items) and search (querying by user photo). It presents Top-K results, Listing Summary fields, and an accept/edit step.

(2) Orchestration layer (HybridSearchOrchestrator).
The orchestrator coordinates the pipeline: OCR → embedding generation → Milvus search → fusion ranking → result normalisation and logging. It provides a single entry point for both indexing and retrieval, ensuring consistent preprocessing, scoring, and measurement.

(3) Perception layer (OCR).
PaddleOCR [8] extracts text from images. OCR is run on both catalogue images (during indexing) and query images (during search). OCR output is treated as uncertain evidence rather than ground truth.

(4) Representation layer (embeddings).
Image embeddings are computed using a vision-language embedding model (BGE-VL) [7]. Text embeddings are computed using a multilingual embedding model (BGE-M3) [6] for OCR text and optional user-entered hints.

(5) Retrieval and storage layer (Milvus).
Milvus [9] serves as the vector database for approximate nearest neighbour retrieval over image and text embeddings. Hybrid retrieval uses multiple collections / vector fields consistent with Milvus hybrid search patterns [10].

3.3 Data Model and Storage Schema

The design separates (i) model-level catalogue metadata, (ii) image-level assets, (iii) derived OCR text, and (iv) embeddings used for retrieval.

Conceptual entities:
- Model (model_id): maker, part_number, category, description (optional fields).
- ImageAsset (image_id): associated with model_id, capture_type (part / label / mixed), image_path.
- OCRText: raw OCR text and normalised OCR text for retrieval.
- Embeddings: image_vector, text_vector (and optional caption_vector).

Milvus collections:
- image_parts: stores image vectors with metadata (model_id, image_id, maker/category if available).
- text_parts: stores text vectors derived from OCR and structured attributes (model_id, text_type, text_raw/text_normalised).
- model_texts: stores model-level aggregated metadata text and an aggregated vector (updated through confirmation/edits).

Data Model & Milvus Schema
Conceptual entities
Model / Part (model-level entity)
•	model_id (string, unique, required)
•	maker (string, optional)
•	part_number (string, optional)
•	category (string, optional)
•	description (string, optional)
ImageAsset (image-level entity)
•	image_id (string, unique)
•	model_id (string, foreign key)
•	image_path (string)
•	capture_type (enum, optional): part | label | mixed
OCRText (derived from an image)
•	ocr_text_raw (string)
•	ocr_text_normalized (string) — used for retrieval
•	ocr_confidence (float, optional)
Embeddings
•	image_vector (float[d_img]) — image embedding
•	text_vector (float[d_txt]) — text embedding
•	caption_vector (float[d_txt], optional) — text embedding from caption
Milvus collections (implementation schema)
The prototype stores modalities separately: image_parts, text_parts, and model_texts.
Collection: image_parts
•	model_id (string, indexed)
•	image_id (string)
•	image_vector (float vector; metric: cosine)
•	image_path (string)
•	Optional filter fields: maker, part_number, category
Collection: text_parts
•	model_id (string, indexed)
•	text_id (string)
•	text_type (enum): ocr | attrs | caption
•	text_raw (string)
•	text_normalized (string)
•	text_vector (float vector; metric: cosine)
Collection: model_texts (model-level aggregation / knowledge accumulation)
•	model_id (string, primary key)
•	metadata_text (string) — aggregated text for the model (attributes + confirmed evidence)
•	metadata_vector (float vector; metric: cosine)
•	last_updated (timestamp)

This separation supports three critical behaviours:
(i) image-only retrieval when no text evidence exists,
(ii) text-driven retrieval for label-heavy queries,
(iii) model-level knowledge accumulation through updates to model_texts after accept/edit actions.

3.4 Core Workflows

3.4.1 Indexing workflow (catalogue ingestion)

The indexing workflow inserts or updates a catalogue model and its assets:

Step 1 — Input validation and metadata capture.
The system receives one or more images plus model_id and optional metadata (maker, part_number, category, description).

Step 2 — OCR and text normalisation.
OCR is applied to each image; the system stores both raw OCR output and a normalised form (case, spacing, common OCR confusions).

Step 3 — Embedding generation.
The system computes:
- image_vector from BGE-VL [7]
- text_vector from BGE-M3 [6] using normalised OCR text and/or structured attribute text

Step 4 — Upsert into Milvus.
Embeddings and metadata are inserted into image_parts and text_parts. Model-level metadata is aggregated and stored in model_texts.

The indexing workflow is designed to be repeatable: a model can be re-indexed to incorporate additional images, corrected metadata, or improved OCR/normalisation rules.

3.4.2 Search workflow (query photo → Top-K + Listing Summary)

The primary workflow is designed for seller listing assistance:

Input.
A user provides a query image (part photo or nameplate photo). Optional user text can be provided if the user knows partial information (e.g., brand string).

Processing.
Step 1 — OCR on query image.
OCR is executed and shown to the user as evidence. If OCR returns empty/low-confidence output, the system proceeds in image-only mode.

Step 2 — Embedding generation.
- Compute query image embedding v_img with BGE-VL [7].
- Compute query text embedding v_txt with BGE-M3 [6] using (OCR text + optional user text).

Step 3 — Milvus retrieval.
- Search image_parts with v_img for Top-N candidates (cosine similarity).
- Search text_parts and/or model_texts with v_txt for Top-N candidates (cosine similarity).
- Combine candidate sets (union) for fusion ranking.

Step 4 — Fusion ranking.
Each candidate i receives a fused score:

score_i = α · sim_img(i) + β · sim_txt(i) + γ · sim_cap(i)

where sim_img and sim_txt are cosine similarities returned from Milvus searches, and sim_cap is optional (0 when not used). Default evaluation weights are fixed for reproducibility (e.g., α = 0.5, β = 0.4, γ = 0.1).

Step 5 — Output.
The UI displays:
- Top-K candidate list (K = 5) with modality scores,
- a Listing Summary for the highest-ranked candidate (maker, part_number, category, description),
- evidence such as OCR text preview and any matched metadata snippets.
 
Figure 3-2. Prototype (P2) Retrieval UI: user image upload → OCR preview → hybrid retrieval execution → Top-K candidate list (K=5).
 
Figure 3-3. Prototype (P2) Part Card UI: Top-K summary Part Card (maker/part number/category/description) and a human confirmation step (accept/edit).

3.4.3 Accept/edit feedback workflow (human-in-the-loop)

The design explicitly includes a human confirmation step because identifier mistakes can be costly in secondhand transactions.

After viewing results, users can:
- Accept: confirm a candidate as correct
- Edit: correct maker/part_number/category fields
- Reject: indicate the results are not correct

On accept/edit:
- the system logs the action with a trace identifier (query_id, timestamp),
- merges corrected fields into model_texts,
- recomputes the aggregated text vector and upserts model_texts.

This implements a lightweight knowledge accumulation loop consistent with relevance-feedback style improvement in retrieval systems [15], without requiring online retraining of the embedding encoders.

3.5 Information Extraction and Identifier Parsing

Raw OCR text often contains noise (units, partial strings, background text). To make OCR useful for listing workflows, the system includes an information extraction (IE) step that transforms OCR output into structured attributes.

The IE pipeline performs:
(i) normalisation (case, spacing, common OCR confusions),
(ii) pattern-based parsing for identifiers (regex rules for typical part/model/serial formats),
(iii) dictionary matching for known makers/brands when available,
(iv) optional time-boxed fallback parsing when deterministic rules fail.

The output schema is:
{ maker, model_no, part_no, serial_no, other_hints, confidence, evidence_snippets }

These attributes serve two roles:
- retrieval enhancement: attributes are concatenated into the text query embedding,
- verification signal: candidates conflicting with extracted identifiers can be penalised in post-processing (when reliable).

3.6 Evaluation-Oriented Design (Instrumentation and Traceability)

To address earlier feedback that evaluation must be embedded in design, the system is instrumented so that each query produces auditable evidence aligned with evaluation metrics.

Logging.
For each query the orchestrator logs:
- OCR output length and confidence summary
- Top-K candidates with (image/text) similarity scores
- component-level timings (OCR time, embedding time, Milvus search time)
- end-to-end latency percentiles computed over batches

Metric mapping.
- Retrieval quality is evaluated by Accuracy@1 and Accuracy@5, matching the “Top-K shortlist + user verification” design.
- OCR robustness is evaluated using character error rate (CER) when ground-truth identifier text exists.
- Interactive feasibility is evaluated using latency percentiles (p50/p90/p95).

This design ensures evaluation is not an afterthought: the prototype produces the data needed to test the core claims of the project and to guide iteration.

3.7 Design Limitations and Time-Boxed Extensions

Current limitations:
- OCR remains brittle for small or reflective text, motivating careful fallbacks and user transparency.
- Visually similar parts can still cause confusions, especially when identifiers are not visible.
- Latency risk persists until profiling and optimisation are complete.

Time-boxed optional extensions:
- Detection/cropping (e.g., YOLO) to reduce background noise and improve OCR and embeddings [14].
- Cross-encoder reranking to improve fine-grained disambiguation among visually similar candidates.
- External enrichment via controlled web evidence retrieval when internal catalogue search fails (future work).
4. Implementation
This chapter describes the implementation of the prototype to date. The implementation follows the orchestrated pipeline described in Chapter 3 and is organised around a single end-to-end workflow that supports catalogue indexing and query-time retrieval. The focus is on implementability, modularity, and reproducibility: OCR, embedding, and retrieval components are separated so they can be tested independently and replaced if needed.

4.1 Technology Stack and Runtime Environment

The prototype is implemented in Python and integrates the following components:

- OCR: PaddleOCR [8]
- Image embedding: BGE-VL [7]
- Text embedding: BGE-M3 [6]
- Vector database: Milvus [9] with hybrid search patterns [10]
- UI: Gradio (prototype demo interface)
- Containerisation: Docker (Milvus deployment)

The repository is structured as a small set of installable packages and runnable services (model utilities, API layer, and demo UI). Deployment scripts support both development and Docker-based execution.

 
Figure 4-1. Prototype (P2) Indexing UI: catalogue image/metadata input → OCR/text normalisation → embedding generation → Milvus insertion.
 
Figure 4-2. Prototype (P2) Retrieval UI: user image upload → OCR preview → hybrid retrieval execution → Top-K candidate list (K=5).
 
Figure 4-3. Prototype (P2) Part Card UI: Top-K summary Part Card (maker/part number/category/description) and a human confirmation step (accept/edit).

4.2 Code Structure and Core Modules

The implementation centres on a HybridSearchOrchestrator class that coordinates all steps required to index and search. At a high level, the system is organised into:

(1) OCR module.
Provides a unified interface to run OCR on an image and return:
- raw text lines
- confidence statistics
- optional language hints
The module also implements normalisation used by both indexing and search.

(2) Embedding modules.
- ImageEmbedder: computes v_img = f_img(image) via BGE-VL [7].
- TextEmbedder: computes v_txt = f_txt(text) via BGE-M3 [6].
Embedding calls are isolated so that performance profiling can attribute latency to each component.

(3) Retrieval module (Milvus client).
Implements:
- upsert_image_vectors into image_parts
- upsert_text_vectors into text_parts / model_texts
- search_image_vectors and search_text_vectors
Search results are returned with similarity scores and metadata needed for fusion and UI display.

(4) Fusion and ranking module.
Combines image and text search results into a unified candidate set and computes fused ranking scores:
score_i = α·sim_img(i) + β·sim_txt(i) + γ·sim_cap(i)
Weights are fixed during evaluation runs for reproducibility.

(5) UI module (Gradio).
Implements two tabs:
- Indexing tab: add/update model assets and metadata
- Search tab: upload query image, preview OCR, view Top-K candidates, accept/edit Listing Summary

4.3 Indexing Pipeline Implementation

The indexing pipeline is responsible for transforming catalogue assets into searchable representations.

Inputs.
- model_id (required)
- one or more images (required)
- optional metadata fields: maker, part_number, category, description

Processing.
Step 1 — OCR extraction and normalisation.
Each image is passed to PaddleOCR [8]. The output is stored as:
- ocr_text_raw: raw concatenation of detected lines
- ocr_text_normalised: normalised form for retrieval and parsing

Step 2 — Attribute construction.
If structured fields are provided (maker/part_number/category), the system builds a metadata_text string. This text can be embedded and stored in model_texts so that confirmed model-level descriptions become retrievable.

Step 3 — Embedding computation.
- Compute image_vector via BGE-VL [7].
- Compute text_vector via BGE-M3 [6] using ocr_text_normalised and metadata_text.

Step 4 — Milvus upsert.
- Upsert image_vector into image_parts with metadata (model_id, image_id, paths, optional filters).
- Upsert text_vector into text_parts with fields (model_id, text_type, raw/normalised text).
- Upsert aggregated metadata into model_texts (model_id primary key), enabling model-level retrieval.

Outputs.
The UI reports completion and shows a short indexing summary (number of images processed, OCR length, and upsert status). Indexing operations are designed to be repeatable so that adding more images or corrected metadata strengthens the index over time.

4.4 Search Pipeline Implementation

The search pipeline implements query-time orchestration for listing assistance.

Inputs.
- query_image (optional but primary)
- query_text (optional)
- top_k (default 5)

Step 1 — OCR on the query image.
OCR is run first so that the UI can show the extracted text as evidence. If OCR returns empty or extremely low-confidence output, a fallback path disables text retrieval and runs image-only retrieval.

Step 2 — Embedding computation.
- Compute query image embedding v_img from the uploaded photo using BGE-VL [7].
- Build a text query string by concatenating:
  - normalised OCR text (if available)
  - user-entered text hint (if provided)
  - extracted structured hints from the IE step
- Compute v_txt via BGE-M3 [6].

Step 3 — Hybrid retrieval in Milvus.
- Image search: query image_parts for Top-N neighbours.
- Text search: query text_parts and/or model_texts for Top-N neighbours.
The system uses cosine similarity for both modalities for consistent fusion scoring.

Step 4 — Candidate union and fusion ranking.
Candidates are merged by model_id. For each candidate, the system stores:
- image similarity score
- text similarity score
- fused score
- key metadata fields for the Listing Summary

Step 5 — Listing Summary generation.
For the highest ranked candidate (and optionally Top-K), the system composes a Listing Summary containing:
- maker
- part_number (and/or model number)
- category
- short description
- evidence (OCR preview and any matched metadata snippets)
This output is designed to map directly to listing form fields.

4.5 Human-in-the-Loop Feedback and Knowledge Accumulation

The UI includes an accept/edit step:

- Accept: the selected model_id is logged as correct for the current query.
- Edit: the user edits fields (e.g., maker/part_number/category) before confirming.

On accept/edit:
- a feedback record is stored (query_id, selected model_id, timestamp, action),
- model_texts is updated by merging corrected fields into metadata_text,
- the aggregated metadata vector is recomputed and upserted.

This provides a practical learning signal for future improvements and supports incremental catalogue quality improvement without retraining encoders.

4.6 Evaluation Tooling and Profiling Hooks

To support reproducible evaluation (Chapter 5), the implementation includes evaluation scripts and instrumentation:

- Retrieval evaluation loop:
  iterate over a fixed query list, run search(top_k=5), and record whether ground-truth model_id appears in Top-K.

- Timing instrumentation:
  record OCR time, embedding time, Milvus search time, and end-to-end time per query. Percentile latencies (p50/p90/p95) are computed over batches.

- Result export:
  evaluation outputs are saved to CSV/JSON so they can be re-run and compared across conditions (image-only vs hybrid, weight ablations, OCR enabled/disabled).

4.7 Implementation Status Summary

Implemented in the prototype:
- End-to-end workflow: indexing + hybrid retrieval + UI output
- OCR integration and OCR evidence display
- Image/text embedding and Milvus storage
- Fusion ranking and Top-K candidate display
- Accept/edit logging and model_texts update foundation
- Image-only baseline evaluation script and dataset split tooling

Not yet implemented / time-boxed:
- Object detection/cropping to isolate parts and nameplates [14]
- Cross-encoder reranking for fine-grained disambiguation
- External enrichment for items missing from the internal catalogue
(a)  (b)  (c) 
(d)  (e)  (f) 
Figure 5: Examples of OCR failure cases: (a) mixed vertical and horizontal text, (b) fine print and small serial numbers, (c) blurred labels, (d) stylized manufacturer logos, (e) distant nameplates, (f) engraved text.
5. Evaluation

This chapter describes the evaluation strategy and the initial evaluation results obtained so far. The evaluation is designed to test whether the prototype supports secondhand listing workflows by (i) retrieving the correct part within a small shortlist, (ii) extracting identifier text with acceptable quality when present, and (iii) operating with interactive performance. Because the project is a draft-stage report, some evaluation components are reported as completed (baseline retrieval), while others are reported as in progress with fixed protocols and reporting templates (hybrid ablation, OCR CER benchmarking, latency profiling, and usability study).






5.1 Evaluation Strategy
Item	Definition (keep concise)
Goal	Validate whether the prototype improves part identification for listing creation under realistic photo conditions.
Inputs	User photo of a part (label/plate or body image). OCR is applied only when identifier text is visible/legible enough to attempt.
Compared 
conditions	C0 Image-only baseline (visual retrieval) vs C2 Hybrid (visual + OCR/text evidence). (Optional: C1 OCR-only when image retrieval is disabled, used only for ablation if needed.)
Primary metric	Accuracy@1, Accuracy@5 (Top-K contains the correct ground-truth item). Report for C0 and C2, plus delta (C2 − C0).
Secondary metrics	OCR CER on identifier text (when GT text is available) + Latency p50/p90 for end-to-end and component-level (OCR, retrieval).
Success criteria	MVP: Accuracy@5 ≥ 0.80 on main split and C2 improves over C0. Stretch: Accuracy@5 ≥ 0.85 and/or stronger gains on “hard cases” (blur/glare/occlusion).
Table 5-1 : One-page Evaluation Overview.

Component	Rule
Split A (Main)	Random sample of N=1000 query images with known ground truth.
Split B (Diversity)	Stratified sample of N=500 to cover key categories/conditions.
Ground truth	For each query: a single correct target item ID (and identifier text GT when available).
Reporting	Always report A and B separately, then overall summary. Include per-condition breakdown for “hard cases” (at least 2–3 groups).
Reproducibility	Fix random seed and store split IDs; reruns must use identical split lists and identical K (K=5).
Table 5-2 here: Dataset Splits & Reporting Rules.

The evaluation follows a retrieval-first philosophy aligned with the system design. Since the output is a Top-K shortlist that users can confirm, the primary success criterion is whether the correct model appears within the shortlist rather than whether the system always ranks it first.

5.1.1 Evaluation objectives

E1. Retrieval effectiveness.
Measure whether the correct ground-truth model appears within the Top-K results. Primary metrics: Accuracy@1 and Accuracy@5.

E2. OCR robustness for identifiers.
Measure OCR quality for identifier text when ground truth is available. Primary metric: character error rate (CER). Secondary: word error rate (WER) where appropriate.

E3. Workflow latency.
Measure whether the pipeline is usable in an interactive setting. Report p50/p90/p95 latency for OCR, retrieval, and end-to-end time.

E4. User-centred usefulness and trust.
Measure whether the system reduces listing effort and supports user confidence. Use task-based measures (time, edits, success rate) and usability questionnaires (SUS and/or SEQ).

5.1.2 Compared conditions (ablation)

The evaluation compares the following conditions:

C0 (Image-only baseline):
OCR disabled or ignored; α=1.0, β=0.0, γ=0.0.

C1 (Text-only):
Only valid when OCR/text exists; image similarity ignored.

C2 (Hybrid fusion):
Image + OCR/text combined with fixed weights (e.g., α=0.5, β=0.4, γ=0.1).
Condition	Random1000 Acc@5	Category500 Acc@5	Notes
C0 (image-only)	0.791	0.812	baseline
C1 (text-only)	TBD	TBD	only valid for text-present queries
C2 (hybrid)	TBD	TBD	α=0.5, β=0.4, γ=0.1

Table 5-3. Hybrid Ablation Reporting Template.

5.1.3 Justification of metrics

Accuracy@5 is the most important metric because the system is designed as an assistive tool. If the correct part appears within the Top‑5, the user can confirm it with minimal effort, which matches realistic secondhand listing workflows. Accuracy@1 is reported as a stricter measure of ranking quality but is not the only definition of success in a human-in-the-loop shortlist design.

CER is used for OCR because part identification can fail due to single-character errors (e.g., confusing “8” with “B” in a part number). A character-level metric is therefore more sensitive and more relevant than a coarse document-level judgement.

Latency percentiles are reported because user experience depends on worst-case delays, not only average time. Percentile reporting (p90/p95) helps identify whether some queries become unusably slow due to OCR or embedding bottlenecks.

5.2 Datasets and Experimental Setup

The evaluation uses two predefined splits to support both realism and diversity:

Split A (Main): Random1000 models.
One query image per model is held out; remaining images are indexed. This evaluates overall retrieval behaviour across a broad catalogue sample.

Split B (Diversity): Category-sampled 500 models.
A stratified sample across 10 categories (50 models each) is used to ensure coverage of diverse part types and to support category-wise analysis.
Dataset	Models	Query images	Index images	Categories	Notes
Random 1000 models	1,000	1,000	13,456	100	1 query per model; remaining images indexed
Category-sampled models	500	500	7,141	10	50 models × 10 categories; 1 query per model
Table 5-4. Dataset Split Summary.

Reproducibility.
Split IDs are fixed using a stored list and random seed, and all runs use the same Top-K (K=5) and identical Milvus index settings.

5.3 Retrieval Evaluation Results (Completed)

5.3.1 Image-only baseline (C0)

A baseline experiment was conducted using image-only retrieval to validate feasibility of retrieval-first identification.
Dataset	Accuracy@1	Accuracy@5
Random 1000 models	0.287	0.791
Category-sampled 500 models	0.306	0.812
Table 5-5 : Image-only baseline results.
 
Figure 5-1: Baseline retrieval performance chart.

Summary of results:
- Random1000: Accuracy@1 = 0.287, Accuracy@5 = 0.791
- Category-sampled 500: Accuracy@1 = 0.306, Accuracy@5 = 0.812

Interpretation (critical).
The results support feasibility of retrieval-first identification: the correct item appears in the Top‑5 for a large fraction of queries, especially in the category-balanced split. However, Random1000 is slightly below the MVP target (Accuracy@5 ≥ 0.80), indicating that image-only retrieval is not reliably sufficient for the full domain. This justifies the hybrid design: OCR/text evidence and/or model-level aggregated text is required to push performance above the target, particularly for look-alike parts and label-driven queries.

5.3.2 Error patterns observed in retrieval

Initial inspection suggests recurring failure cases:
- visually similar variants: near-identical parts differ only by model number not visible in the query photo
- background clutter: embeddings capture scene noise when the part occupies only a small region
- catalogue coverage: if the correct model is absent, the system can only return “similar” candidates

These patterns align with the design decision to (i) add text evidence when available, (ii) time-box detection/cropping, and (iii) support user confirmation rather than autonomous prediction.

5.4 OCR Robustness (Initial Evidence + Benchmark Plan)

5.4.1 Observed OCR failure modes (qualitative)

The project analysed representative OCR failures drawn from real catalogue images. Common cases include:
- mixed vertical/horizontal text
- very small serial numbers or fine print
- blurred labels
- stylised logos and fonts
- distant nameplates
- engraved text
(a) (b (c) 
(d) (e) (f) 
Figure 5-2. OCR failure patterns observed in real catalogue images (a) mixed vertical and horizontal text, (b) fine print and small serial numbers, (c) blurred labels, (d) stylized manufacturer logos, (e) distant nameplates, (f) engraved text.

Interpretation.
These examples show that OCR is valuable but unreliable in uncontrolled secondhand imagery. This supports the design principle of treating OCR as uncertain evidence and providing a robust image-only fallback path.

5.4.2 CER benchmarking (in progress)

A controlled OCR benchmark set (n=50–100) is being prepared with ground-truth identifier strings. CER (and optionally WER) will be computed on this subset. The benchmark will report:
- overall CER
- CER by scenario group (blur/glare/occlusion/low-resolution)
- representative failure examples and normalisation effects

5.5 Latency Evaluation (in progress)

To evaluate interactive feasibility, the orchestrator logs component-level timings per query:
- OCR time
- image embedding time
- text embedding time
- Milvus search time
- end-to-end time

The evaluation will report percentile latencies (p50/p90/p95) for OCR, retrieval, and end-to-end runtime. The design target is interactive performance suitable for listing workflows (e.g., low single-digit seconds), but the report prioritises transparent measurement and bottleneck identification over optimistic claims.

5.6 Usability Evaluation (planned with fixed protocol)

To validate user usefulness and trust, a small usability study is planned (n=5–8). Participants will perform tasks such as:
T1: upload a photo → inspect Top‑5 → confirm correct candidate
T2: accept/edit listing fields (maker/part_number/category)

Metrics:
- task completion time
- number of edited fields
- task success rate
- SEQ (per task) and SUS (overall), plus qualitative feedback on trust and transparency
Item	Plan
Participants	n=5–8 (internal colleagues recruited via Slack)
Tasks	(T1) Upload part photo → inspect Top K → confirm correct candidate. (T2) Use accept/edit to finalise listing fields (maker/part number/category).
Quant metrics	Task completion time (seconds), # of edited fields, task success rate (%)
Questionnaires	SEQ (per task), SUS (or UMUX Lite) + 2–3 open-ended questions (“why”, “trust”, “suggestions”)
Success criteria	SUS ≥ 70 or SEQ mean ≥ 5/7; ≥10% time reduction or fewer edits vs manual baseline
Table 5-6 : Usability test protocol.

The usability evaluation is explicitly aligned with the system’s human-in-the-loop design: the goal is not “perfect autonomous correctness,” but reduced user effort and increased confidence through evidence-backed suggestions.

5.7 Hybrid Retrieval Ablation (in progress)

To verify that OCR/text adds discriminative value (especially for label-heavy queries), the hybrid ablation will be reported on both splits:

- C0 image-only
- C1 text-only (when valid)
- C2 hybrid fusion

Results will be reported as Accuracy@1/@5 with deltas (C2 − C0), and broken down by “hard case” groups (blur/glare/occlusion) where OCR is expected to matter most.

5.8 Objective Status and Critical Assessment
Objective	Metric	Current status	Evidence
O1 MVP workflow	end-to-end demo success	Green	UI screenshots + demo checklist
O2 Retrieval	Acc@5 ≥ 0.80	Amber	baseline close / split-dependent
O3 OCR quality	CER ≤ 8%	Amber	OCR failure patterns identified; CER pending
O4 Latency	p95 targets	Amber	timing hooks in progress
O5 Efficiency	≥10% time reduction or fewer edits	Amber	protocol ready; study pending

Table 5-7 : Objective status dashboard.

Current evidence supports:
- MVP workflow feasibility: end-to-end pipeline runs and produces Top‑5 + Listing Summary output.
- retrieval-first viability: image-only baseline approaches the MVP target and exceeds it in category-balanced scenarios.
- need for hybrid evidence: Random1000 baseline underperforms slightly, making hybrid fusion and/or post-verification essential.

Remaining risks:
- OCR brittleness: OCR may mislead retrieval if treated as a hard signal, requiring careful normalisation and transparency.
- latency: until p95 profiling is complete, interactive feasibility remains uncertain.
- look-alike parts: fine-grained disambiguation may require reranking or better region focus.

Planned improvements are therefore prioritised in the following order:
(1) complete hybrid ablation and OCR CER benchmarking to quantify gains and limits,
(2) latency profiling and optimisation (batching/caching),
(3) time-boxed detection/cropping experiments,
(4) optional reranking if measurable gains justify complexity.

Github repository
https://github.com/hshlalla/Smart_vision

References
[1] Grid Dynamics. “Identifying screws, a practical case study for visual search” Grid Dynamics. 2019.
[2] eBay Inc. “Find It On eBay: Using Pictures Instead of Words” innovation.ebayinc.com 2017.
[3] The First Media. ““비용은 줄이고 신뢰도는 높이고!”…중고거래 시장에서도 AI가 대세” thefirstmedia.net, 2025.
[4] MDPI. “Recognition of Industrial Spare Parts Using an Optimized Convolutional Neural Network Model.” mdpi.com, 2024.
[5] Zilliz (Milvus). “Building a Search by Image Shopping Experience with VOVA and Milvus.” zilliz.com. 2021
[6] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. “M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation” arxiv.org, 2024.
[7] BGE. “BGE-VL.” bge-model.com, 2024.
[8] Google Lens. “What is Google Lens?” lens.google.
[9] advertisement. “E-commerce Is Coming to Snapchat with Amazon Product Search Tool” advertisemint.com .
[10] Qiaoyang Li, Guiming Chen. “Recognition of industrial machine parts based on transfer learning with convolutional neural network” journals.plos.org, 2021.
[11] Myung Seok Shim, Christopher Thiele, Jeremy Vila, Nishank Saxena, Detlef Hohl. cambridge. “Recognition of Industrial Spare Parts Using an Optimized Convolutional Neural Network Model.” cambridge.org, 2023.
[12] Grid Dynamics. “Visual search: How to find manufacturing parts in a cinch” Grid Dynamics. 2019.
[13] Zilliz. “Exploring Multimodal Embeddings with FiftyOne and Milvus.” medium.com 2024.
[14] Myung Seok Shim , Christopher Thiele, Jeremy Vila, Nishank Saxena, Detlef Hohl. cambridge.org. “Content-based image retrieval for industrial material images with deep learning and encoded physical properties” cambridge.org 2023.
[15] milvus.io. “What is Milvus.” milvus.io.
[16] milvus.io. “Multi-Vector Hybrid Search.” milvus.io.
[17] Zilliz (Milvus). “Combine AI Models for Image Search using ONNX and Milvus.” zilliz.com. 2021
[18] arxiv. “Revisiting Relevance Feedback for CLIP-based Interactive Image Retrieval” arxiv.org, 2024.
[19] Zilliz (Milvus). “How does multimodal image-text search work?.” zilliz.com. 2021
[20] paddle “PaddleOCR 3.0 Technical Report.” arxiv..org 2025

