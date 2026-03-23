# 2. Literature Review

This chapter critically reviews prior research and industry practice across several intersecting domains: visual product retrieval, the shift from closed-set object detection to open-world hybrid search, the limitations of Optical Character Recognition (OCR) in industrial settings, the emergence of Vision-Language Models (VLMs), and the necessity of human-in-the-loop (HITL) system design. Together, these strands of literature provide the theoretical and engineering basis for the architectural choices made in this project.

## 2.1 The Evolution of Visual Product Retrieval and Object Detection

The shift from text-based querying to image-based retrieval has fundamentally changed how users search for products in both e-commerce and industrial settings. Historically, product retrieval depended heavily on keyword indexing and manually curated metadata. As catalogues expanded, however, the vocabulary-mismatch problem became increasingly severe: non-expert sellers often lacked the exact technical terminology needed to describe or search for industrial components.

To bridge this semantic gap, Content-Based Image Retrieval (CBIR) systems were extensively researched and commercialised. Early CBIR systems relied on handcrafted visual descriptors such as SIFT and HOG. These methods could support duplicate-like matching under controlled conditions, but they lacked higher-level semantic understanding. With the rise of deep learning, Convolutional Neural Networks such as ResNet made it possible to extract richer semantic embeddings directly from pixels. Industry deployments, such as eBay’s visual search rollout, showed that deep learning could reduce listing and search friction at marketplace scale [6], [12].

In industrial contexts, visual identification has also often been framed as an object-detection or image-classification problem. A large body of work uses models such as YOLO for part detection, defect localisation, or controlled-factory recognition tasks [13]. More recently, self-supervised Vision Transformers such as CLIP and DINOv2 have become central because they provide strong generic visual representations without requiring exhaustive manual labelling [4], [5].

**Critical Evaluation:** These approaches remain limited in the secondhand industrial domain. First, closed-set object detection assumes a predefined label space. Rare, discontinued, or previously unseen parts, which are common in secondhand marketplaces, do not fit this assumption without retraining. Second, industrial and electronic parts are highly fine-grained and visually homogeneous. Two variants may share an almost identical macro-appearance while differing only in a small printed identifier. Generic visual embeddings tend to cluster such items together and therefore fail to provide the micro-level distinction needed for accurate listing support. Industry discussions of manufacturing-part visual search point to the same gap between generic similarity and true part identification [10]. For these reasons, purely global visual retrieval or closed-set classification is insufficient on its own.

## 2.2 From Closed-Set Detection to Open-World Hybrid Search

To move beyond the limitations of closed-set classification, modern retrieval systems increasingly rely on dense vector embedding spaces and Approximate Nearest Neighbour (ANN) search. Instead of assigning an image to one fixed class, the system converts it into a high-dimensional vector and searches for nearby items in embedding space. Foundational systems such as FAISS made large-scale similarity search practical, and later vector databases such as Milvus turned this approach into a scalable data-management layer [14], [15].

This shift is especially important in open-world domains because the indexed inventory can grow continuously without retraining the embedding model. However, purely visual vector search still fails when the decisive differences between parts are textual rather than geometric.

To address this, recent retrieval systems have moved toward hybrid vector search. Hybrid search allows dense visual similarity to be combined with lexical, sparse, or scalar evidence during retrieval and ranking. In practice, this means that a candidate can be retrieved because it looks similar, but then promoted because its maker, part number, or drafted metadata aligns with the query. This is also where strong text-embedding models such as BGE-M3 become important: they offer multilingual, multi-granularity text representations that are well suited to short alphanumeric identifiers typical of industrial part numbers [2], [8].

**Critical Evaluation:** Pure visual vector search can often recover the correct neighbourhood of parts, but it cannot reliably identify the exact part variant when discriminative evidence is textual. Hybrid search is therefore not a minor enhancement but a structural requirement for industrial-part retrieval. By combining a strong visual encoder with dense text representations and vector-database support for multimodal retrieval, a system can navigate a domain that is visually homogeneous but textually distinctive.

## 2.3 The Limitations of OCR in Industrial Imagery

Recognising that purely visual retrieval cannot reliably recover fine-grained textual identifiers such as exact part numbers or manufacturer codes, many systems have historically combined visual models with Optical Character Recognition (OCR). A practical OCR pipeline such as PaddleOCR typically consists of two sequential stages: a text-detection stage that locates candidate text regions, and a text-recognition stage that decodes those cropped regions into strings [9], [16], [17], [18].

This pipelined architecture performs well on structured documents, clean signage, and front-facing labels where text is the main visual subject. It is therefore understandable that earlier industrial retrieval systems often assumed that extracting more text would improve search accuracy by recovering exact alphanumeric identifiers.

**Critical Evaluation:** In open-world, user-generated industrial imagery, this assumption breaks down. Real secondhand-marketplace photographs are characterised by glare from metallic surfaces, worn labels, cluttered backgrounds, low contrast, skewed viewpoints, and mixed text orientations. These conditions weaken both text detection and text recognition.

The deeper limitation is architectural rather than merely statistical. OCR is semantically blind: it is designed to extract visible strings, not to determine which strings are identity-bearing. Industrial components often contain large amounts of irrelevant text, including voltage ratings, frequency values, safety warnings, country-of-origin markings, and packaging remnants. When this raw text is embedded directly and used for retrieval, it can distort the ranking signal. The vector space becomes shaped by frequent but non-discriminative specifications rather than by the unique part identifier.

This semantic-noise problem explains why OCR-heavy pipelines often retrieve unrelated parts that merely share a common voltage, resistance, or warning string. In addition, running a full text-detection and text-recognition pipeline on high-resolution images introduces substantial latency, which is difficult to justify in an interactive listing workflow. For this reason, both the literature and the findings of this project suggest that OCR is better treated as selective supporting evidence than as a default primary retrieval path.

## 2.4 Multimodal Embeddings and Vision-Language Models (VLMs)

To address the sequential bottlenecks and semantic blindness of traditional OCR pipelines, recent research has moved towards unified Vision-Language Models (VLMs). Models such as LLaVA and the Qwen-VL family combine a visual encoder with a language-model backbone and use cross-modal alignment to reason over images and text in a shared semantic space [1], [19]. Instead of processing vision and text in isolated algorithmic silos, these models can interpret spatial layout, textual prominence, logo context, and object structure together.

This broader multimodal trend also overlaps with developments in embodied and physical-world AI, where systems are expected to interpret unstructured physical environments rather than only digital text or curated images [20], [21]. Although the present project is not a robotics system, this trend is still relevant because industrial-part identification likewise requires AI to understand objects in context, including label hierarchy, visual wear, and surrounding clutter.

In parallel, textual metadata retrieval has been strengthened by embedding models such as BGE-M3, which provide multilingual and multi-granularity representations well suited to short, dense alphanumeric identifiers [2].

**Critical Evaluation:** The key advantage of VLMs in this domain is not simply better caption generation, but more context-aware filtering of identity-relevant evidence. A VLM can distinguish a prominent manufacturer mark from a generic warning label and can prioritise text that is structurally central to product identity. This reduces dependence on standalone OCR as the primary signal and makes the system better suited to open-world, fine-grained secondhand-part retrieval. In this project, that literature directly supports the design shift from OCR-heavy extraction toward a vision-language-dominant pipeline in which OCR is demoted to an optional verification aid.

## 2.5 Critical Evaluation: The Limits of One-Shot VLM Identification

Given the strong reasoning and extraction capabilities of contemporary Vision-Language Models, an appealing engineering hypothesis is that a system might rely on a single end-to-end VLM inference. In such an architecture, the image would be passed to a powerful model with a prompt such as “identify this industrial part and output its exact metadata,” seemingly removing the need for a vector database or a hybrid retrieval layer.

**Critical Evaluation:** Current literature on LLM and VLM reliability suggests that this is unsafe for high-stakes, fine-grained domains. Three structural weaknesses are especially important.

1. **Hallucinated identifiers**  
   Generative models produce probable tokens rather than verified facts. Reliability studies show that these systems can output fluent but incorrect content, including highly plausible identifiers [22]. In industrial-part listing, a one-character error in a part number is enough to invalidate the listing.

2. **Static parametric knowledge in an open-world domain**  
   A VLM’s internal knowledge is bounded by its training distribution and cut-off. It cannot be assumed to know the live inventory of a specific seller, nor to robustly identify rare or discontinued parts that were weakly represented during training.

3. **Weakly grounded outputs**  
   A one-shot generated answer does not automatically provide verifiable external evidence. Even when the output appears correct, the user is left without a grounded basis for trust.

These limitations motivate a retrieval-first alternative. Retrieval-Augmented Generation (RAG) showed the broader value of separating parametric reasoning from external knowledge access [23]. In the present project, a closely related idea is applied multimodally: the model is used to extract useful features and metadata cues, but the final candidate suggestions are grounded through retrieval from an external indexed database. This architecture reduces the risk of hallucinated identifiers and ensures that suggested items correspond to retrievable, inspectable evidence rather than to unconstrained generation.

## 2.6 Human-in-the-Loop (HITL) and Interactive Relevance Feedback

Even when retrieval quality is strong, fully autonomous identification remains risky in commercial settings where incorrect listings may lead directly to financial loss, failed searches, return friction, and loss of buyer trust. Literature on human-centred AI consistently warns against over-automation in domains where precision, accountability, and user confidence matter [24], [25].

Human-in-the-loop design addresses this issue by positioning AI as a decision-support mechanism rather than a decision-making authority. Relevance-feedback ideas from information retrieval are also relevant here: the system presents plausible candidates, the user inspects them, and the interaction refines the final outcome. Dong et al. (2021) showed that explainable, interactive image-retrieval systems can improve user confidence and task performance by exposing evidence rather than returning a single opaque result [3].

**Critical Evaluation:** For industrial-part identification, this design logic is especially compelling. A Top-K shortlist respects the fact that the human seller still possesses the strongest grounding source: direct access to the physical item. The user can visually compare candidates, verify the proposed evidence, and correct metadata where necessary. This means the system operates as a cognitive and workflow aid rather than an unchecked classifier. The HITL loop also has a secondary systems benefit: confirmed and corrected outputs can improve the quality of the indexed data over time.

The key literature strands and the stance adopted in this project are summarised in **Table 2**.

**Table 2. Comparison of retrieval approaches and their implications for industrial-part identification**

| Approach | Strengths | Limitations in industrial context | Project's stance |
| --- | --- | --- | --- |
| Traditional Visual Search (e.g., CLIP, DINOv2) | Excellent for global shapes and colours; fast. | Fails to distinguish fine-grained details (e.g., identical parts with different serial numbers). | Insufficient on its own. |
| Pipeline: Vision + OCR | Extracts text to supplement visual search. | Highly sensitive to noise, glare, and complex backgrounds; extracts irrelevant specs (e.g., "12V"). | Use OCR only for secondary verification, not primary search. |
| Vision-Language Models (e.g., Qwen3-VL) | Context-aware text understanding; interprets layouts and logos without explicit OCR. | Heavier computational load; potential for occasional hallucination. | Primary retrieval engine; balances text and visual context. |
| Human-in-the-Loop (HITL) | High user trust; allows manual verification and correction. | Requires user interaction; not fully autonomous. | Core design philosophy; outputs a Top-K shortlist with evidence. |

*Table note:* This table is an analytical summary rather than a benchmark table. It consolidates the main literature themes reviewed in this chapter and shows how each one informed the architectural stance adopted in the project.

## 2.7 Summary of Literature and Architectural Justification

The literature reviewed in this chapter shows that while visual retrieval and OCR are both well established, their naive combination is fundamentally suboptimal for the noisy, fine-grained, and open-world domain of industrial components. Generic visual embeddings fail on micro-level distinctions, closed-set detectors fail on dynamic inventories, and traditional OCR pipelines generate substantial semantic noise and latency. In addition, relying entirely on generative VLM inference introduces unacceptable risks of hallucination and weakly grounded output.

Taken together, these findings point toward a specific architectural direction. An effective industrial-part identification system should use Vision-Language Models for context-aware feature extraction, combine visual and textual evidence through hybrid retrieval, ground its candidate suggestions in an external database, and maintain a human-in-the-loop interface for verification and correction. This directly motivates the retrieval-first, selective-OCR, and evidence-backed architecture designed and evaluated in the remainder of this report.
