# 2. Literature Review

This chapter critically reviews prior research and industry practice across four key areas: visual product retrieval, the limitations of OCR in industrial settings, the emergence of Vision-Language Models (VLMs), and human-in-the-loop (HITL) system design. Together, these strands of literature provide the technical basis for the architectural choices made in this project.

## 2.1 Visual Product Retrieval in E-commerce and Industry

Visual search has become a major component of modern e-commerce, allowing users to retrieve products with images rather than text alone. Large platforms such as eBay have shown that image-based retrieval can reduce search friction for consumer goods, particularly when users do not know the exact product name or keywords [6], [12]. Recent research has further demonstrated the effectiveness of large-scale visual representations such as CLIP and DINOv2 for capturing global semantic similarity across images [4], [5].

**Critical Evaluation:** While these systems perform well for categories with distinctive overall appearance, such as clothing, furniture, or household goods, they are less suitable for industrial and electronic parts. In these domains, multiple variants may share almost identical global shape and differ only in a small printed identifier, nameplate, or logo. Generic visual embeddings tend to cluster such items by macro-appearance while overlooking the micro-level textual details required for exact identification. Industry-oriented discussions of manufacturing-part visual search also point to this gap between generic similarity and true part identification [10]. This makes purely global image similarity insufficient for secondhand industrial-part listing support.

## 2.2 The Limitations of OCR in Industrial Imagery

To compensate for the limits of purely visual retrieval, many systems incorporate Optical Character Recognition (OCR) to extract text from product images. Traditional OCR pipelines perform strongly on documents or clean front-facing labels, where the text is large, well aligned, and high contrast.

**Critical Evaluation:** Industrial-part images in secondhand marketplaces rarely satisfy these conditions. Real user images often contain glare, blur, wear, cluttered backgrounds, and partial occlusion, all of which degrade OCR quality. More importantly, raw OCR often captures excessive irrelevant text, such as electrical ratings, safety notices, packaging remnants, or background clutter. When all extracted text is treated equally in retrieval, the truly useful identifiers, such as maker names and part numbers, can be diluted by noisy strings. As a result, more OCR output does not necessarily lead to better retrieval. This directly supports the design decision in this project to treat OCR as a secondary verification signal rather than the primary retrieval engine.

## 2.3 Multimodal Embeddings and Vision-Language Models (VLMs)

Recent work has shifted from loosely coupled pipelines of vision plus standalone OCR towards unified Vision-Language Models that jointly interpret visual structure and embedded text. Models in the Qwen-VL family, including Qwen3-VL, have shown strong capability in reading, localising, and interpreting text within complex visual scenes while preserving contextual understanding [1]. In parallel, textual metadata retrieval has benefited from strong embedding models such as BGE-M3, which provide multilingual and multi-granularity representations suitable for matching short, dense, alphanumeric identifiers [2].

**Critical Evaluation:** The main advantage of VLMs in this domain is not simply better captioning, but more context-aware interpretation. Instead of blindly extracting raw strings, a VLM can distinguish a manufacturer's logo from a generic warning label or recognise which visible text is structurally relevant to product identity. This reduces dependence on standalone OCR as the primary signal and is more appropriate for open-world, fine-grained secondhand parts. When combined with a strong text embedding model such as BGE-M3 for metadata and identifier search, the system becomes better suited to the mixed visual-textual nature of industrial-part retrieval.

## 2.4 Human-in-the-Loop (HITL) and User Trust

Even when retrieval quality is strong, fully autonomous identification remains risky in commercial settings where incorrect listings may lead to pricing errors, failed searches, and loss of buyer trust. Research on human-in-the-loop systems consistently shows that user trust depends not only on raw model accuracy, but also on transparency, inspectability, and user control. Dong et al. (2021) showed that explainable and interactive image-retrieval systems can improve confidence and task performance by exposing evidence rather than returning a single opaque result [3].

**Critical Evaluation:** This literature strongly supports the core design philosophy of the present project. Secondhand industrial-part identification should not be framed as a closed-set black-box prediction task. Instead, it is better treated as a decision-support workflow in which the system narrows the search space to a manageable shortlist and provides the evidence needed for user verification. In secondhand listing workflows, reducing uncertainty to a plausible Top-K set is often more realistic and more useful than insisting on a single autonomous prediction. This directly justifies the system's retrieval-first, evidence-backed, human-editable design.

The key literature strands and the stance adopted in this project are summarised in **Table 2**.

**Table 2. Comparison of retrieval approaches and their implications for industrial-part identification**

| Approach | Strengths | Limitations in industrial context | Project's stance |
| --- | --- | --- | --- |
| Traditional Visual Search (e.g., CLIP, DINOv2) | Excellent for global shapes and colours; fast. | Fails to distinguish fine-grained details (e.g., identical parts with different serial numbers). | Insufficient on its own. |
| Pipeline: Vision + OCR | Extracts text to supplement visual search. | Highly sensitive to noise, glare, and complex backgrounds; extracts irrelevant specs (e.g., "12V"). | Use OCR only for secondary verification, not primary search. |
| Vision-Language Models (e.g., Qwen3-VL) | Context-aware text understanding; interprets layouts and logos without explicit OCR. | Heavier computational load; potential for occasional hallucination. | Primary retrieval engine; balances text and visual context. |
| Human-in-the-Loop (HITL) | High user trust; allows manual verification and correction. | Requires user interaction; not fully autonomous. | Core design philosophy; outputs a Top-K shortlist with evidence. |

*Table note:* This table is an analytical summary rather than a benchmark table. It consolidates the main literature themes reviewed in this chapter and shows how each one informed the architectural stance adopted in the project.

## 2.5 Summary

The literature suggests that while visual retrieval and OCR are both well established, their naive combination is suboptimal for industrial parts because of strong visual similarity and highly noisy text extraction. Recent Vision-Language Models such as Qwen3-VL provide a more context-aware alternative, while models such as BGE-M3 strengthen metadata and identifier matching. Most importantly, work on user trust and interactive retrieval shows that these technologies are more useful when embedded in a human-in-the-loop workflow rather than presented as a fully autonomous classifier. This theoretical foundation directly motivates the retrieval-first, evidence-backed, human-in-the-loop listing assistant developed in this project.
