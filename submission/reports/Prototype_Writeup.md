# Prototype Write-Up

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. Template Used

This prototype follows the CM3020 template **"Orchestrating AI models to achieve a goal."**

## 2. Project Overview

The overall project aims to help users identify industrial and electronic parts from photos in secondhand listing workflows. This is not well framed as a closed-set classification problem because secondhand inventories are open-world, long-tail, and fine-grained. Many items look visually similar, while the decisive evidence is often a small maker label, model code, or part number.

This prototype implements the core identification workflow of the project. A user uploads one or more images, and the system combines OCR, multimodal embeddings, vector search, lexical matching, reranking, and structured metadata handling to produce a shortlist of likely candidates. In the wider project, this retrieval core is also connected to internal catalog search and an agent layer for evidence gathering.

## 3. Features Implemented

The main implemented features are:

1. Hybrid retrieval over image, OCR text, metadata text, and optional user query text.
2. Metadata-aware ranking boosts for maker, part number, and description matches.
3. Multi-image indexing so multiple views of the same product can be stored under one model.
4. GPT-based metadata preview before indexing, allowing the user to review and correct fields.
5. Catalog retrieval over indexed PDF documents.
6. Agent orchestration that can combine hybrid search, catalog search, and web search.
7. Safer write-back behaviour, where saving is confirmation-based rather than automatic.

## 4. Algorithms, Techniques, and Methods

The prototype uses a retrieval-first hybrid pipeline rather than a single prediction model.

### 4.1 Retrieval Flow

```text
User image and optional text
 -> OCR / caption / embedding generation
 -> Milvus search across multiple collections
 -> candidate merge by model_id
 -> lexical/spec-aware score adjustment
 -> reranking
 -> Top-K shortlist
```

This design is appropriate because:

- image similarity alone is not sufficient for fine-grained part identification,
- OCR is useful but noisy,
- users need a shortlist with evidence rather than an opaque single guess.

### 4.2 Techniques Used

- OCR: PaddleOCRVL with PaddleOCR fallback
- Image embedding: Qwen3-VL-Embedding-2B
- Text embedding: BGE-M3
- Reranking: Qwen3-VL-Reranker-2B
- Metadata and caption support: Qwen3-VL / GPT-based paths
- Retrieval backend: Milvus multi-collection vector search

### 4.3 Technical Challenge

This prototype is technically challenging because it is not a simple model demo. It must orchestrate multiple uncertain evidence channels:

- OCR may fail,
- text may be missing,
- visually similar variants may exist,
- user images may be noisy,
- output must be structured enough for a listing workflow.

This makes the prototype a systems-level AI integration task rather than a single-model prediction task.

## 5. Explanation of Important Code

Only key parts of the code are discussed here.

### 5.1 Hybrid Search Orchestrator

The core component is the `HybridSearchOrchestrator`. It handles preprocessing, OCR, embedding generation, Milvus retrieval, candidate merging, and final ranking. The important design choice is that results from different channels are merged by `model_id`, rather than treated as unrelated hits.

This matters because one candidate may have strong image similarity while another may have better OCR or exact textual evidence. The orchestrator combines these signals into a single result structure.

### 5.2 Score Fusion

The ranking logic does not rely on dense similarity alone. It also uses lexical evidence and exact-field boosts:

```python
final_score = min(1.0, similarity * 0.65 + lexical_score * 0.20 + exact_field_boost * 0.15)
```

This means the system can favour a result that not only looks similar, but also contains the correct maker or part number. That is important in this domain because visually similar parts are common.

### 5.3 Metadata Preview Before Saving

Another important implementation feature is metadata preview before indexing. Instead of immediately saving uploaded images, the system first generates a metadata draft using GPT. The user can then correct maker, part number, category, and description before confirming the save. This is a better fit for real listing workflows and is safer than automatic write-back.

## 6. Visual Representation

The final PDF should include:

1. an overall architecture diagram,
2. a screenshot of the search interface,
3. a screenshot of the metadata preview and confirm flow,
4. an example of hybrid score evidence,
5. OCR and retrieval failure examples.

These visuals will make the prototype easier to understand and will support the evaluation section.

## 7. Evaluation of Prototype Success

The prototype is successful as an end-to-end retrieval-first identification assistant.

Strengths:

- A working prototype exists across the web, API, and model layers.
- The retrieval-first design is implemented in code, not just proposed conceptually.
- The system already shows why image-only search is not enough and why multimodal retrieval is justified.
- Catalog retrieval, agent orchestration, and safer write-back improve the prototype beyond a simple demo.

Current limitations:

- OCR CER benchmarking is not yet fully completed.
- Full hybrid ablation results are still pending.
- Latency percentile reporting is instrumented but not yet fully summarised.
- A complete accept/edit/reject review workflow is still partial.

The fairest evaluation is therefore that the system is a **retrieval-first, human-in-the-loop identification assistant**, not a fully autonomous identification system.

## 8. How It Can Be Improved

The next improvements should be:

1. automated fixed-split retrieval benchmarking,
2. OCR on/off and reranker on/off ablation experiments,
3. batch latency reporting with percentile summaries,
4. a complete reviewed write-back workflow,
5. full validation of the latest model stack in a stronger runtime environment.

## 9. Conclusion

This prototype demonstrates a technically meaningful orchestration of OCR, multimodal embeddings, vector retrieval, reranking, and user-confirmed metadata handling for a difficult open-world identification task. Its strength lies in integrating multiple AI components into a practical workflow, rather than relying on a single model to solve the problem alone.

For final submission, this text should be converted to PDF, shortened or tightened if needed to stay within the 2000-word limit, and supplemented with screenshots and diagrams.
