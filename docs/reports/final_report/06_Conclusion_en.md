# 6. Conclusion and Future Work

This chapter summarises the project’s main achievements, highlights its architectural and technical contributions, and discusses the limitations that define the boundary between the current prototype and future productisation or research work.

## 6.1 Project Summary

This project set out to reduce the friction of identifying and listing industrial and electronic parts on secondhand marketplaces. Following the “Orchestrating AI models” template, the system was designed to support non-expert sellers by helping them infer listing-relevant metadata from user-uploaded photos and related evidence.

An important change emerged during evaluation. The initial working assumption was that an OCR-heavy retrieval pipeline would provide the strongest results by extracting explicit identifiers. However, the empirical benchmark results showed that, in industrial settings, traditional OCR often introduced both severe noise and substantial latency. As a result, the project pivoted toward a vision-dominant hybrid retrieval architecture that uses Qwen3-VL for layout-aware image understanding and BGE-M3 for robust text representation [1], [2]. This orchestration is embedded within a human-in-the-loop workflow, so the system functions as an evidence-backed decision-support assistant rather than a fully autonomous classifier [3].

## 6.2 Key Contributions

The main contributions of the project can be summarised as follows:

1. **Empirical evidence on OCR limitations**  
   The project provides concrete experimental evidence that blindly treating OCR as a default first-stage retrieval signal can reduce retrieval quality and increase latency in noisy industrial imagery.

2. **A retrieval-first vision-language orchestration pipeline**  
   The system successfully integrates Qwen3-VL-based image understanding, BGE-M3 text retrieval, metadata-aware scoring, and hybrid ranking into a single operational retrieval workflow [1], [2]. In the main benchmark, the stronger vision-dominant configuration achieved approximately `91%` Accuracy@1.

3. **A practical human-in-the-loop prototype**  
   The project delivered a working end-to-end prototype built around a `preview-confirm` workflow. This directly addresses the user requirements for transparency, editability, and evidence visibility, shifting the system from black-box prediction toward AI-assisted user verification [3].

4. **A system-level contribution beyond a single model demo**  
   The project integrates search, indexing, catalog retrieval, and agent-assisted evidence expansion into one coherent workflow. This is important because the core contribution is not a new foundation model, but a practical orchestration of multiple AI components for a real problem setting.

## 6.3 Limitations and Future Work

Although the prototype validates the retrieval-first architecture, several limitations remain and define the next stage of work.

- **Workflow enhancements**  
  The current indexing path begins with image upload. A useful future extension would be **metadata-only draft registration**, allowing users to start a listing from known metadata and attach images later. In addition, the current `preview-confirm` sequence could be expanded into a fully **audited review-and-writeback workflow** with revision history and rollback support.

- **Selective OCR and region-focused refinement**  
  The evaluation suggests that OCR should not be applied indiscriminately. A future version should adopt a **selective verification policy**, triggering OCR only when additional evidence is needed. Region-focused processing, such as label-region detection, rotation correction, and multi-view evidence aggregation, would also likely improve difficult cases.

- **Scaling and deployment**  
  For broader operational use, the system should be migrated to a stable Linux/GPU environment to reduce local hardware constraints and improve reproducibility for heavier multimodal paths.

- **Broader validation**  
  While the current experiments provide useful evidence, future work should include larger-scale and more diverse evaluation, including more extensive user studies with novice and experienced sellers, stronger hard-case benchmarks, and broader deployment-oriented validation.

## 6.4 Concluding Remarks

Fine-grained secondhand part identification is fundamentally different from generic consumer image search. This project shows that a naïve accumulation of AI components does not automatically improve system quality. Instead, the most effective design emerged from critically testing the trade-offs between visual retrieval, OCR, reranking, latency, and user trust.

The final result is best described as a **retrieval-first, human-in-the-loop identification assistant**. This description matches both the implemented workflow and the available evidence. The project therefore makes a meaningful contribution not by claiming fully autonomous identification, but by demonstrating that state-of-the-art AI components can be orchestrated into a practical, evidence-backed listing assistant for a difficult real-world domain.
