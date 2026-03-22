# 5. Evaluation

This chapter critically evaluates the implemented system against the project’s core aim: to provide a robust, interactive identification assistant for secondhand parts. The evaluation focuses on retrieval effectiveness, latency trade-offs, and the practical limitations of OCR in industrial settings.

## 5.1 Evaluation Methodology

To keep the evaluation close to the real problem domain, the main offline benchmark was framed around a dataset of `1,000` industrial and electronic part images.

- **Dataset split:** `900` index images and `100` query images.
- **Main benchmark comparison:** `C2` versus `C4`.
- **Supporting local validation:** `C3` and `C1` were used to refine the final practical recommendation.
- **Primary metrics:** `Accuracy@1`, `Accuracy@5`, `MRR`, exact identifier hit rate, and latency.
- **Runtime environment:** Apple Silicon, `32GB` unified memory.

The compared configurations are summarised in **Table 5.1**.

**Table 5.1. Configuration summary and operational interpretation**

| Configuration | Description | Accuracy@1 | Latency impact | Operational recommendation |
| --- | --- | ---: | --- | --- |
| C2 | OCR ON + Reranker ON | Lower (due to text noise) | High | Not recommended as the default path |
| C4 | OCR OFF + Reranker ON | ~91% | Medium | Strong main-benchmark configuration |
| C3 | OCR OFF + Reranker OFF | Highly competitive | Low (fastest) | Recommended default operating configuration |

*Table note:* `C2` and `C4` were directly compared in the main `1000-item` report benchmark. `C3` was adopted as the final operational recommendation after additional local validation showed that disabling the reranker preserved practical retrieval quality while substantially reducing latency.

Additional experimental artefacts and extended result tables are provided in Appendix D.

## 5.2 Retrieval Effectiveness (C2 vs C4)

The initial hypothesis was that aggressively extracting text through OCR would improve retrieval by recovering exact identifiers such as part numbers. However, the empirical results did not support this assumption.

**[Insert Table 5.2: Retrieval effectiveness comparison (C2 vs C4)]**  
*(작성 가이드: Accuracy@1, Accuracy@5, MRR, exact identifier hit를 포함한 표 삽입)*

In the main benchmark, `C2` achieved `Accuracy@1 = 0.86`, `Accuracy@5 = 0.95`, `MRR = 0.903`, and `Exact identifier hit = 0.81`. By contrast, the vision-dominant configuration `C4` achieved `Accuracy@1 = 0.91`, `Accuracy@5 = 0.97`, `MRR = 0.939`, and `Exact identifier hit = 0.88`.

This is an important result. It shows that, in this domain, a cleaner vision-first retrieval path can outperform an OCR-heavy path even when OCR appears to add more textual evidence. In practice, the extra OCR-derived text often introduced noise rather than useful discriminative information.

## 5.3 Latency and System Performance

Interactive responsiveness is essential for a human-in-the-loop listing assistant. If preview generation or search takes too long, the system becomes difficult to use in a realistic marketplace workflow.

**[Insert Figure 5.1: Latency comparison across configurations]**  
*(작성 가이드: C2, C4, C3의 total latency를 비교하는 bar chart 삽입)*

Latency profiling showed that `C2` imposed the largest runtime cost. Its warm mean total latency was approximately `8.24s`, with OCR dominating the total runtime. `C4` reduced this to approximately `1.42s` by removing the OCR-heavy path while still keeping reranking enabled.

Additional local validation then showed that `C3 (OCR OFF, Reranker OFF)` reduced warm mean total latency further to approximately `731.13ms`. This made `C3` the most practical operating mode under the current deployment environment.

The latency results therefore support a clear engineering conclusion: removing always-on OCR provides the largest speed gain, and removing reranking further improves responsiveness when the reranker does not provide a meaningful quality benefit.

## 5.4 Qualitative Analysis of OCR Limitations

To understand why the OCR-heavy configuration underperformed, a qualitative error analysis was carried out on retrieval failures and difficult identifier cases.

Three recurring patterns were observed:

1. **Irrelevant specification noise**  
   OCR extracted visible strings such as `12V`, `50Hz`, resistance values, warning labels, and table fragments. These strings were often semantically unhelpful for identification and diluted the importance of true identifiers.

2. **Layout and orientation failure**  
   Industrial labels frequently contain mixed vertical and horizontal text, oblique viewing angles, or partially occluded identifiers. Traditional OCR struggled to preserve the intended reading order under these conditions.

3. **Logo and context failure**  
   Manufacturer identity was sometimes expressed through stylised logos or contextual label structure rather than plain text. Vision-language modelling handled such cases more robustly than text-only extraction.

**[Insert Figure 5.2: Example OCR failure cases in industrial imagery]**  
*(작성 가이드: noisy label, irrelevant specs, mixed orientation이 보이는 사례 이미지 삽입)*

These failure modes help explain why raw OCR text is not a reliable primary retrieval signal in this problem setting.

## 5.5 OCR Benchmark and Verification Role

The OCR-specific benchmark should be interpreted separately from the main retrieval benchmark. On the identifier-visible subset, the measured results were:

- **PaddleOCR:** Exact full-string `0.19`, CER `1.12`, part-number recall `0.35`, maker recall `0.44`
- **Qwen-only:** Exact full-string `0.57`, CER `0.46`, part-number recall `0.75`, maker recall `0.82`
- **OCR + Qwen merged:** Exact full-string `0.61`, CER `0.41`, part-number recall `0.79`, maker recall `0.86`

These results suggest that OCR still has value, but not as a default first-stage retrieval driver. Instead, OCR is more useful as a supplementary verification signal that can support or cross-check candidate interpretation when needed.

## 5.6 Usability Pilot Results

A small post-task usability pilot was also conducted. The response file includes `6` submissions, but one entry was invalid because the external testing link was offline at the time of access. The usable participant count was therefore `n = 5`.

The questionnaire summary is shown in **Table 5.3**.

**Table 5.3. Pilot usability questionnaire summary (`n = 5` usable responses)**

| Item | Mean score |
| --- | ---: |
| I could understand how to use the interface without help. | 4.4 / 5 |
| The search results were useful for narrowing down candidate parts. | 5.0 / 5 |
| The evidence shown with the results helped me judge whether the result was trustworthy. | 4.4 / 5 |
| The metadata preview reduced the effort needed to prepare listing information. | 4.4 / 5 |
| I would prefer using this prototype over completely manual search for similar tasks. | 4.6 / 5 |
| I felt confident making a decision from the shortlist provided by the system. | 4.8 / 5 |

*Table note:* The raw response file contained six submissions. One response was excluded from the summary because the participant could not access the prototype due to an offline external test link, so the usable response count was `n = 5`. A raw summary is provided in Appendix D.

Across the usable responses, the mean Likert scores were:

- interface understanding: `4.4 / 5`
- usefulness of search results: `5.0 / 5`
- trust support from evidence display: `4.4 / 5`
- usefulness of metadata preview: `4.4 / 5`
- preference over fully manual search: `4.6 / 5`
- confidence in shortlist-based decision-making: `4.8 / 5`

The qualitative feedback was also broadly positive. Participants consistently highlighted the relevance of the shortlist, the usefulness of evidence display, and the reduction in manual effort during metadata preparation. Suggested improvements focused on product refinement rather than core usability failure, such as compare-view support, clearer onboarding hints, export integration, and confidence-score display.

Representative comments help illustrate this pattern. One participant wrote that *“The shortlist of candidate parts was highly relevant. It successfully filtered out the noise and showed exactly what I was looking for.”* Another highlighted the evidence layer directly, describing *“the ‘evidence’ section that explains why a certain part was suggested”* as the most useful feature because it increased confidence in the system’s judgement. At the same time, the improvement suggestions were mostly additive rather than corrective, including requests for a compare view, confidence scores, onboarding tooltips, and export support.

These results should be interpreted cautiously because the sample size remains small, but they are still useful as pilot evidence that the workflow is understandable, trusted, and practically relevant to listing assistance.

## 5.7 Critical Analysis Against Project Objectives

The evaluation results require a critical revision of the project’s earlier assumption that “adding more AI components” would automatically improve performance. In practice, a naïve combination of vision, OCR, and reranking degraded both speed and retrieval quality in the target domain.

The main benchmark showed that `C4` outperformed `C2`, indicating that the OCR-heavy path was not the strongest design choice. The additional local validation then showed that `C3` could preserve highly competitive retrieval quality while delivering the lowest practical latency. Taken together, these findings support the following final position:

- the system should remain **retrieval-first** rather than classifier-first,
- vision-language-based image understanding should be the **primary retrieval path**,
- OCR should be treated as **selective supporting evidence** rather than a universal default,
- and the user should remain in control of the final decision through a **human-in-the-loop workflow**.

This design pivot from an OCR-heavy pipeline to a retrieval-first, evidence-backed, human-in-the-loop assistant is the central technical conclusion of the project.

## 5.8 Final Evaluation Summary

The main findings of the evaluation can be summarised as follows:

1. In the main `1000-item` benchmark, `C4` outperformed `C2` in both retrieval quality and latency.
2. OCR-heavy retrieval increased noise and cost more than expected in industrial imagery.
3. Additional local validation showed that `C3` was the most practical default configuration under the current runtime constraints.
4. OCR remains useful as a verification aid rather than as the main retrieval engine.
5. The pilot usability responses indicate that users found the shortlist, evidence display, and metadata preview practically useful.
6. The evaluation supports the project’s broader architectural claim that a retrieval-first, human-in-the-loop assistant is more appropriate than a fully automatic classifier for this domain.
