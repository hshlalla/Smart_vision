# 5. Evaluation

This chapter critically evaluates the implemented Smart Image Part Identifier system against the project's overarching aim: to provide a robust, interactive, and practically responsive identification assistant for secondhand industrial and electronic parts. The evaluation is organised around three main dimensions: an offline benchmark focused on retrieval effectiveness and computational latency, a qualitative analysis of OCR failure modes in industrial imagery, and a pilot usability study examining the effectiveness of the human-in-the-loop (HITL) workflow.

## 5.1 Benchmark Design, Dataset, and Metrics

To keep the evaluation close to the real problem domain, the main offline benchmark was defined over a dataset of exactly `1,000` industrial and electronic part images. The phrase "1000-item benchmark" in this report therefore refers to a `1,000`-image dataset with a held-out `900 / 100` gallery-query split rather than to `1,000` independent queries.

Unlike standard academic image datasets, which often contain clearly separated categories and centrally framed objects, this dataset is composed of user-generated photographs of highly homogeneous industrial parts. The images include realistic marketplace perturbations such as cluttered backgrounds, metallic glare, blur, low contrast, and mixed text orientations. These characteristics make the benchmark more representative of secondhand listing conditions than a clean closed-set classification dataset.

The main benchmark was structured as follows.

- **Index gallery:** `900` items were indexed into the retrieval system to represent the existing marketplace inventory.
- **Held-out query set:** `100` unseen items were reserved as query inputs to simulate new user uploads.
- **Main benchmark comparison:** `C2` versus `C4`.
- **Supporting local validation:** `C3` and `C1` were evaluated in a separate current-index validation setting to refine the final operational recommendation.
- **Runtime environment:** Apple Silicon with `32GB` unified memory.

The main reported metrics are defined as follows.

- **Accuracy@1:** the proportion of queries for which the correct item appears in the Top-1 result.
- **Accuracy@5:** the proportion of queries for which the correct item appears within the Top-5 shortlist. This metric is especially important because the HITL interface is designed around shortlist review rather than fully automatic single-label prediction.
- **MRR (Mean Reciprocal Rank):** a ranking-quality measure that rewards placing the correct item near the top of the list.
- **Exact identifier hit:** the proportion of queries for which the exact maker and part identifier were correctly recovered or aligned during retrieval.
- **CER (Character Error Rate):** the normalised character-level error of identifier extraction in the OCR-focused sub-benchmark; lower values indicate fewer extraction errors.
- **Mean total latency:** the end-to-end runtime per query under the stated hardware environment.

Extended result artefacts, raw summaries, and protocol notes are provided in Appendix F.

## 5.2 Evaluated Configurations and Main Benchmark Results

The central engineering question in this project was how to balance visual understanding, exact text extraction, and computational speed. Three configurations were therefore considered in the broader evaluation process, although only `C2` and `C4` formed the directly controlled main benchmark:

- **C2 (OCR-heavy legacy hypothesis):** OCR enabled, reranker enabled.
- **C4 (vision-dominant main benchmark configuration):** OCR disabled, reranker enabled.
- **C3 (fast practical baseline):** OCR disabled, reranker disabled. This was validated separately in a local current-index setting rather than in the main controlled benchmark.

The initial hypothesis was that aggressively extracting text through OCR would improve retrieval by recovering explicit identifiers such as part numbers. The main `900 / 100` benchmark did not support this assumption.

**Table 5.1. Main benchmark retrieval comparison (`900`-item gallery, `100` held-out queries)**

| Configuration | Accuracy@1 | Accuracy@5 | MRR | Exact identifier hit | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 (OCR ON + Reranker ON) | 0.86 | 0.95 | 0.903 | 0.81 | 8.24s |
| C4 (OCR OFF + Reranker ON) | 0.91 | 0.97 | 0.939 | 0.88 | 1.42s |

*Table note:* Table 5.1 reports the directly comparable main benchmark. Both rows use the same `900 / 100` split and the same evaluation protocol.

As shown in Table 5.1, the vision-dominant configuration `C4` outperformed the OCR-heavy configuration `C2` on every reported retrieval-quality metric while also reducing mean latency from `8.24s` to `1.42s`. This is a strong result because it shows that, in this domain, additional OCR text does not automatically improve retrieval. Instead, it often injects noise into the ranking process.

This difference is especially visible in the ranking metrics. The OCR-heavy path reduced `MRR` from `0.939` to `0.903`, indicating that noisy OCR evidence not only hurt final Top-1 accuracy but also degraded the overall ranking order. The `Accuracy@5` result is equally important for the human-in-the-loop workflow. With `Accuracy@5 = 0.97`, configuration `C4` placed the correct item inside the review shortlist for `97%` of held-out queries, which is operationally meaningful even when the system is not acting as a fully automatic classifier.

## 5.3 Supporting Local Validation and Operational Recommendation

The final operational recommendation was not based on the main benchmark alone. Additional local validation was also conducted on the current indexed environment to test whether reranking materially improved practical retrieval quality.

**Table 5.2. Supporting local validation for the final operating recommendation**

| Configuration | Group Hit@1 | Group Hit@5 | MRR | Exact item Top-1 | Warm mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 (OCR OFF + Reranker OFF) | 1.0000 | 1.0000 | 1.0000 | 0.9667 | 731.13ms |
| C1 (OCR OFF + Reranker ON) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 89337.71ms |

*Table note:* Table 5.2 comes from a separate local validation setting and is not directly interchangeable with Table 5.1. It is included to justify the practical operating recommendation rather than to replace the main benchmark.

The local validation shows that the reranker contributed negligible practical quality gain in the tested environment while imposing an extreme runtime penalty. `C1` improved exact item Top-1 from `0.9667` to `1.0000`, but increased warm mean total latency from `731.13ms` to `89337.71ms`. Under the current deployment constraints, that trade-off is not operationally acceptable.

**Figure 5.1. Mean end-to-end latency across C2, C4, and C3.**  
*Insert a bar chart here comparing mean total latency for `C2 (8.24s)`, `C4 (1.42s)`, and `C3 (731.13ms)`. The figure should make the OCR bottleneck visually obvious and show why `C3` is the most practical default under the current runtime constraints.*

Latency profiling clarifies where this cost came from. In the report benchmark, `C2` was dominated by OCR-stage overhead, with approximately `7.11s` spent in preprocessing and OCR alone. In `C4`, the removal of always-on OCR reduced the cost profile substantially, leaving preprocessing and reranking as the main remaining contributors. This confirms that traditional OCR is the dominant latency bottleneck in the legacy pipeline.

Taken together, Table 5.1 and Table 5.2 support a clear engineering conclusion. `C4` was the strongest configuration in the main controlled benchmark, but `C3` became the final recommended default operating mode because it preserved strong practical retrieval behaviour in local validation while reducing runtime to an interactive level.

## 5.4 Qualitative Analysis of OCR Limitations

To understand why the OCR-heavy configuration underperformed, a qualitative error analysis was carried out by manually inspecting difficult retrieval failures and identifier-visible cases from the held-out query set.

Three recurring patterns were observed:

1. **Irrelevant specification noise**  
   OCR extracted visible strings such as `12V`, `50Hz`, current ratings, resistance values, warning labels, and table fragments. Although these strings were often technically correct as text, they were semantically weak for primary identification. When injected into the retrieval signal, they diluted the weight of the true identifiers and promoted false positives based on shared specifications rather than actual part identity.

2. **Layout and orientation failure**  
   Industrial labels frequently contain mixed vertical and horizontal text, oblique viewpoints, cylindrical placement, and partially occluded identifiers. Traditional OCR struggled to preserve the intended reading order or group logically related tokens under such spatial conditions.

3. **Logo and context failure**  
   Manufacturer identity was sometimes expressed through stylised logos or contextual label structure rather than plain text. Vision-language modelling handled such cases more robustly because it could use the visual form and layout prominence of the mark rather than attempting to decode it only as standard characters.

**Figure 5.2. Representative OCR failure cases in industrial imagery.**  
*Insert example images here showing at least three failure types: irrelevant electrical specifications such as `12VDC` or `50Hz`, cluttered or reflective labels, and mixed vertical-horizontal text layouts. The figure should illustrate why raw OCR output becomes noisy even when some visible strings are technically correct.*

These failure modes qualitatively explain why raw, uncontextualised OCR text is not a reliable primary retrieval signal for this domain and directly support the quantitative drop in `MRR` observed in the OCR-heavy path.

## 5.5 OCR Benchmark and Verification Role

The OCR-specific benchmark should be interpreted separately from the main retrieval benchmark. It focuses on an identifier-visible subset and measures extraction quality rather than end-to-end retrieval quality.

**Table 5.3. Identifier extraction benchmark on the identifier-visible subset**

| Method | Exact full-string | CER | Part-number recall | Maker recall |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR | 0.19 | 1.12 | 0.35 | 0.44 |
| Qwen-only | 0.57 | 0.46 | 0.75 | 0.82 |
| OCR + Qwen merged | 0.61 | 0.41 | 0.79 | 0.86 |

The CER column is especially informative. Because CER is a normalised edit-distance measure, values above `1.0` indicate that the predicted output contains severe extra noise relative to the target identifier rather than only small character substitutions. In practical terms, the `PaddleOCR` result (`CER = 1.12`) shows that raw OCR output was heavily polluted by irrelevant strings and structural reading errors.

The pattern in Table 5.3 is therefore important. OCR alone was weak on exact identifier recovery, while the Qwen-only path produced substantially better identifier quality. A merged OCR+Qwen strategy improved the strongest scores slightly, but at additional runtime cost. This supports a narrower role for OCR: not as the default first-stage retrieval driver, but as supplementary verification evidence when additional confirmation is needed.

## 5.6 Usability Pilot Results

To validate the human-in-the-loop workflow, a small post-task usability pilot was also conducted. The response file contained `6` submissions, but one entry was excluded because the participant could not access the prototype due to an offline external test link. The usable participant count was therefore `n = 5`.

**Table 5.4. Pilot usability questionnaire summary (`n = 5` usable responses)**

| Item | Mean score |
| --- | ---: |
| I could understand how to use the interface without help. | 4.4 / 5 |
| The search results were useful for narrowing down candidate parts. | 5.0 / 5 |
| The evidence shown with the results helped me judge whether the result was trustworthy. | 4.4 / 5 |
| The metadata preview reduced the effort needed to prepare listing information. | 4.4 / 5 |
| I would prefer using this prototype over completely manual search for similar tasks. | 4.6 / 5 |
| I felt confident making a decision from the shortlist provided by the system. | 4.8 / 5 |

*Table note:* The raw response file contained six submissions. One response was excluded because the participant could not access the prototype due to an offline external test link, so the usable response count was `n = 5`. A cleaned raw summary is provided in Appendix F.

The quantitative pattern is broadly positive. Participants rated shortlist usefulness highest (`5.0 / 5`) and reported strong confidence in making a final decision from the shortlist (`4.8 / 5`). Interface learnability, evidence-based trust, and the usefulness of metadata preview all remained positive at `4.4 / 5` or above. This usability pattern is broadly consistent with the high `Accuracy@5` result of the vision-dominant retrieval path, although the pilot should still be interpreted as small-scale supporting evidence rather than as a formal causal validation.

The qualitative feedback was also consistent with the retrieval-first architecture. Participants repeatedly highlighted the relevance of the shortlist, the usefulness of evidence display, and the reduction in manual effort during metadata preparation. One participant wrote that *"The shortlist of candidate parts was highly relevant. It successfully filtered out the noise and showed exactly what I was looking for."* Another identified *"the 'evidence' section that explains why a certain part was suggested"* as the most useful feature. Suggested improvements were primarily additive, such as compare-view support, clearer onboarding hints, export integration, and confidence-score display, rather than evidence of a core workflow failure.

## 5.7 Critical Analysis Against Project Objectives

The evaluation results require a critical revision of the earlier assumption that adding more AI components would automatically improve performance. In practice, the naive accumulation of vision, OCR, and reranking degraded both speed and retrieval quality in the target domain.

The main benchmark showed that `C4` outperformed `C2`, indicating that the OCR-heavy path was not the strongest design choice. The supporting local validation then showed that `C3` preserved strong practical retrieval behaviour while delivering the lowest acceptable latency. The OCR-specific benchmark further clarified that OCR still has value, but primarily as selective verification evidence rather than as an always-on first-stage retrieval engine. Finally, the pilot usability results showed that the shortlist-and-evidence workflow was understandable and useful to users.

Taken together, these findings support four final design conclusions:

1. **Retrieval-first architecture:** the system should remain retrieval-first rather than behaving as a black-box classifier.
2. **Vision-dominant primary path:** vision-language-based image understanding should act as the primary retrieval path.
3. **Selective verification:** OCR should be demoted to selective supporting evidence rather than being used by default.
4. **Human authority:** the user should retain final control through a human-in-the-loop workflow.

This pivot from an OCR-heavy pipeline to a retrieval-first, evidence-backed, human-in-the-loop assistant is the central technical conclusion of the project. Under the current deployment constraints, `C3` is therefore the most practical default operating configuration, while `C4` remains the strongest result from the controlled main benchmark.
