# 8. Appendices

This section contains supplementary materials, cleaned raw summaries, and superseded design artefacts that document the project’s iterative development from the earlier draft to the final prototype. The appendices are intended to preserve useful supporting evidence without overloading the main report. Earlier artefacts are included only where they clarify design evolution and should be labelled clearly as earlier concepts or superseded designs when they no longer match the final implementation.

## Appendix A. User Requirements and Survey Evidence

Appendix A contains the supporting material for the early requirements-elicitation survey (`n = 6`) and the later pilot usability evaluation. This appendix should include the survey prompt, a cleaned response summary, requirement-implication notes, the usability questionnaire, and a cleaned summary table derived from the pilot responses. Raw spreadsheets or screenshots should only be included when they remain interpretable and clearly labelled.

## Appendix B. Evolution of System Architecture (Draft vs. Final)

Appendix B preserves selected architecture artefacts from the draft stage, including earlier OCR-heavy pipeline concepts, broader orchestration sketches, and early interaction flows. These materials are retained to demonstrate evidence-driven refinement rather than inconsistency. In particular, they help show how the project moved from earlier OCR-heavy assumptions toward the final preview-confirm, retrieval-first, and vision-dominant architecture described in Chapter 3.

## Appendix C. User Interface Iterations

Appendix C contains supplementary interface screenshots that are too detailed for the core report or the final demonstration video. It may combine final web screenshots with a small number of earlier prototype screens, such as Gradio-based interfaces, to show how the workflow evolved from an early concept into the current login, search, indexing, catalog, and agent experience. For caption-ready wording, see `appendix_C_D_caption_drafts_en.md`.

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

Appendix D extends the qualitative discussion in Chapter 5 by presenting additional OCR and retrieval failure cases. These examples should be captioned briefly to show the failure type, such as irrelevant specification noise, mixed-orientation text, glare, blur, ambiguous labels, or wrong Top-1 but correct Top-5 retrieval. This appendix is useful because it demonstrates that failure in this domain is structured rather than random. For caption-ready wording, see `appendix_C_D_caption_drafts_en.md`.

## Appendix E. Project Management and Risk Assessment

Appendix E contains the project-management and risk-assessment material discussed more extensively in the draft. Suitable contents include milestone structures, work-breakdown notes, planning screenshots, sprint summaries, risk tables, and impact/mitigation notes. This appendix helps show that changes in scope and implementation were deliberate responses to technical findings, runtime constraints, and evaluation evidence.

## Appendix F. Evaluation Data and Extended Results

Appendix F serves as the extended technical appendix for Chapters 4 and 5. The main body should retain only the central benchmark findings, while this appendix preserves longer result tables, protocol notes, split descriptions, schema material, usability raw summaries, and API or response-structure summaries that would otherwise overload the core report. It should make clear that the final operational recommendation was derived from multiple strands of evidence rather than from a single headline metric.
