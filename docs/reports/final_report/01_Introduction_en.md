# 1. Introduction

This project follows the CM3020 template **“Orchestrating AI models to achieve a goal.”** Its core objective is to assist non-expert sellers on secondhand marketplaces by reducing the friction of listing industrial and electronic parts. More specifically, the system is designed to help users identify the manufacturer, model name, and part number of an item primarily from user-uploaded photos.

## 1.1 Background and Motivation

While consumer goods can often be searched using their general appearance, industrial parts present a different challenge. In secondhand marketplaces, incorrect identification of an industrial component can lead to pricing errors, failed searches, and reduced buyer trust. The decisive identifying signals are rarely the overall shape of the item; instead, they are often small details such as model codes, serial numbers, or nameplates.

However, these crucial identifiers are frequently difficult to capture in user-generated photos because of glare, blur, low resolution, wear, and partial occlusion. As a result, non-expert sellers often spend significant time manually cross-referencing manuals, catalogues, and search engines in order to create accurate listings.

Industry trends suggest that photo-based assistance can significantly reduce listing friction. For example, eBay’s image-search rollout showed that users could search by pictures rather than keywords alone, and more recent AI-assisted listing tools have begun to estimate listing information directly from uploaded images [6], [7], [12]. However, these tools are mainly optimised for general consumer goods. Identifying industrial parts is substantially harder because many components appear visually identical across different variants. For this reason, the problem must be approached as more than simple generic image similarity.

## 1.2 Problem Statement

The central premise of this project is that identifying secondhand industrial parts should not be treated as a closed-set classification problem, but rather as an open-world, retrieval-first, human-in-the-loop decision-support challenge.

Initially, it was reasonable to hypothesise that aggressively combining generic visual similarity with Optical Character Recognition (OCR) would produce the best results. However, industrial-domain constraints show that OCR is often highly noisy, extracting irrelevant electrical specifications, background text, and packaging details that can actually degrade retrieval quality. Therefore, an effective and practical system should:

1. Leverage robust vision-language models to understand layout and visual context, rather than relying blindly on raw text extraction.
2. Treat OCR not as a primary search engine, but as an optional secondary verification signal.
3. Degrade gracefully when text is missing or illegible.
4. Produce structured outputs such as maker and part number that are suitable for listing workflows.
5. Present transparent evidence to the user, allowing manual verification and correction.

## 1.3 Domain and User Requirements

This framing directly addresses both domain constraints and user needs. From a domain perspective, secondhand inventories are open-world and long-tail; rare, discontinued, or newly encountered parts appear continuously, making pure closed-set classifiers less suitable. The domain is also highly fine-grained, with many parts differing only by subtle markings or identifiers.

To validate user requirements, a short elicitation survey (`n = 6`) was conducted with users familiar with secondhand platforms. The responses showed that writing detailed specifications is difficult and that reliance on manual search remains high. Although there was clear demand for photo-based identification support, users also indicated that they would not blindly trust AI-generated outputs. Transparency, the ability to inspect evidence, and the option to edit results were repeatedly emphasised. Accordingly, the system is designed to provide a shortlist of Top-K candidates together with supporting evidence, leaving the final decision to the user.

The main survey findings and their design implications are summarised in **Table 1**.

**Table 1. Survey findings and requirement implications (`n = 6`)**

| Survey finding | Responses | Requirement implication |
| --- | --- | --- |
| Writing detailed specifications is difficult | 3/6 respondents (50%) | Need structured metadata output (FR6: Part Card) |
| Users rely on manual search for identification | 5/6 use search engines; 4/6 check labels/manuals | Photo-based identification support (FR1-FR4) |
| Users are open to AI assistance | 5/6 would probably use a photo-based system | Listing assistance is desirable |
| No unconditional trust in AI outputs | 0/6 trust without review; 4/6 would still review | Human-in-the-loop confirmation (FR7) |
| Transparency and editability increase trust | 2/6 cited transparency; 2/6 editability | Explainability and edit features (NFR7) |
| Workflow speed matters | 1/6 cited speed explicitly | Latency targets (NFR3) |

*Table note:* This table summarises the small requirement-elicitation survey conducted during the early design phase. Because the sample size is limited (`n = 6`), the findings are used as directional design evidence rather than as statistically generalisable claims.

A brief summary of the survey material and response themes is included in Appendix A.

## 1.4 Aim and Objectives

**Aim**  
To build, orchestrate, and evaluate an end-to-end prototype that takes a user-uploaded photo of a part and produces a Top-5 shortlist of candidate matches, accompanied by a structured listing summary and supporting evidence, in order to streamline the listing creation process.

**Objectives**

**O1. Working MVP Workflow**  
Deliver an end-to-end flow supporting catalogue indexing, multimodal retrieval, and a human-in-the-loop confirmation step.

**O2. Retrieval Effectiveness**  
Evaluate retrieval performance using metrics such as Accuracy@1, Accuracy@5, and MRR on an open-world dataset, and investigate the balance between visual signals and OCR-derived text.

**O3. Latency and Interactive Feasibility**  
Instrument and measure component-level latency, including percentile summaries, to assess whether the system is viable for interactive listing workflows.

**O4. OCR Robustness Analysis**  
Conduct a targeted benchmark on identifier extraction using character-level error metrics in order to analyse the limitations of traditional OCR in industrial imagery.

**O5. User-Centred Evaluation**  
Assess the practical usefulness of the system through evidence transparency, editability, and pilot usability feedback.

## 1.5 Report Structure

The remainder of this report is organised as follows. Chapter 2 critically reviews related literature on visual retrieval, OCR, multimodal embeddings, and interactive systems. Chapter 3 explains the system design, including the architecture and the way evaluation strategy is embedded within it. Chapter 4 describes the implementation across the web, API, and model layers. Chapter 5 presents a critical analysis of the evaluation results, highlighting the design pivot toward a vision-language-dominant architecture based on empirical evidence. Finally, Chapter 6 summarises the project’s contributions, limitations, and future work.
