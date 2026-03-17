# Peer Review Submission

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. Project Overview

This project is **Smart Image Part Identifier for Secondhand Platforms**. It is a retrieval-first AI prototype that helps identify industrial and electronic parts from user photos. Instead of returning a single opaque prediction, the system combines image retrieval, OCR, metadata-aware ranking, and optional supporting evidence to produce a shortlist of likely candidates for listing assistance.

The current prototype includes:

1. image-based hybrid retrieval,
2. OCR and text-assisted search,
3. metadata preview before saving,
4. catalog PDF retrieval,
5. agent-assisted evidence gathering.

## 2. Submission Format for Peer Review

This peer review submission is intended to be reviewed primarily through a **demo video plus written instructions**.

This is the most practical submission format because the full system depends on:

- a web frontend,
- a FastAPI backend,
- Milvus vector storage,
- OCR and multimodal model runtime,
- environment-specific dependencies.

For that reason, a video-based submission is more reliable for student reviewers than asking every reviewer to reproduce the full runtime locally.

If possible, the demo should include a short voiceover. This makes it easier for reviewers to understand what they are looking at, which features are being demonstrated, and why those features matter from a user perspective.

## 3. Reviewer Questions

Please answer the following three questions using this scale:

- agree
- partially agree
- neither agree nor disagree
- partially disagree
- disagree

### Question 1

**I could understand how to use the prototype interface without additional explanation.**

### Question 2

**The retrieval results and supporting evidence looked appropriate for the example query.**

### Question 3

**The prototype seemed technically challenging and more substantial than a simple single-model demo.**

In addition, reviewers will also answer the generic criterion:

- **Is the final product of high quality?**

## 4. What Reviewers Should Look For

Reviewers should focus on:

1. whether the search flow is understandable,
2. whether the results look plausible for the demonstrated query,
3. whether the prototype appears to provide meaningful evidence rather than only a raw guess,
4. whether the project looks technically substantial.

The emphasis should be on how well the current prototype achieves its stated aims, rather than on hypothetical future improvements.

## 5. Software and Hardware Requirements

### Recommended review mode

The recommended review mode is simply to watch the demo video.

### If a reviewer wants to run the project

The following are required:

- Python 3.12+
- Node.js 20+
- Docker / Docker Compose
- Milvus
- sufficient RAM and storage
- an OpenAI API key for metadata preview and some assisted flows

Because the system combines OCR, multimodal embeddings, retrieval, reranking, and vector storage, runtime setup may vary depending on hardware and operating system. For peer review purposes, the video demonstration should therefore be treated as the primary review artifact.

## 6. Reviewer Instructions

Please review the submission in the following order:

1. Watch the demo video.
2. Observe the search input and output flow.
3. Assess whether the shortlist and evidence appear useful.
4. Answer the three Likert-scale questions above.

## 7. Short Submission Summary

This submission is a video-based demonstration of a retrieval-first AI prototype for identifying industrial and electronic parts from secondhand listing images. Reviewers are asked to evaluate interface clarity, result appropriateness, and technical challenge using the three provided questions.
