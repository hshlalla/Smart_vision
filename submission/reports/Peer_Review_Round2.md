# Peer Review Round 2 Submission

University of London  
Bachelor in Computer Science  
CM3020 Artificial Intelligence

## 1. Project Template

This project follows the CM3020 template **"Orchestrating AI models to achieve a goal."**

## 2. Project Overview

The project is **Smart Image Part Identifier for Secondhand Platforms**. It is a retrieval-first AI prototype that helps identify industrial and electronic parts from user photos in secondhand listing workflows. Instead of making a single opaque prediction, the system combines OCR, multimodal embeddings, vector retrieval, metadata-aware ranking, and supporting evidence to return a shortlist of likely candidates.

## 3. Instructions for Running and Testing the Project

### Recommended review mode

The recommended review mode is to watch the demo video.

This is the most practical option because the full system depends on:

- a web frontend,
- a FastAPI backend,
- Milvus vector storage,
- OCR runtime,
- multimodal model dependencies,
- environment-specific software compatibility.

### If a reviewer wants to run the project

The following are required:

- Python 3.12+
- Node.js 20+
- Docker / Docker Compose
- Milvus
- sufficient RAM and storage
- an OpenAI API key for some assisted flows

### Practical note

Because the prototype combines several heavy dependencies, the video demonstration should be treated as the primary review artifact for peer review purposes.

Ideally, the demo should be a short 3-5 minute voiceover video rather than a silent screen capture. This helps reviewers understand what they are seeing, which features are important, and how those features relate to user needs.

### Suggested review procedure

1. Watch the demo video.
2. Observe the input flow and the output shortlist.
3. Check whether the evidence and results seem plausible.
4. Answer the three questions below.

The emphasis should be on how well the current prototype achieves its stated aims, rather than on hypothetical future improvements.

## 4. Reviewer Questions

Please answer the following questions using this scale:

- Disagree
- Partially disagree
- Neither agree nor disagree
- Partially agree
- Agree

### Question 1

**I could understand how to use the prototype interface without additional explanation.**

### Question 2

**The retrieval results and supporting evidence looked appropriate for the demonstrated query.**

### Question 3

**The prototype appeared technically challenging and more substantial than a simple single-model demo.**

## 5. Generic Criterion

In addition to the three questions above, reviewers will also consider the generic criterion:

- **Is the final product of high quality?**

## 6. What Reviewers Will Provide

In the peer review system, reviewers will provide:

1. a response to Question 1,
2. a response to Question 2,
3. a response to Question 3,
4. a response to the generic quality criterion,
5. comments on the best things about the project,
6. suggestions for improvement.

This means the submission should focus on giving reviewers clear instructions and three appropriate Likert-scale questions, which is what this document provides.
