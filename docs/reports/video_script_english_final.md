# Final Video Script (English, Page-by-Page Version)

This document is the **final English script for the demo video**, designed to be read while recording the UI directly.  
It follows the actual web flow from the login page through search, indexing, catalog, and agent features.

## 1. Video Structure Principles

This video should follow these principles.

- Start from the login page and move through the real web flow in order.
- For each page, explain **what the page is for** before showing actions.
- For important buttons, explain what they do before clicking and what happens after clicking.
- Focus on what a user can do, rather than low-level implementation detail.
- End with a short summary of strengths and current limitations.

---

## 2. Key Points to Emphasize by Page

### Login Page
- The system works as a web application with a structured user flow.
- This is not just a single-model demo, but a service-like prototype.

### Search Page
- Hybrid retrieval using both text and images
- Shortlist-based results
- Evidence-backed ranking

### Indexing Page
- Metadata preview after image upload
- User review and confirm flow
- Multi-image input support

### Catalog Page
- Turning PDFs into searchable knowledge sources
- Using internal documents as evidence

### Agent Page
- Orchestration of hybrid search, catalog search, and external evidence
- A decision-support workflow, not just a search box

### Language Toggle
- Bilingual interface
- Improved practical accessibility

---

## 3. Final Script

Hello. My project is **Smart Image Part Identifier for Secondhand Platforms**.  
This project is a web-based system designed to help users identify parts and prepare listing metadata from uploaded images, especially in secondhand marketplace workflows.

I will start from the login page.  
This is the entry point to the system.  
The project is not just a single model running in isolation. It is implemented as a real web application with a structured user workflow. The web application also includes a language toggle. Users can switch between Korean and English. This is not the core algorithmic feature, but it improves usability and presentation quality. Since I am Korean, I included Korean support so that the system can be used more naturally. The interface can be switched with a button.  
After login, users can move to the search, indexing, catalog, and agent features.

Next is the indexing page.  
This page is used when a new item is being registered.  
The user can upload one image or multiple images for the same item. If different views of the item are uploaded, such as the front, side, and label surface, retrieval can become easier and more reliable.  
Instead of forcing the user to fill in all metadata manually from the beginning, the system first generates a metadata draft.  
I will now press the extraction button.

When metadata extraction is triggered, the system generates a metadata preview using Qwen or GPT-based VLM support, including fields such as maker, category, part number, and description.  
The user does not have to save this directly. They can review and edit it first.  
Only when the user presses **confirm** does actual indexing and saving happen.

This is safer than fully automatic write-back, and it fits a real secondhand listing workflow more naturally.  
It also supports multiple images, so different views such as the front, side, and label surface can all contribute evidence.  
This is especially important in fine-grained part identification.

The system can also use OCR as an additional support path to reduce hallucination. In text-heavy cases, OCR can be stronger than image-only interpretation, so scanning a label through the OCR path can also be a useful strategy.

Now I will show the search page.  
This page allows the user to search for related parts using text only, or by using both an image and text together.  
The important point here is that the system does not behave like a simple classifier. Instead, it follows a **retrieval-first approach**.  
Rather than forcing one final answer, it returns a shortlist of plausible candidates and lets the user judge them. Because part identification often depends on related candidates rather than a single definitive answer, this shortlist is very useful.

Here, the user can type text and also upload an image.  
When the search button is pressed, the system combines image embeddings, textual information, metadata signals, and lexical matching to retrieve candidates.  
Because of this, the result is not based only on “similar pictures”, but can also reflect clues such as maker names or part numbers.  
I will now press the search button.

When we look at the search results, one important strength of the system is that it tries to provide **evidence-backed results**.  
It does not only show a ranking. It is designed so that the user can understand why a candidate is relevant.  
This is one of the key strengths of the project.

Next is the Catalog page.  
This feature allows the user to upload PDF documents and turn their contents into searchable knowledge.  
For example, if a part catalogue or manual is uploaded, the system can later use that document as supporting evidence during retrieval.  
This means the system is not limited to image search alone. It can also use the user’s own reference documents as searchable knowledge sources.

Next is the Agent page.  
This feature is designed to combine hybrid search, catalog search, and, when needed, external evidence to produce richer responses.  
So instead of stopping at a shortlist, the system can support the user’s decision with multiple evidence sources.  
In addition, if a product is not already registered and the system finds useful information from the web, I added an upsert path so that it can later be reflected into Milvus.

To summarize, the main strength of this project is not only the performance of one model.  
Its real contribution is that it integrates **search, metadata assistance, catalog retrieval, agent orchestration, and user confirmation into one practical workflow**.  
In that sense, it is best described not as a simple image classifier, but as a **retrieval-first, human-in-the-loop identification assistant prototype**.

At the current stage, the system already supports an end-to-end search and indexing workflow, and the experiments and evaluation have also been completed.  
However, the most accurate way to describe it is still as a **retrieval-first, human-in-the-loop identification assistant**, rather than a fully autonomous identification system.

Thank you.

---

## 4. Short Page-by-Page Lines

If you want to speak in shorter phrases while recording, you can use these.

### Login Page
- “This is the entry point of the system.”
- “The project is implemented as a web-based workflow, not just a model demo.”

### Search Page
- “This page supports hybrid retrieval using both images and text.”
- “Instead of forcing one answer, it provides a shortlist.”

### Indexing Page
- “This page is for registering a new item.”
- “It uses a preview and confirm structure instead of automatic saving.”

### Catalog Page
- “This feature turns PDF documents into searchable knowledge sources.”

### Agent Page
- “This page orchestrates multiple tools and evidence sources.”

### Language Toggle
- “The interface supports both Korean and English.”

---

## 5. Buttons and Actions Worth Showing

- Login button
- Search button
- Image upload
- Metadata extraction button
- Confirm save button
- PDF upload
- Agent query input
- Language toggle

---

## 6. Strongest Points to Emphasize

1. **It works as a real web application**
2. **It follows a retrieval-first shortlist-based design**
3. **It provides evidence-backed results**
4. **It uses a preview -> edit -> confirm indexing workflow**
5. **It includes catalog and agent features, not just search**

---

## 7. Things Not to Over-Explain

Do not spend too much time in the video on:

- library version conflicts
- local machine environment issues
- every detailed experiment number
- low-level code structure

Those belong in the written report.  
In the video, it is better to focus on **page flow, user actions, and practical value**.
