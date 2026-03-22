# Video Recording Cues (English)

This document is a practical recording guide for the compact English script.  
It tells you **which page to show, when to pause, and which button to click** while reading the script.

Recommended script:

- [`video_script_english_compact.md`](/Users/mac/project/Smart_vision/docs/reports/video_script_english_compact.md)

---

## 1. Opening Screen

### Show

- Browser open on the project home or login page

### Say

- “Hello. My project is Smart Image Part Identifier for Secondhand Platforms.”

### Pause / Action

- Hold for 2 seconds so the title screen is visible

---

## 2. Login Page

### Show

- Login page
- Language toggle visible if possible

### Say

- “This page is the entry point to the system.”
- “The project is not just a single model demo. It is implemented as a real web application with a structured user flow.”
- “The interface also supports both Korean and English through a language toggle.”

### Pause / Action

1. Briefly click the language toggle once
2. Pause for 1 second
3. Return to the preferred interface language
4. Click the login button

### Notes

- Do not spend too long here
- 15 to 20 seconds is enough

---

## 3. Indexing Page

### Show

- Index page with image upload area

### Say

- “This page is used to register a new item.”
- “The user can upload one image or multiple images for the same item.”
- “Different views, such as the front, side, and label surface, help the system gather stronger evidence.”

### Pause / Action

1. Select multiple images
2. Wait until thumbnails appear

### Continue Saying

- “Instead of requiring the user to write all metadata manually, the system first generates a metadata draft.”
- “When I press the extraction button, the system uses a VLM-based path with Qwen or GPT support to suggest fields such as maker, category, part number, and description.”

### Pause / Action

1. Click the metadata extraction button
2. Wait for the preview fields to appear
3. Pause for 2 to 3 seconds so the generated metadata is visible

### Continue Saying

- “The user can review and edit these suggestions, and only when the user presses confirm does the actual indexing and saving happen.”
- “This is safer than fully automatic write-back and fits a real listing workflow more naturally.”
- “The system can also use OCR as an additional support path, especially when a label or identifier is visible and text evidence is important.”

### Pause / Action

1. Click into one or two metadata fields to show editability
2. Optionally change one field
3. If stable to do so, click the confirm button
4. Pause briefly after confirmation

### Notes

- If confirm is slow, do not wait too long in silence
- Keep talking while the system processes

---

## 4. Search Page

### Show

- Search page
- Search form and results area

### Say

- “This page allows the user to search using text only, or using both an image and text together.”
- “The important point here is that the system follows a retrieval-first approach.”
- “Instead of forcing one final answer, it returns a shortlist of plausible candidates and lets the user judge them.”

### Pause / Action

1. Enter a short search text
2. Upload a query image if you want to demonstrate multimodal search

### Continue Saying

- “When the search button is pressed, the system combines image embeddings, text signals, metadata signals, and lexical matching to retrieve candidates.”
- “Because of this, the result is not based only on visual similarity. It can also reflect clues such as maker names or part numbers.”

### Pause / Action

1. Click the search button
2. Wait for the results to load

### Continue Saying

- “When we look at the results, an important strength is that the system tries to provide evidence-backed output.”
- “It does not only show a ranking. It is designed so that the user can understand why a candidate is relevant.”

### Pause / Action

1. Scroll slightly through the results
2. Hover over or point at evidence fields, scores, or result metadata
3. Pause for 2 seconds so the shortlist is clearly visible

---

## 5. Catalog Page

### Show

- Catalog upload/search page

### Say

- “This feature allows the user to upload PDF documents and turn them into searchable knowledge sources.”
- “For example, a parts catalogue or manual can later be used as supporting evidence during retrieval.”

### Pause / Action

1. Show the upload area
2. If prepared, select a PDF
3. If upload is already completed beforehand, show the resulting catalog list or search area instead

### Notes

- Keep this section short
- Around 15 to 20 seconds is enough

---

## 6. Agent Page

### Show

- Agent page with chat interface

### Say

- “This feature combines hybrid search, catalog search, and, when needed, external evidence to provide richer support.”
- “In addition, if a product is not already registered and useful information is found through the web path, I added an upsert path so that it can later be reflected into Milvus.”

### Pause / Action

1. Type a short query into the agent input
2. Send the query
3. Pause while the response appears

### Notes

- If the response is long, scroll only slightly
- The goal is to show orchestration, not to read the full answer

---

## 7. Closing Summary

### Show

- Return to a stable page, ideally search results or the main app shell

### Say

- “Its real contribution is that it integrates search, metadata assistance, catalog retrieval, agent orchestration, and user confirmation into one practical workflow.”
- “In that sense, it is best described as a retrieval-first, human-in-the-loop identification assistant prototype.”
- “The most accurate description of the project is therefore not a fully autonomous identification system, but a practical identification assistant that helps users narrow down candidates, inspect evidence, and prepare listing information more efficiently.”
- “Thank you.”

### Pause / Action

- Hold the final screen for 2 seconds after saying “Thank you.”

---

## 8. Practical Timing Target

Recommended approximate timing:

- Opening + login: 20 seconds
- Indexing: 60 to 80 seconds
- Search: 60 to 80 seconds
- Catalog: 15 to 20 seconds
- Agent: 20 to 30 seconds
- Closing: 20 seconds

Total target:

- About **3 minutes 30 seconds to 4 minutes 30 seconds**

---

## 9. Recording Tips

- Do one full dry run before recording
- Prepare images and PDFs in advance
- Avoid silent waiting during long actions
- If a page is slow, keep speaking while it loads
- If one feature is unstable, shorten that section instead of forcing a long demo
