# Final Video Script (English, Compact Recording Version)

This version is a **shorter English script** intended for a recording time of roughly **4 to 4.5 minutes**, depending on speaking speed and UI pauses.

## Final Script

Hello. My project is **Smart Image Part Identifier for Secondhand Platforms**.  
This is a web-based system designed to help users identify parts and prepare listing metadata from uploaded images, especially in secondhand marketplace workflows.

I will start from the login page.  
This page is the entry point to the system.  
The project is not just a single model demo. It is implemented as a real web application with a structured user flow. The interface also supports both Korean and English through a language toggle. After login, users can move to the search, indexing, catalog, and agent features.

Next is the indexing page.  
This page is used to register a new item.  
The user can upload one image or multiple images for the same item. Different views, such as the front, side, and label surface, help the system gather stronger evidence.

Instead of requiring the user to write all metadata manually, the system first generates a metadata draft.  
When I press the extraction button, the system uses a VLM-based path with Qwen or GPT support to suggest fields such as maker, category, part number, and description.  
The user can review and edit these suggestions, and only when the user presses **confirm** does the actual indexing and saving happen.

This is safer than fully automatic write-back and fits a real listing workflow more naturally.  
The system can also use OCR as an additional support path, especially when a label or identifier is visible and text evidence is important.

Now I will show the search page.  
This page allows the user to search using text only, or using both an image and text together.  
The important point here is that the system follows a **retrieval-first approach**.  
Instead of forcing one final answer, it returns a shortlist of plausible candidates and lets the user judge them.

When the search button is pressed, the system combines image embeddings, text signals, metadata signals, and lexical matching to retrieve candidates.  
Because of this, the result is not based only on visual similarity. It can also reflect clues such as maker names or part numbers.

When we look at the results, an important strength is that the system tries to provide **evidence-backed output**.  
It does not only show a ranking. It is designed so that the user can understand why a candidate is relevant.  
This is one of the key strengths of the project.

Next is the Catalog page.  
This feature allows the user to upload PDF documents and turn them into searchable knowledge sources.  
For example, a parts catalogue or manual can later be used as supporting evidence during retrieval.

Next is the Agent page.  
This feature combines hybrid search, catalog search, and, when needed, external evidence to provide richer support.  
In addition, if a product is not already registered and useful information is found through the web path, I added an upsert path so that it can later be reflected into Milvus.

To summarize, the main strength of this project is not only the performance of one model.  
Its real contribution is that it integrates **search, metadata assistance, catalog retrieval, agent orchestration, and user confirmation into one practical workflow**.  
In that sense, it is best described as a **retrieval-first, human-in-the-loop identification assistant prototype**.

At the current stage, the system already supports an end-to-end search and indexing workflow, and the experiments and evaluation have also been completed.  
The most accurate description of the project is therefore not a fully autonomous identification system, but a practical identification assistant that helps users narrow down candidates, inspect evidence, and prepare listing information more efficiently.

Thank you.

## Notes

- Recommended speaking speed: calm and slightly slower than normal
- Best for UI recording with short pauses for button clicks
- If the recording still feels long, shorten the Catalog and Agent descriptions to one sentence each
