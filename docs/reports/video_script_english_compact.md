# Final Video Script (English, Compact Recording Version)

This version is a **shorter English script** intended for a recording time of roughly **3.5 to 4 minutes**, depending on speaking speed and UI pauses.

## Final Script

Hello. My project is **Smart Image Part Identifier for Secondhand Platforms**.  
This web-based system helps users identify industrial parts and prepare listing metadata from uploaded images.

I will begin at the login page.  
From here, users can access the search, indexing, catalog, and agent functions, and they can also switch between Korean and English.

Next is the indexing page.  
Users can upload one image or multiple images for the same item.  
The system first generates a metadata draft using a VLM-based path and suggests fields such as maker, category, part number, and description.  
The user reviews and edits the draft, and only after pressing **confirm** does the system save the final record.  
This preview-before-confirm flow is safer and better suited to a real listing workflow.

Now I will show the search page.  
Users can search with text only, or with both image and text together.  
The key idea is a **retrieval-first approach**.  
Instead of forcing one final answer, the system returns a shortlist of likely candidates and lets the user inspect them.

The system combines image embeddings, text signals, metadata signals, and lexical matching.  
Because of this, the results are based not only on visual similarity, but also on identifier clues such as maker names or part numbers.  
The interface also shows evidence for why a result is relevant, which improves trust and supports human decision making.

Next is the Catalog page.  
Here, users can upload PDF manuals or catalogues and turn them into searchable evidence sources.

Next is the Agent page.  
This feature combines hybrid search, catalog search, and optional external evidence to provide richer support in one workflow.

To summarize, the main contribution of this project is not just one model.  
It is the integration of search, metadata assistance, catalog retrieval, agent support, and user confirmation into one practical workflow.  
The system is best described as a **retrieval-first, human-in-the-loop identification assistant prototype**.

Thank you.

## Notes

- Recommended speaking speed: calm and slightly slower than normal
- Best for UI recording with short pauses for button clicks
- If the recording is still long, shorten the Catalog and Agent sections to one sentence each
