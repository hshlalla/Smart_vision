# Appendix C-D Caption Drafts (EN)

This working file provides caption-ready draft text for `Appendix C. User Interface Iterations` and `Appendix D. Qualitative Error Analysis`.

## Appendix C. User Interface Iterations

### Figure C1. Login page of the final web prototype

Draft caption:

> **Figure C1.** Final login page of the Smart Vision web prototype. This screen provides the authenticated entry point to the retrieval and indexing workflow used throughout the final user study and video demonstration.

Source:
- manual capture needed

### Figure C2. Search page for image-assisted retrieval

Draft caption:

> **Figure C2.** Final search interface supporting image-assisted retrieval. The page allows the user to submit a query image and optional text, inspect the returned shortlist, and review evidence-backed candidate matches.

Source:
- `docs/images/fig_4_2_web_search_ui.png`

### Figure C3. Indexing page with metadata preview and confirmation workflow

Draft caption:

> **Figure C3.** Indexing interface showing the preview-before-confirm workflow. Uploaded images are used to draft listing metadata, after which the user can review, edit, and confirm the final item record.

Source:
- `docs/images/fig_index_ui.png`

### Figure C4. Catalog page for document-grounded lookup

Draft caption:

> **Figure C4.** Catalog interface used to retrieve supporting information from indexed documentation. This page extends the core retrieval workflow by allowing the user to consult internal catalogue-style evidence.

Source:
- `docs/images/fig_catalog_ui.png`

### Figure C5. Agent page combining multiple evidence sources

Draft caption:

> **Figure C5.** Agent interface used for conversational evidence gathering. The page illustrates how hybrid retrieval, catalogue lookup, and supporting reasoning can be combined in a single user-facing workflow.

Source:
- `docs/images/fig_4_4_agent_chat_ui.png`

### Figure C6. Bilingual language-toggle interface

Draft caption:

> **Figure C6.** Bilingual interface example showing the language-toggle support added to the final web prototype. This feature improves accessibility and reflects the practical deployment orientation of the system.

Source:
- manual capture needed

### Figure C7. Early prototype or Gradio-based interface

Draft caption:

> **Figure C7.** Earlier prototype interface retained for comparison. This figure is included as an early concept to show how the project progressed from a simpler prototype interaction model toward the final structured web workflow.

Source:
- use an early Gradio screenshot if available in the draft materials

## Appendix D. Qualitative Error Analysis

### Figure D1. Irrelevant specification noise extracted by OCR

Draft caption:

> **Figure D1.** Example of irrelevant specification noise in industrial imagery. OCR extracts visible strings such as voltage or resistance values, but these do not uniquely identify the part and can instead distort retrieval ranking.

Source:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`
- manual failure-case screenshot

### Figure D2. Mixed vertical and horizontal text causing OCR instability

Draft caption:

> **Figure D2.** Mixed-orientation identifier text. This type of layout frequently causes OCR token fragmentation or ordering errors, whereas the vision-language path remains more robust because it reasons over the full label layout.

Source:
- manual failure-case screenshot

### Figure D3. Engraved or low-contrast label case

Draft caption:

> **Figure D3.** Engraved or low-contrast identifier region. Traditional OCR is brittle under weak contrast and partial wear, which makes this class of example useful for explaining why OCR was demoted from the default retrieval path.

Source:
- manual failure-case screenshot

### Figure D4. Logo-like manufacturer mark not reliably recovered as text

Draft caption:

> **Figure D4.** Logo-like manufacturer mark that functions as visual evidence rather than clean text. This type of case illustrates the benefit of the vision-language pathway over a purely text-extraction-based approach.

Source:
- manual failure-case screenshot

### Figure D5. Wrong Top-1 but correct Top-5 retrieval example

Draft caption:

> **Figure D5.** Example where the system misses the exact item at rank 1 but still returns the correct match within the shortlist. Such cases support the human-in-the-loop design, because they show why a ranked candidate list is more realistic than a single forced prediction.

Source:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/CURRENT_EXPERIMENT_STATUS.md`
- manual result screenshot

### Figure D6. Blur, glare, occlusion, or low-resolution input case

Draft caption:

> **Figure D6.** Difficult query image with blur, glare, occlusion, or low resolution. This figure is useful for showing the remaining boundary conditions of the prototype even after the shift to a vision-dominant retrieval pipeline.

Source:
- manual failure-case screenshot

## Fast Placement Priority

If time is limited, prioritise the following:

1. `Figure C2`
2. `Figure C3`
3. `Figure C4`
4. `Figure C5`
5. `Figure D1`
6. `Figure D2`
7. `Figure D5`

## Example Main-Body Cross-References

- “Additional interface screenshots are provided in Appendix C (Figures C1-C7).”
- “Representative OCR and retrieval failure cases are shown in Appendix D (Figures D1-D6).”
