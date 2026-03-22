# Appendix C-D Caption Drafts (KO)

이 문서는 `Appendix C. User Interface Iterations`와 `Appendix D. Qualitative Error Analysis`에 바로 붙여 넣을 수 있는 캡션 초안을 정리한 작업용 문서다.

## Appendix C. User Interface Iterations

### Figure C1. Login page of the final web prototype

캡션 초안:

> **Figure C1.** Final login page of the Smart Vision web prototype. This screen provides the authenticated entry point to the retrieval and indexing workflow used throughout the final user study and video demonstration.

자료:
- 수동 캡처 필요

### Figure C2. Search page for image-assisted retrieval

캡션 초안:

> **Figure C2.** Final search interface supporting image-assisted retrieval. The page allows the user to submit a query image and optional text, inspect the returned shortlist, and review evidence-backed candidate matches.

자료:
- `docs/images/fig_4_2_web_search_ui.png`

### Figure C3. Indexing page with metadata preview and confirmation workflow

캡션 초안:

> **Figure C3.** Indexing interface showing the preview-before-confirm workflow. Uploaded images are used to draft listing metadata, after which the user can review, edit, and confirm the final item record.

자료:
- `docs/images/fig_index_ui.png`

### Figure C4. Catalog page for document-grounded lookup

캡션 초안:

> **Figure C4.** Catalog interface used to retrieve supporting information from indexed documentation. This page extends the core retrieval workflow by allowing the user to consult internal catalogue-style evidence.

자료:
- `docs/images/fig_catalog_ui.png`

### Figure C5. Agent page combining multiple evidence sources

캡션 초안:

> **Figure C5.** Agent interface used for conversational evidence gathering. The page illustrates how hybrid retrieval, catalogue lookup, and supporting reasoning can be combined in a single user-facing workflow.

자료:
- `docs/images/fig_4_4_agent_chat_ui.png`

### Figure C6. Bilingual language-toggle interface

캡션 초안:

> **Figure C6.** Bilingual interface example showing the language-toggle support added to the final web prototype. This feature improves accessibility and reflects the practical deployment orientation of the system.

자료:
- 수동 캡처 필요

### Figure C7. Early prototype or Gradio-based interface

캡션 초안:

> **Figure C7.** Earlier prototype interface retained for comparison. This figure is included as an early concept to show how the project progressed from a simpler prototype interaction model toward the final structured web workflow.

자료:
- draft에 남아 있는 Gradio 캡처가 있으면 사용

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

### Figure D1. Irrelevant specification noise extracted by OCR

캡션 초안:

> **Figure D1.** Example of irrelevant specification noise in industrial imagery. OCR extracts visible strings such as voltage or resistance values, but these do not uniquely identify the part and can instead distort retrieval ranking.

자료:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/qwen3_vl_1000_sample_final_report_en.md`
- 수동 failure screenshot 캡처

### Figure D2. Mixed vertical and horizontal text causing OCR instability

캡션 초안:

> **Figure D2.** Mixed-orientation identifier text. This type of layout frequently causes OCR token fragmentation or ordering errors, whereas the vision-language path remains more robust because it reasons over the full label layout.

자료:
- 수동 failure screenshot 캡처

### Figure D3. Engraved or low-contrast label case

캡션 초안:

> **Figure D3.** Engraved or low-contrast identifier region. Traditional OCR is brittle under weak contrast and partial wear, which makes this class of example useful for explaining why OCR was demoted from the default retrieval path.

자료:
- 수동 failure screenshot 캡처

### Figure D4. Logo-like manufacturer mark not reliably recovered as text

캡션 초안:

> **Figure D4.** Logo-like manufacturer mark that functions as visual evidence rather than clean text. This type of case illustrates the benefit of the vision-language pathway over a purely text-extraction-based approach.

자료:
- 수동 failure screenshot 캡처

### Figure D5. Wrong Top-1 but correct Top-5 retrieval example

캡션 초안:

> **Figure D5.** Example where the system misses the exact item at rank 1 but still returns the correct match within the shortlist. Such cases support the human-in-the-loop design, because they show why a ranked candidate list is more realistic than a single forced prediction.

자료:
- `experiments/qwen3_vl_1000_sample_final_report_ko.md`
- `experiments/CURRENT_EXPERIMENT_STATUS.md`
- 수동 result screenshot 캡처

### Figure D6. Blur, glare, occlusion, or low-resolution input case

캡션 초안:

> **Figure D6.** Difficult query image with blur, glare, occlusion, or low resolution. This figure is useful for showing the remaining boundary conditions of the prototype even after the shift to a vision-dominant retrieval pipeline.

자료:
- 수동 failure screenshot 캡처

## 빠른 배치 추천

시간이 부족하면 아래 순서로 먼저 넣는 것이 좋다.

1. `Figure C2`
2. `Figure C3`
3. `Figure C4`
4. `Figure C5`
5. `Figure D1`
6. `Figure D2`
7. `Figure D5`

## 본문 연결 문장 예시

- “Additional interface screenshots are provided in Appendix C (Figures C1-C7).”
- “Representative OCR and retrieval failure cases are shown in Appendix D (Figures D1-D6).”
