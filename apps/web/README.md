# Smart Vision Front

Mobile-friendly web UI (richer replacement for the Gradio demo).

Screens
- Chat: `/app/chat` (agent bot)
- Search: `/app/search` (hybrid search)
- Index: `/app/index` (upload + index)
- Catalog: `/app/catalog` (PDF RAG index + search)

Current UI features
- Index
  - metadata preview mode switch: `Auto`, `GPT`, `Local`
  - optional `Label OCR` popup for label-only supporting images
  - duplicate candidate banner when the preview draft appears to match an already indexed part
  - explicit user choice between:
    - keeping the upload as a new model
    - adding the upload to the existing indexed model
  - async indexing status polling
- Search
  - image/text hybrid search
  - reranker toggle for experiments
- Catalog
  - PDF text mode or PaddleOCR-VL mode can be selected at indexing time
  - markdown-style table chunks are rendered as tables in catalog results when possible
- Chat
  - agent bot with internal search first
  - matched product image is shown when agent returns an internal identified item
  - if catalog RAG evidence is used, the answer can include document source/page citations in addition to the matched product card
  - internal inventory answers and open-world web answers can still differ depending on the question
  - simple markdown tables in answers are rendered as tables when possible

Why the duplicate banner exists
- In real use, the same part can be uploaded again later with cleaner photos, better labels, or richer metadata.
- Creating a new model every time would fragment the inventory and weaken retrieval quality.
- The Index screen therefore exposes the decision to the user instead of silently forcing either `new model` or `merge`.

## Setup

```bash
cd apps/web
npm install
```

Create `apps/web/.env` (or export env vars):

```bash
VITE_API_BASE_URL=http://127.0.0.1:8001
```

If the UI is opened from another device, replace `127.0.0.1` with the API host machine IP.

## Run (dev)

```bash
npm run dev -- --host
```

## Build

```bash
npm run build
npm run preview -- --host
```

## Login

The UI calls the API auth endpoints:

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`

Configure API credentials via env vars in `apps/api`:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
```

If auth is disabled (`AUTH_ENABLED=false`), the login screen shows a "continue" button.
