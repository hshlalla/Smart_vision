# Smart Vision Front

Mobile-friendly web UI (richer replacement for the Gradio demo).

## Setup

```bash
cd front
npm install
```

Create `front/.env` (or export env vars):

```bash
VITE_API_BASE_URL=http://localhost:8000
```

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

Configure API credentials via env vars in `smart-vision-api`:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
```
