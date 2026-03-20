# External User Test Setup

This guide is for sharing the Smart Vision prototype with external testers who are **not** on the same local network.

## What Works Already

- The web app already supports external binding via Vite.
- The API already runs on `0.0.0.0:8001`.
- CORS is already open with `*`.
- The frontend already reads the API URL from `VITE_API_BASE_URL`.

Because of this, **no code change is required** for basic external testing. You only need to:

1. Run the API locally.
2. Expose the API with a tunnel.
3. Put the API tunnel URL into `apps/web/.env`.
4. Run the web app locally.
5. Expose the web app with a tunnel.
6. Share the web tunnel URL with testers.

## Recommended Option

Use `ngrok` or `Cloudflare Tunnel`.

- `ngrok` is simpler for quick testing.
- `Cloudflare Tunnel` is also good, but setup is slightly longer.

This project guide uses `ngrok` because it is the fastest path for usability testing.

## Software Requirements

- Node.js and npm
- Python virtual environment for the API
- `ngrok`

Optional but recommended:

- A Google Form link for the usability survey
- A short task sheet for participants

## 1. Start the API

From the project root:

```bash
cd /Users/mac/project/Smart_vision/apps/api
./scripts/run_dev.sh
```

Expected local API URL:

```text
http://127.0.0.1:8001
```

## 2. Expose the API

In a new terminal:

```bash
ngrok http 8001
```

Copy the public HTTPS forwarding URL. It will look like:

```text
https://abc123.ngrok-free.app
```

This is the external API URL.

## 3. Update the Web Environment

Edit [`apps/web/.env`](/Users/mac/project/Smart_vision/apps/web/.env):

```env
VITE_API_BASE_URL=https://abc123.ngrok-free.app
```

Replace the example URL with your actual `ngrok` API URL.

## 4. Start the Web App

In a new terminal:

```bash
cd /Users/mac/project/Smart_vision/apps/web
npm install
npm run dev -- --host
```

Expected local web URL:

```text
http://127.0.0.1:5173
```

## 5. Expose the Web App

In another terminal:

```bash
ngrok http 5173
```

Copy the public HTTPS forwarding URL. It will look like:

```text
https://xyz789.ngrok-free.app
```

This is the URL you send to participants.

## 6. What to Send to Testers

Send three things together:

1. The web app link
2. The task instructions
3. The Google Form link

Example message:

> Please use the prototype at: `https://xyz789.ngrok-free.app`
>
> Complete the three test tasks.
>
> After testing, please fill in the survey: `<Google Form URL>`

## Suggested Task Flow

Use the task sheet from the usability documents:

- identify a part from an uploaded image
- review the retrieved shortlist
- revise metadata and save the result

## Important Limitations

- `ngrok` free URLs change every time you restart the tunnel.
- If you restart the API tunnel, you must update `apps/web/.env` and restart the web app.
- If the local machine sleeps, tunnels will stop.
- External testers depend on your local machine staying online.

## Recommended Testing Sequence

1. Start API
2. Start API tunnel
3. Put API tunnel URL into `apps/web/.env`
4. Start web app
5. Start web tunnel
6. Test once yourself from a phone using mobile data
7. Send the web URL to participants

## Fast Verification Checklist

- API docs open from the tunnel URL
- Web app opens from the tunnel URL
- Search page loads
- Image upload works
- Results render without CORS errors
- Google Form link opens

## If You Want More Stability

For a more stable setup than a local laptop:

- deploy the web app to Vercel or Netlify
- deploy the API to a Linux server

For quick usability testing before submission, `ngrok` is sufficient.
