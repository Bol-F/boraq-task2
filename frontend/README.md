# Churn risk frontend

This directory contains the deployable Next.js interface for the Telecom
Customer Churn API. The browser submits all 19 customer fields to same-origin
Next.js Route Handlers. Those server-only handlers call the Django API, so the
Render hostname is not exposed to browser code and browser CORS configuration
is not required.

## Local development

Start Django at `http://127.0.0.1:8000`, then run:

```bash
cp .env.example .env.local
npm install
npm run dev
```

PowerShell users can replace `cp` with `Copy-Item`. Open
`http://localhost:3000`.

The only required setting is:

```dotenv
RENDER_API_URL=http://127.0.0.1:8000
```

Use a base URL with no API endpoint suffix. Keep this variable server-only; do
not rename it to `NEXT_PUBLIC_RENDER_API_URL`.

## Verification

Run every frontend gate with:

```bash
npm run verify
```

Or run the checks individually:

```bash
npm run lint
npm run test
npm run typecheck
npm run build
```

## Vercel deployment

Import the repository into Vercel, set the root directory to `frontend`, and
configure `RENDER_API_URL` with the verified HTTPS base URL of the Render API
for Preview and Production. The committed `vercel.json` runs the production
build; GitHub Actions runs the complete lint, test, type-check, and build gate.
Redeploy after changing environment variables.

Do not add a deployment URL to project documentation until the health proxy,
prediction proxy, browser form, and real model response have all been verified.
