# MedAssist AI — Frontend

React + TypeScript + Vite + Tailwind CSS + shadcn-style UI for the MedAssist AI backend.

## Prerequisites
- Node.js 20+
- The backend running on `http://localhost:8000` (see the root README).

## Install & run
```powershell
cd C:\Users\ayman\MedAssist-AI\frontend
npm install
npm run dev
```
Open http://localhost:5173. Vite proxies `/api` to `http://localhost:8000`.

## Build
```powershell
npm run build
npm run preview
```

## Configuration
Set `VITE_API_BASE_URL` (e.g. in `.env`) to point at a remote backend; otherwise the Vite proxy is used.

## Notes
- Pages that depend on not-yet-implemented backend endpoints (Agentic RAG, Benchmark) show honest empty states and never fabricate data.
- Medical disclaimer is shown persistently in the app shell.
