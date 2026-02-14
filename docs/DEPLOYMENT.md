# Deployment Guide (Docker + GitHub)

## A) Run locally with Docker

1. Ensure you have a trained checkpoint at:
   - `artifacts/best_model.pt`

2. Build and run:
```bash
docker compose up --build
```

3. Open:
- UI: http://localhost:8000
- Health: http://localhost:8000/api/health
- Docs: http://localhost:8000/docs

## B) Push to GitHub

```bash
git remote add origin <your-repo-url>
git push -u origin <branch>
```

## C) Deploy from GitHub (recommended paths)

### Option 1: Render (easy)
- Create a new **Web Service** from your GitHub repo.
- Environment:
  - Build command: `docker build -t fhad .`
  - Start command: handled by container CMD
- Add persistent disk / mount if you need to keep model artifacts.
- Set env var:
  - `FHAD_CHECKPOINT=artifacts/best_model.pt`

### Option 2: Railway / Fly.io
- Use Docker deployment from GitHub.
- Expose port `8000`.
- Set same env vars as above.

### Option 3: VPS
```bash
git clone <repo>
cd <repo>
docker compose up -d --build
```

## Important
- This application is a research tool, not clinical software.
- If `/api/health` shows `degraded`, the checkpoint path is wrong/missing.
