from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from ..data import VALID_EXTENSIONS
from ..inference import FetalHeadInferenceService


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def _load_service() -> FetalHeadInferenceService:
    checkpoint = os.getenv("FHAD_CHECKPOINT", "artifacts/best_model.pt")
    image_size = int(os.getenv("FHAD_IMAGE_SIZE", "256"))
    threshold = float(os.getenv("FHAD_THRESHOLD", "0.5"))

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    return FetalHeadInferenceService(checkpoint_path, image_size=image_size, threshold=threshold)


def create_app() -> FastAPI:
    app = FastAPI(title="Fetal Head Abnormality Detection", version="1.1.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.state.inference_service = None
    app.state.startup_error = None

    @app.on_event("startup")
    async def startup() -> None:
        try:
            app.state.inference_service = _load_service()
            app.state.startup_error = None
        except Exception as exc:
            app.state.inference_service = None
            app.state.startup_error = str(exc)

    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok" if app.state.inference_service else "degraded",
            "model_loaded": bool(app.state.inference_service),
            "error": app.state.startup_error or "",
        }

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        service = app.state.inference_service
        if service is None:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {app.state.startup_error or 'not loaded'}")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix and suffix not in VALID_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            try:
                Image.open(io.BytesIO(content)).verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

            pred = service.predict_path(tmp_path)
            return JSONResponse(
                {
                    "filename": file.filename,
                    "mean_probability": pred.mean_probability,
                    "foreground_ratio": pred.foreground_ratio,
                    "risk_label": pred.risk_label,
                    "warning": "Research support only. Not a clinical diagnosis.",
                }
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return app


app = create_app()
