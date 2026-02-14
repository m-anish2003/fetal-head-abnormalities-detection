# Fetal Head Abnormalities Detection

A production-oriented fetal ultrasound project with:
- segmentation model training/evaluation,
- benchmark tooling for Dice/IoU optimization,
- and a professional web interface for inference.

## 1) Installation

### Standard
```bash
pip install -e .
```

### Constrained/offline/proxy fallback
```bash
pip install -r requirements.txt
```

### If editable install fails due build isolation
```bash
pip install --no-build-isolation -e .
```

---

## 2) Data Preparation

```bash
python scripts/organize_dataset.py \
  --raw-train raw_dataset/training_data \
  --raw-test raw_dataset/test_data \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --test-images data/test/images
```

```bash
python scripts/check_alignment.py --images data/train/images --masks data/train/masks
```

---

## 3) Training (optimized)

### Auto split (patient-aware by default)
```bash
python -m fhad.train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-split 0.2 \
  --loss bce_dice \
  --output-dir artifacts
```

### Explicit val/test split (recommended final reporting)
```bash
python -m fhad.train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --test-images data/test/images \
  --test-masks data/test/masks \
  --loss bce_dice \
  --output-dir artifacts
```

Outputs:
- `artifacts/best_model.pt`
- `artifacts/metrics.json` (history + held-out test metrics when test data is provided)

---

## 4) Benchmark baseline vs optimized

```bash
python -m fhad.train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --epochs 10 \
  --benchmark \
  --output-dir artifacts/benchmark
```

Produces `artifacts/benchmark/benchmark.json` with:
- `baseline_bce_noaug`
- `bce_dice_aug`

---

## 5) Evaluate a checkpoint

```bash
python -m fhad.evaluate \
  --images data/test/images \
  --masks data/test/masks \
  --checkpoint artifacts/best_model.pt
```

---

## 6) CLI Inference

```bash
python -m fhad.predict \
  --input data/test/images \
  --checkpoint artifacts/best_model.pt \
  --output-dir predictions
```

Outputs:
- `<name>_mask.png`
- `predictions_summary.json` (mean probability, foreground ratio, risk label)

---

## 7) Professional Web App

### Run backend (FastAPI)
```bash
FHAD_CHECKPOINT=artifacts/best_model.pt PORT=8000 ./scripts/run_web.sh
```

Then open:
- UI: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

### What web app supports
- Upload ultrasound image from browser.
- Run model inference using saved checkpoint.
- Return probability summary and risk label.
- Ready for container/API deployment via `uvicorn`.

> Important: Risk label is a research-support heuristic and **not** clinical diagnosis.

---


## 8) Docker (final production run)

Build and run with Docker Compose:
```bash
docker compose up --build
```

Then open:
- UI: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

Model path in container is controlled by:
- `FHAD_CHECKPOINT` (default: `artifacts/best_model.pt`)

Detailed hosting guide (GitHub + cloud options):
- `docs/DEPLOYMENT.md`

---
## 9) Quality gates

```bash
python -m compileall src scripts tests
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

CI (`.github/workflows/ci.yml`) runs compile + tests on push/PR.
