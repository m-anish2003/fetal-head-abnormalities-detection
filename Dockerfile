FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p artifacts && pip install --no-build-isolation -e .

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000 FHAD_CHECKPOINT=artifacts/best_model.pt

CMD ["./scripts/run_web.sh"]
