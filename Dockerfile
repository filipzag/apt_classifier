# Vestige — API + React front end in one image.
#
# Build:  docker compose build app
# Run:    docker compose up        (starts Chroma and the app; ingest once with
#         `docker compose run --rm app python scripts/ingest.py` or `make ingest` on the host)
#
# Stage 1 builds the web app with Node; stage 2 is the CPU-only Python runtime that serves the
# API and the built bundle from one process. GPU hosts should run from the .venv instead.
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /srv/app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=app:app . .
COPY --from=web --chown=app:app /web/dist ./web/dist
RUN mkdir -p /data && chown -R app:app /data
USER app
VOLUME ["/data"]
EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8100/api/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8100"]
