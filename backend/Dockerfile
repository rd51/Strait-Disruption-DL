# Backend image — serves the 24/7 GDELT collector and (via an entrypoint
# override) the API. Build context is the PROJECT ROOT, not backend/.
FROM python:3.12-slim

# Fail fast and log immediately: unbuffered stdout is what makes `docker logs`
# useful for a long-running service.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the pip layer.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend

# Run unprivileged. The mounted data volume must be writable by this uid.
RUN useradd --create-home --uid 10001 collector \
    && mkdir -p /data && chown -R collector:collector /data /app
USER collector

# HORMUZ_ROOT is set EXPLICITLY because .dockerignore strips the repo markers
# (*.md, docker-compose.yml) that common/paths.py searches for. Without it the
# resolver silently falls back to counting parent directories — precisely the
# fragility that module exists to eliminate.
ENV HORMUZ_ROOT=/app \
    PYTHONPATH=/app \
    HORMUZ_DATA_ROOT=/data/raw/gdelt \
    HORMUZ_DATA_DIR=/data \
    AIS_DATA_ROOT=/data/raw/ais \
    HEALTH_HEARTBEAT=/data/raw/gdelt/_state/heartbeat \
    HEALTH_MAX_AGE=2100

# Healthy = the service touched its heartbeat recently. A hung socket or a
# wedged loop stops the heartbeat while the process lives on, which a plain
# "is the process running" check would miss entirely. Each service overrides
# HEALTH_HEARTBEAT/HEALTH_MAX_AGE for its own cadence.
HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD python -c "\
import os,sys,time; \
p=os.environ['HEALTH_HEARTBEAT']; m=float(os.environ.get('HEALTH_MAX_AGE',2100)); \
sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < m else 1)"

# Default service; ais-collector and api override this in docker-compose.yml.
ENTRYPOINT ["python", "-m", "backend.ingest.gdelt.poller"]
