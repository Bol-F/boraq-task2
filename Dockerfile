# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.11-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project \
        --python /usr/local/bin/python


FROM python:3.11-slim AS runtime

ENV DJANGO_DEBUG=false \
    PATH="/app/.venv/bin:${PATH}" \
    PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser manage.py ./
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser ml_pipeline ./ml_pipeline
COPY --chown=appuser:appuser predictions ./predictions
COPY --chown=appuser:appuser models/model.pkl models/model_metadata.json ./models/

RUN export DJANGO_SECRET_KEY=build-only-insecure-validation-key-do-not-use \
    DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1; \
    python manage.py collectstatic --noinput \
    && python manage.py check \
    && python -c \
        "import importlib.util; assert importlib.util.find_spec('mlflow') is None" \
    && python manage.py shell -c \
        "from predictions.services.model_loader import get_model_bundle; get_model_bundle()"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; request = urllib.request.Request(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/api/health/\", headers={\"X-Forwarded-Proto\": \"https\"}); urllib.request.urlopen(request, timeout=4)"]

CMD ["sh", "-c", "exec gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --threads 2 --timeout 120 --access-logfile - --error-logfile -"]
