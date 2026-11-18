FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MTHL_MODEL_PATH=/app/artifacts/demo_model.joblib

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt ./requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts/demo_model.joblib ./artifacts/demo_model.joblib

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-build-isolation --no-deps . \
    && chown -R appuser:appuser /app

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "mthl.api:app", "--host", "0.0.0.0", "--port", "8000"]
