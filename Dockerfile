# MedAssist AI backend image.
# Python 3.12 is used here because the pinned async PostgreSQL driver (asyncpg)
# has prebuilt wheels for 3.12, while the local dev interpreter is 3.14.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY sample_documents ./sample_documents
COPY pyproject.toml ./

# Non-root runtime user; the data volume must be writable by this user.
RUN useradd --create-home --uid 10001 medassist \
    && mkdir -p /app/data \
    && chown -R medassist:medassist /app
USER medassist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
