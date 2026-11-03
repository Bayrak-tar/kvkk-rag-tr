FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY eval ./eval
RUN python -m pip install --no-cache-dir ".[retrieval,web]"

COPY configs ./configs
COPY docs ./docs

EXPOSE 8001
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
