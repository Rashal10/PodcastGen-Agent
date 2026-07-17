FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PODCAST_OUTPUT_DIR=/app/outputs \
    COQUI_TOS_AGREED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml README.md ./
COPY podcast_gen_agent ./podcast_gen_agent

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

RUN mkdir -p /app/outputs

EXPOSE 8000

CMD ["uvicorn", "podcast_gen_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
