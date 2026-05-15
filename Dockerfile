FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AGENT_MAX_STEPS=35
ENV SUBMISSION_MAX_WORKERS=4
ENV SUBMISSION_EASY_MAX_WORKERS=4
ENV SUBMISSION_MEDIUM_MAX_WORKERS=4
ENV SUBMISSION_HARD_MAX_WORKERS=1
ENV SUBMISSION_TASK_TIMEOUT_SECONDS=1800

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
