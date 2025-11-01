# Use official Python base image
FROM python:3.12-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (useful for spaCy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy deps first
COPY requirements.txt .

# Install deps + spaCy model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

# Copy application code
COPY ./app ./app
COPY ./helper_lib ./helper_lib
COPY ./models ./models

# GAN code + weights (so /gan/sample works immediately)
COPY ./gan ./gan
COPY ./artifacts ./artifacts

# Expose API port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
