# Cloud Run image: backend plus the explorer page at /edx. Secrets come from env vars, never .env.
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend frontend
COPY dist/next-token-data.json dist/next-token-data.json
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
