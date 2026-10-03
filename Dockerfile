FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gemini_web2api ./gemini_web2api

# Railway supplies PORT; default 8081 for local docker runs.
# Secrets (PROXY_API_KEY, cookie) are injected via environment/volumes — never baked in.
EXPOSE 8081
CMD ["sh", "-c", "python -m gemini_web2api --host 0.0.0.0 --port ${PORT:-8081}"]
