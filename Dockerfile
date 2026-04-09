# ── Build stage ──────────────────────────────────────────────────────────────
# Force rebuild: 2026-04-09
FROM python:3.11-slim

# Metadata
LABEL maintainer="business-sim-env"
LABEL description="Multi-agent business simulation OpenEnv environment"

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure src is importable
ENV PYTHONPATH=/app

# HF Spaces requires port 7860
EXPOSE 7860

# Switch to non-root for HF Spaces compatibility
USER appuser

# Start FastAPI server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
