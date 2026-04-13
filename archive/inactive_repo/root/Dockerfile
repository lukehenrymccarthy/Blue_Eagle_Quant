# ── SectorScope — NiceGUI dashboard ──────────────────────────────────────────
FROM python:3.12-slim

# System deps (needed for pyarrow, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (data/ is volume-mounted at runtime — not baked in)
COPY . .

# Port exposed by NiceGUI
EXPOSE 8080

# Default: run the dashboard
CMD ["python", "dashboard/app.py"]
