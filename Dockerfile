# ═══════════════════════════════════════════════════════════════════════
#  Dockerfile — Traffic Forecasting API
#  Builds a Docker image that runs the FastAPI backend.
#
#  Build:  docker build -t trafficiq .
#  Run:    docker run -p 8000:8000 trafficiq
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# System dependencies for osmnx / geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
