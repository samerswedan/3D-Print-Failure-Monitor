# syntax=docker/dockerfile:1

# ---- Base image -------------------------------------------------------------
# Slim Python base keeps the image small while staying compatible with
# ultralytics/torch. README targets Python 3.10+.
FROM python:3.11-slim AS base

# Avoid interactive prompts and keep Python output unbuffered (good for logs).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Writable config dirs so ultralytics/matplotlib/streamlit don't try to
    # write to read-only locations when running as a non-root user.
    HOME=/app \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# ---- System dependencies ----------------------------------------------------
# libglib2.0-0  -> required by OpenCV (libgthread)
# libgomp1      -> OpenMP runtime used by torch/numpy
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----------------------------------------------------
# Install the CPU-only build of torch/torchvision first. The default PyPI
# wheels on Linux pull in CUDA and bloat the image by ~2GB; the CPU index
# keeps it lean. Remove this line (and let requirements.txt install torch)
# if you want GPU wheels.
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch/torchvision already satisfied above).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application ------------------------------------------------------------
COPY app.py main.py ./
COPY models/ ./models/

# Run as a non-root user for safety; give it ownership of the app dir so the
# app can write temporary alert snapshots and config files.
RUN useradd --create-home --home-dir /app --shell /bin/bash appuser \
    && mkdir -p /app/.ultralytics \
    && chown -R appuser:appuser /app
USER appuser

# Streamlit dashboard port.
EXPOSE 8501

# Basic healthcheck against Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else sys.exit(1)" || exit 1

# Launch the web dashboard, bound to all interfaces so it's reachable from the
# host. Pass DISCORD_WEBHOOK_URL at runtime (-e or --env-file).
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
