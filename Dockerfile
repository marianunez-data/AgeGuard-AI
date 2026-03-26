# ══════════════════════════════════════════════════════════════════
#  AgeGuard AI — Production Docker Image
#  Serves ONNX model via FastAPI endpoint
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir \
    onnxruntime \
    opencv-python-headless \
    numpy \
    Pillow \
    fastapi \
    uvicorn \
    pydantic \
    PyYAML

# Copy source code and model
COPY src/ ./src/
COPY models/onnx/ageguard_v1.onnx ./models/onnx/
COPY configs/base_config.yaml ./configs/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
