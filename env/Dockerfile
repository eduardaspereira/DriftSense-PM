FROM python:3.11-slim

LABEL maintainer="Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães <edp@uminho.pt>"
LABEL description="DriftSense-PM: Drift-Aware Predictive Maintenance Benchmark"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set working directory
WORKDIR /app

# Default command: run baseline training
CMD ["python", "scripts/train_baseline_full.py"]
