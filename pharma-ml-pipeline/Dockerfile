# Use NVIDIA's TensorFlow image as base
FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RDKit with conda
RUN pip install rdkit

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-service:5000

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Set up entry point for different modes
RUN chmod +x ./scripts/entrypoint.sh
ENTRYPOINT ["./scripts/entrypoint.sh"]

# Default command (can be overridden)
CMD ["streamlit"]
