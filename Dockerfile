FROM python:3.12-slim

# Install system dependencies required for OpenCV, audio processing, and downloading
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
# We install PyTorch CPU version to save massive amounts of RAM and image size since most cloud deployments don't have GPUs by default
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Start the continuous monitor daemon
CMD ["python", "src/rss_monitor.py"]
