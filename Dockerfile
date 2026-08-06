FROM python:3.10-slim

# Install system dependencies for OpenCV, FFmpeg, and Git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 5000 (Hugging Face proxy target)
EXPOSE 5000

# Run Flask server
CMD ["python", "web_server.py"]
