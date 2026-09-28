FROM python:3.7-slim

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV, Git, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces: UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Upgrade pip to compatible version
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download pre-trained COCO weights during image build so startup is instant
RUN wget -q -O mask_rcnn_coco.h5 https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Copy application files and custom mrcnn module
COPY --chown=user:user . $HOME/app

# Ensure proper permissions
RUN chown -R user:user $HOME/app

USER user

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Start Gradio App
CMD ["python", "app.py"]
