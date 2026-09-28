---
title: Methane Detection From Hyperspectral Imagery (H-MRCNN)
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Overhead Hyperspectral Methane Detection (H-MRCNN)

Web deployment demo of **Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery** (WACV 2020).

## Features
- **Interactive UI**: Upload hyperspectral imagery or `.npy` tiles to detect and segment methane plumes.
- **Instance Segmentation**: Visualizes plume boundaries, bounding boxes, and emission hotspots using Mask R-CNN.
- **Plume Statistics**: Computes pixel surface area and detection confidence scores.

## How to Deploy to Hugging Face Spaces (100% Free)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space Name (e.g. `methane-detection-hmrcnn`).
3. Select License: **MIT**.
4. Select Space SDK: **Docker** -> **Blank**.
5. Choose Free Hardware: **CPU Basic (2 vCPU · 16 GB RAM)** - *Free forever*.
6. Push or upload all files from this directory to your Hugging Face Space repository:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
   git add .
   git commit -m "Deploy Methane Detection App"
   git push -u origin main
   ```
7. Hugging Face will automatically build your Docker container and give you a public URL (e.g. `https://<YOUR_USERNAME>-<SPACE_NAME>.hf.space`).
