import os
import glob
import urllib.request
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# Force CPU mode for deployment if GPU is not available
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat as tf
tf.disable_v2_behavior()
import keras
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
class SingleDetectorInferenceConfig(Config):
    NAME = "point_source"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + point_source
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

config = SingleDetectorInferenceConfig()
MODEL_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Model Weight Management
# ---------------------------------------------------------
WEIGHTS_PATH = "mask_rcnn_coco.h5"

def ensure_weights():
    # Check if a custom trained weight exists in logs
    custom_weights = sorted(glob.glob("logs/**/*.h5", recursive=True))
    if custom_weights:
        print(f"[INFO] Using trained custom weights: {custom_weights[-1]}")
        return custom_weights[-1]

    # Fallback to base COCO weights
    if not os.path.exists(WEIGHTS_PATH):
        print("[INFO] Downloading pre-trained base weights...")
        url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
        urllib.request.urlretrieve(url, WEIGHTS_PATH)
        print("[INFO] Downloaded mask_rcnn_coco.h5 successfully.")
    return WEIGHTS_PATH

# Initialize Mask R-CNN model
print("[INFO] Loading Mask R-CNN Model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
active_weights = ensure_weights()
model.load_weights(active_weights, by_name=True)
print(f"[INFO] Model loaded successfully with weights: {active_weights}")

# ---------------------------------------------------------
# Inference Logic
# ---------------------------------------------------------
def detect_methane(input_file, confidence_threshold):
    if input_file is None:
        return None, None, "Please upload an image or .npy tile."

    file_path = input_file.name if hasattr(input_file, 'name') else str(input_file)
    
    # Load input based on extension
    if file_path.endswith('.npy'):
        img_array = np.load(file_path)
    else:
        # Standard image (PNG/JPG)
        img_bgr = cv2.imread(file_path)
        img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalize dimensions if needed
    if len(img_array.shape) == 2:
        img_array = np.dstack([img_array, img_array, img_array])
    elif img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]

    # Resize image to model resolution
    h, w, _ = img_array.shape
    resized_img = cv2.resize(img_array.astype(np.uint8), (1024, 1024))
    
    # Run Mask R-CNN detection
    results = model.detect([resized_img], verbose=0)[0]
    rois = results["rois"]
    masks = results["masks"]
    scores = results["scores"]
    class_ids = results["class_ids"]

    # Filter detections by user-selected confidence threshold
    valid_indices = [idx for idx, s in enumerate(scores) if s >= confidence_threshold]
    num_plumes = len(valid_indices)

    # Create visualization overlays
    overlay_img = resized_img.copy()
    binary_mask = np.zeros((1024, 1024), dtype=np.uint8)

    detection_logs = []
    detection_logs.append(f"### Detection Results")
    detection_logs.append(f"- **Plumes Detected**: {num_plumes}")
    detection_logs.append(f"- **Confidence Threshold**: {confidence_threshold:.2f}")

    for rank, idx in enumerate(valid_indices, 1):
        m = masks[:, :, idx]
        score = scores[idx]
        y1, x1, y2, x2 = rois[idx]
        
        # Accumulate mask
        binary_mask[m > 0] = 255

        # Red mask overlay on image
        color_mask = np.zeros_like(overlay_img)
        color_mask[m > 0] = [255, 30, 30]
        overlay_img = np.where(m[:, :, None] > 0, cv2.addWeighted(overlay_img, 0.6, color_mask, 0.4, 0), overlay_img)

        # Draw bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f"Methane Plume #{rank}: {score * 100:.1f}%"
        cv2.putText(overlay_img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        plume_area_px = int(np.sum(m > 0))
        detection_logs.append(f"- **Plume #{rank}**: Confidence = `{score * 100:.2f}%`, Area = `{plume_area_px} px`, BBox = `[{y1}, {x1}, {y2}, {x2}]`")

    if num_plumes == 0:
        detection_logs.append("*No methane plumes detected above the current confidence threshold.*")

    # Convert binary mask to RGB for display
    mask_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

    return overlay_img, mask_vis, "\n".join(detection_logs)

# ---------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------
title = "🛰️ Overhead Hyperspectral Methane Plume Detection (H-MRCNN)"
description = """
**AI Model**: Hyper-Spectral Mask R-CNN (H-MRCNN) for autonomous Methane (CH₄) plume segmentation and localization.  
Based on WACV 2020: *Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery*.

Upload a hyperspectral tile image or sample to detect and segment methane emission plumes.
"""

with gr.Blocks(title="Methane Detection H-MRCNN") as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Image (PNG/JPG) or .npy Tile", file_types=[".png", ".jpg", ".jpeg", ".npy"])
            conf_slider = gr.Slider(minimum=0.3, maximum=0.99, value=0.7, step=0.05, label="Detection Confidence Threshold")
            submit_btn = gr.Button("🔍 Run Methane Detection", variant="primary")

        with gr.Column(scale=1):
            output_overlay = gr.Image(label="Overlay Detection & Bounding Boxes", type="numpy")
            output_mask = gr.Image(label="Extracted Binary Plume Mask", type="numpy")
            results_markdown = gr.Markdown()

    submit_btn.click(
        fn=detect_methane,
        inputs=[input_file, conf_slider],
        outputs=[output_overlay, output_mask, results_markdown]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
