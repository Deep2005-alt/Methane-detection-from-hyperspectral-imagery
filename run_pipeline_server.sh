#!/usr/bin/env bash
# ==============================================================================
# Methane Detection from Hyperspectral Imagery (H-MRCNN)
# Server / Colab / Kaggle Execution Script
# ==============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Working directory: ${REPO_DIR}"

# 1. Download MS-COCO Pretrained Weights if missing
COCO_WEIGHTS_DIR="${REPO_DIR}/single_detector/src/custom-mask-rcnn-detector"
COCO_WEIGHTS="${COCO_WEIGHTS_DIR}/mask_rcnn_coco.h5"

if [ ! -f "${COCO_WEIGHTS}" ]; then
    echo "[INFO] Downloading mask_rcnn_coco.h5..."
    wget -q --show-progress -O "${COCO_WEIGHTS}" "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
else
    echo "[INFO] Found ${COCO_WEIGHTS}"
fi

# 2. Add custom mask_rcnn to PYTHONPATH
export PYTHONPATH="${REPO_DIR}/single_detector/src/custom-mask-rcnn-detector/mask_rcnn:${PYTHONPATH}"
echo "[INFO] PYTHONPATH set with custom mask_rcnn."

# 3. Check arguments: default to full pipeline (preprocess + train)
TASK="${1:-full}"

if [ "${TASK}" == "preprocess" ] || [ "${TASK}" == "full" ]; then
    echo "=========================================="
    echo "[STAGE 1] Running Data Preprocessing..."
    echo "=========================================="
    cd "${REPO_DIR}/single_detector"
    chmod +x data_generator.sh
    ./data_generator.sh "${REPO_DIR}/single_detector/data"
fi

if [ "${TASK}" == "train" ] || [ "${TASK}" == "full" ]; then
    echo "=========================================="
    echo "[STAGE 2] Running Mask R-CNN Training..."
    echo "=========================================="
    cd "${REPO_DIR}/single_detector/src/custom-mask-rcnn-detector"
    python detector.py --mode train
fi

if [ "${TASK}" == "predict" ]; then
    IMAGE_PATH="$2"
    if [ -z "${IMAGE_PATH}" ]; then
        echo "[ERROR] Please specify image path: ./run_pipeline_server.sh predict <path_to_tile.npy>"
        exit 1
    fi
    echo "=========================================="
    echo "[STAGE 3] Running Inference on ${IMAGE_PATH}..."
    echo "=========================================="
    cd "${REPO_DIR}/single_detector/src/custom-mask-rcnn-detector"
    python detector.py --mode predict --image "${IMAGE_PATH}"
fi

echo "[INFO] Pipeline script completed successfully!"
