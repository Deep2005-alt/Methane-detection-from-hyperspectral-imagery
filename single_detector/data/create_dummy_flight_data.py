#!/usr/bin/env python3
"""
Generates synthetic sample hyperspectral flight data for testing
the preprocessing and Mask R-CNN training pipeline end-to-end.
"""
import os
import numpy as np
import cv2

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHT_NAME = "ang_test_flight_cmf_v1f"
FLIGHT_DIR = os.path.join(DATA_DIR, FLIGHT_NAME)
os.makedirs(FLIGHT_DIR, exist_ok=True)

# Create 512 x 512 test scene with 4 channels: R, G, B, Methane Radiance
height, width = 512, 512
channels = 4

# Synthetic terrain (bands 0, 1, 2) and methane radiance (band 3)
terrain = (np.random.rand(height, width, 3) * 200).astype(np.float32)
radiance = np.zeros((height, width, 1), dtype=np.float32)

# Insert synthetic methane plume hotspots in the center
cv2.circle(radiance, (200, 200), 40, (1500.0,), -1)
cv2.circle(radiance, (350, 300), 25, (2200.0,), -1)

# Combined 4-channel hyperspectral array (ENVI format: row, col, band)
cube = np.concatenate([terrain, radiance], axis=2).astype(np.float32)

# Write raw ENVI binary
img_path = os.path.join(FLIGHT_DIR, f"{FLIGHT_NAME}_img")
cube.tofile(img_path)

# Write ENVI header (.hdr)
hdr_path = os.path.join(FLIGHT_DIR, f"{FLIGHT_NAME}_img.hdr")
hdr_content = f"""ENVI
description = {{Synthetic Test Hyperspectral Flight Data}}
samples = {width}
lines   = {height}
bands   = {channels}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 0
wavelength units = nm
"""
with open(hdr_path, "w") as f:
    f.write(hdr_content)

# Write Ground Truth Mask PNG (3-channel BGR):
# channel 2 = red (point source > 200), channel 0 = blue (diffused source)
mask_png = np.zeros((height, width, 3), dtype=np.uint8)
cv2.circle(mask_png, (200, 200), 40, (0, 0, 255), -1)   # Red: point source
cv2.circle(mask_png, (350, 300), 25, (0, 0, 255), -1)   # Red: point source

mask_path = os.path.join(FLIGHT_DIR, f"{FLIGHT_NAME}_img_mask.png")
cv2.imwrite(mask_path, mask_png)

print(f"[SUCCESS] Synthetic flight data created at: {FLIGHT_DIR}")
print(f" - {img_path}")
print(f" - {hdr_path}")
print(f" - {mask_path}")
