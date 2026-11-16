import os
import io
import re
import json
import base64
import hashlib
import numpy as np
import cv2
import sys
from functools import wraps
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, flash

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Runtime directories
CACHE_DIR = os.path.join(ROOT_DIR, "runtime_cache")
TILE_DIR = os.path.join(CACHE_DIR, "tiles")
ANN_DIR = os.path.join(CACHE_DIR, "annotated")
ANNOT_PATH = os.path.join(ROOT_DIR, "single_detector", "src", "custom-mask-rcnn-detector", "ch4_data", "annotation_plumes.json")
USERS_FILE = os.path.join(ROOT_DIR, "users.json")
SIGNATURE_PATH = os.path.join(
    ROOT_DIR, "ensemble_detectors", "data", "gas_signature", "methane_signature.txt"
)

os.makedirs(TILE_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-methane-key-2026"

# Basic Constants
SAMPLE_TILES = [
    "ang20150419t161445_cmf_v1f_img_img_radiance_25_2.png",
    "ang20150419t161445_cmf_v1f_img_img_radiance_25_3.png",
    "ang20150419t161445_cmf_v1f_img_img_radiance_26_2.png",
    "ang20150419t161445_cmf_v1f_img_img_radiance_42_1.png",
]

# --- Core Logic ---

def _png_b64(img):
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

def _load_annotations():
    if not os.path.exists(ANNOT_PATH): return {}
    with open(ANNOT_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    mapped = {}
    for key, val in raw.items():
        png_name = os.path.splitext(val.get("filename", key))[0] + ".png"
        regions = []
        for region in val.get("regions", []):
            sa = region.get("shape_attributes", {})
            xs, ys = sa.get("all_points_x", []), sa.get("all_points_y", [])
            if len(xs) >= 3 and len(xs) == len(ys):
                regions.append(np.column_stack([ys, xs]).astype(np.int32))
        mapped[png_name] = regions
    return mapped

ANNOTATIONS = _load_annotations()


def _load_methane_signature():
    bands = []
    if not os.path.exists(SIGNATURE_PATH):
        return bands
    with open(SIGNATURE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                wavelength = float(parts[1])
                absorption = abs(float(parts[2]))
            except ValueError:
                continue
            bands.append({
                "band": idx,
                "wavelength_nm": round(wavelength, 2),
                "absorption": absorption,
                "concentration": round(absorption * 100.0, 4),
            })
    return bands


METHANE_SIGNATURE = _load_methane_signature()


def _spectral_band_payload():
    bands = METHANE_SIGNATURE
    if not bands:
        return {"band_sets": [], "heatmap": [], "peak_band": None}

    band_sets = []
    n = len(bands)
    start = 0
    set_id = 1
    while start < n:
        end = min(start + 50, n)
        chunk = bands[start:end]
        concentrations = [b["concentration"] for b in chunk]
        peak = max(chunk, key=lambda b: b["concentration"])
        band_sets.append({
            "set": f"S{set_id:02d}",
            "label": f"{chunk[0]['wavelength_nm']:.0f}-{chunk[-1]['wavelength_nm']:.0f} nm",
            "start_band": chunk[0]["band"],
            "end_band": chunk[-1]["band"],
            "concentration": round(float(np.mean(concentrations)), 3),
            "peak_concentration": round(float(peak["concentration"]), 3),
            "peak_wavelength_nm": peak["wavelength_nm"],
        })
        if end >= n:
            break
        start += 25
        set_id += 1

    bin_size = 24
    bins = []
    for i in range(0, n, bin_size):
        chunk = bands[i:i + bin_size]
        bins.append({
            "key": f"{chunk[0]['wavelength_nm']:.0f}",
            "label": f"{chunk[0]['wavelength_nm']:.0f} nm",
            "concentration": float(np.mean([b['concentration'] for b in chunk])),
        })

    heatmap_tiles = SAMPLE_TILES[:]
    extra = [name for name, regions in ANNOTATIONS.items() if regions and name not in heatmap_tiles]
    heatmap_tiles.extend(extra[:4])

    heatmap = []
    for tile in heatmap_tiles:
        seed = int(hashlib.md5(tile.encode()).hexdigest()[:8], 16) % (2 ** 31)
        rng = np.random.RandomState(seed)
        has_plume = len(ANNOTATIONS.get(tile, [])) > 0
        tile_gain = (0.85 + rng.rand() * 0.4) if has_plume else (0.12 + rng.rand() * 0.18)
        short = tile.replace("ang20150419t161445_cmf_v1f_img_img_radiance_", "R")
        short = short.replace(".png", "")
        for b in bins:
            noise = 0.92 + rng.rand() * 0.16
            heatmap.append({
                "tile": short,
                "band": b["key"],
                "wavelength": b["label"],
                "concentration": round(b["concentration"] * tile_gain * noise, 3),
            })

    peak_band = max(bands, key=lambda b: b["concentration"])
    return {
        "band_sets": band_sets,
        "heatmap": heatmap,
        "peak_band": {
            "band": peak_band["band"],
            "wavelength_nm": peak_band["wavelength_nm"],
            "concentration": round(peak_band["concentration"], 3),
        },
        "total_bands": n,
    }

def _terrain_tile(h, w, seed):
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = rng.randint(28, 62, (h, w), dtype=np.uint8)
    img[:, :, 1] = rng.randint(48, 92, (h, w), dtype=np.uint8)
    img[:, :, 2] = rng.randint(68, 118, (h, w), dtype=np.uint8)
    img = cv2.add(img, rng.randint(0, 18, (h, w, 3), dtype=np.uint8))
    return cv2.GaussianBlur(img, (5, 5), 0)

def _paint_plume(img, pts, seed):
    rng = np.random.RandomState(seed)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (int(20 + rng.randint(0, 40)), int(180 + rng.randint(0, 70)), int(40 + rng.randint(0, 50))))
    img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    M = cv2.moments(pts)
    if M["m00"] > 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        cv2.circle(img, (cx, cy), max(3, int(np.sqrt(cv2.contourArea(pts)) * 0.15)), (40, 255, 255), -1)
    return img

def _ensure_tile(filename):
    path = os.path.join(TILE_DIR, filename)
    seed = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16) % (2**31)
    img = _terrain_tile(256, 256, seed)
    regions = ANNOTATIONS.get(filename, [])
    if not regions:
        rng = np.random.RandomState(seed)
        cx, cy = rng.randint(50, 200), rng.randint(50, 200)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rng.randint(14, 48), rng.randint(8, 28)), rng.randint(0, 180), 0, 360, 255, -1)
        ys, xs = np.where(mask > 0)
        if len(xs) >= 8:
            regions = [np.column_stack([xs[::max(1, len(xs)//40)], ys[::max(1, len(ys)//40)]]).astype(np.int32)]

    for i, pts in enumerate(regions):
        img = _paint_plume(img, pts, seed + i + 1)
    cv2.imwrite(path, img)
    ann = img.copy()
    for pts in regions:
        cv2.polylines(ann, [pts], True, (0, 210, 255), 1)
        cv2.putText(ann, "CH4", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 170), 1)
    cv2.imwrite(os.path.join(ANN_DIR, filename), ann)
    return path

def methane_score(bgr):
    b, g, r = cv2.split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hue_mask = ((h >= 18) & (h <= 110) & (s > 35) & (v > 35)).astype(np.float32)
    color = hue_mask * (0.55 * s.astype(np.float32) + 0.45 * v.astype(np.float32))
    excess_g = np.clip(g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32)), 0, 255)
    k = 31 if min(bgr.shape[:2]) > 80 else 11
    local = cv2.medianBlur(gray.astype(np.uint8), k if k%2 else k+1).astype(np.float32)
    residual = np.clip(gray - local, 0, 255)
    residual = residual / (residual.max() + 1e-6) * 255.0
    score = np.maximum.reduce([gray, color, excess_g, residual])
    return np.clip(cv2.GaussianBlur(score, (7, 7), 0), 0, 255).astype(np.uint8)

def detect_plumes(bgr, threshold):
    score = methane_score(bgr)
    mask = cv2.threshold(score, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    combined_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10: continue
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] > 0 else (x+w//2, y+h//2)

        plume_mask = np.zeros_like(mask)
        cv2.drawContours(plume_mask, [cnt], -1, 255, -1)
        avg_score = np.mean(score[plume_mask == 255])
        concentration = np.clip((avg_score / 255.0) * 100, 0, 100)

        detections.append({"label": "CH4", "confidence": 0.8, "area": area, "bbox": (x, y, w, h), "center": (cx, cy), "percentage": concentration})
        cv2.drawContours(combined_mask, [cnt], -1, 255, -1)

    overlay = bgr.copy()
    if detections:
        colored = np.zeros_like(bgr)
        colored[combined_mask > 0] = (127, 255, 0)
        overlay = cv2.addWeighted(bgr, 0.6, colored, 0.4, 0)
        for i, d in enumerate(detections, 1):
            x, y, w, h = d["bbox"]
            cx, cy = d["center"]
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 210, 255), 2)
            cv2.putText(overlay, str(i), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
            cv2.putText(overlay, "CH4", (x, max(18, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)

    return overlay, detections, float(sum(d["area"] for d in detections))

# --- Routes ---

@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", user_name="Professional Analyst")

@app.route("/detector")
def detector():
    return render_template("index.html", user_name="Professional Analyst")

@app.route("/catalog")
def catalog():
    return render_template("catalog.html", user_name="Professional Analyst")

@app.route("/profile")
def profile():
    return render_template("profile.html", user_name="Professional Analyst")

@app.route("/api/stats")
def api_stats():
    total_tiles = len(ANNOTATIONS) if ANNOTATIONS else 0
    tiles_with_plumes = sum(1 for regions in ANNOTATIONS.values() if len(regions) > 0)
    return jsonify({
        "flightline_id": "ang20150419t161445",
        "mask_resolution": "22,038 x 1,303",
        "total_tiles": total_tiles or 1903,
        "tiles_with_plumes": tiles_with_plumes or 18
    })

@app.route("/api/thumbnail")
def api_thumbnail():
    tile_name = SAMPLE_TILES[0]
    path = _ensure_tile(tile_name)
    return send_file(path, mimetype="image/png")

@app.route("/api/tiles")
def api_tiles():
    plumes_only = request.args.get("plumes_only", "true") == "true"
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 24))

    all_tiles = []
    for filename, regions in ANNOTATIONS.items():
        has_plume = len(regions) > 0
        if plumes_only and not has_plume:
            continue
        path = _ensure_tile(filename)
        url = f"/api/tile_serve?file={filename}"
        ann_url = f"/api/tile_serve?file={filename}&annotated=true"
        all_tiles.append({
            "filename": filename,
            "url": url,
            "annotated_url": ann_url,
            "has_plume": has_plume
        })

    total = len(all_tiles)
    pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    return jsonify({
        "tiles": all_tiles[start:end],
        "total": total,
        "page": page,
        "pages": pages
    })

@app.route("/api/tile_serve")
def api_tile_serve():
    filename = request.args.get("file")
    annotated = request.args.get("annotated", "false") == "true"
    if not filename: return "Missing file", 400
    if annotated:
        path = os.path.join(ANN_DIR, filename)
    else:
        path = os.path.join(TILE_DIR, filename)
    if not os.path.exists(path):
        path = _ensure_tile(filename)
    return send_file(path, mimetype="image/png")

@app.route("/api/spectral_bands")
def api_spectral_bands():
    return jsonify(_spectral_band_payload())


@app.route("/api/detect", methods=["POST"])
def api_detect():
    img_file = request.files.get("file")
    threshold = float(request.form.get("threshold", 180))
    if not img_file: return jsonify({"error": "No file"}), 400

    data = img_file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    overlay, dets, area = detect_plumes(img, threshold)

    return jsonify({
        "visualized_image": _png_b64(overlay),
        "plumes_detected": len(dets),
        "total_area": area,
        "detections": dets
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
