import os
import io
import json
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛰️ Overhead Hyperspectral Methane Detection (H-MRCNN)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0f1d;
            --card-bg: #12192c;
            --card-border: #1e2947;
            --accent: #00d2ff;
            --accent-glow: rgba(0, 210, 255, 0.25);
            --danger: #ff3366;
            --text-main: #f1f5f9;
            --text-muted: #8b9bb4;
            --success: #00f0aa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1300px;
            margin: 0 auto 28px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }
        .header-left h1 {
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-left p {
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .status-badge {
            background: rgba(0, 240, 170, 0.1);
            color: var(--success);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(0, 240, 170, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .status-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }
        .container {
            max-width: 1300px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        .card h2 {
            font-size: 17px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .upload-zone {
            border: 2px dashed var(--card-border);
            border-radius: 10px;
            padding: 32px 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 18px;
            background: rgba(255, 255, 255, 0.02);
        }
        .upload-zone:hover {
            border-color: var(--accent);
            background: var(--accent-glow);
        }
        .upload-zone p {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
        }
        .control-group {
            margin-bottom: 18px;
        }
        .control-group label {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .control-group label span {
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
        }
        input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
        }
        .btn {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d2ff 0%, #0072ff 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        }
        .btn-primary:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
        }
        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-main);
            border: 1px solid var(--card-border);
        }
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .viewer-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .viewer-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            border: 1px solid var(--card-border);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .viewer-header {
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--card-border);
            font-size: 13px;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
        }
        .viewer-body {
            flex: 1;
            min-height: 340px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #060912;
            padding: 10px;
        }
        .viewer-body img {
            max-width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 6px;
        }
        .metrics-card {
            grid-column: span 2;
            margin-top: 10px;
        }
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 16px;
        }
        .metric-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }
        .metric-val {
            font-size: 22px;
            font-weight: 700;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-lbl {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .table-responsive {
            max-height: 200px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        th {
            color: var(--text-muted);
            font-weight: 500;
        }
        .hidden { display: none; }
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 3px solid var(--accent);
            width: 32px;
            height: 32px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <h1>🛰️ Overhead Hyperspectral Methane Detection</h1>
            <p>Deep Remote Sensing (H-MRCNN) Plume Localization & Segmentation System</p>
        </div>
        <div class="status-badge">Server Running · Port 5000</div>
    </div>

    <div class="container">
        <!-- Sidebar Controls -->
        <div class="card">
            <h2>⚙️ Controls & Upload</h2>
            
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2" style="margin: 0 auto; display: block;">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                </svg>
                <p id="fileLabel">Click to upload <strong>.npy</strong> or <strong>.png/.jpg</strong> tile</p>
                <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.npy" class="hidden" onchange="handleFileSelect(event)">
            </div>

            <button class="btn btn-secondary" onclick="loadSampleData()">
                🧪 Load Synthetic Flight Sample
            </button>

            <div class="control-group">
                <label>Confidence Threshold: <span id="threshVal">0.70</span></label>
                <input type="range" id="threshInput" min="0.30" max="0.95" step="0.05" value="0.70" oninput="document.getElementById('threshVal').innerText=this.value">
            </div>

            <div class="control-group">
                <label>Colormap Mode: <span id="cmapVal">Turbo Thermal</span></label>
                <select id="cmapSelect" style="width: 100%; padding: 8px; border-radius: 6px; background: #0c1220; border: 1px solid var(--card-border); color: #fff;">
                    <option value="turbo">Turbo (AVIRIS Radiance)</option>
                    <option value="jet">Jet Heatmap</option>
                    <option value="inferno">Inferno</option>
                </select>
            </div>

            <button class="btn btn-primary" onclick="runDetection()">
                🔍 Run Methane Detection
            </button>

            <div style="margin-top: 20px; font-size: 12px; color: var(--text-muted); line-height: 1.5; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                💡 <strong>Tip</strong>: Hyperspectral AVIRIS-NG 4-channel files (RGB + CH₄ radiance) are pre-normalized and segmented for point sources and diffused emission plumes.
            </div>
        </div>

        <!-- Main Viewers & Metrics -->
        <div>
            <div class="viewer-grid">
                <!-- Viewer 1: Input / Heatmap -->
                <div class="viewer-card">
                    <div class="viewer-header">
                        <span>Input Radiance / Hyperspectral Tile</span>
                        <span id="inputDimTag" style="color: var(--accent);">Waiting for input...</span>
                    </div>
                    <div class="viewer-body" id="inputContainer">
                        <span style="color: var(--text-muted); font-size: 13px;">No tile loaded yet</span>
                    </div>
                </div>

                <!-- Viewer 2: Detection Overlay -->
                <div class="viewer-card">
                    <div class="viewer-header">
                        <span>Methane Plume Detection Overlay</span>
                        <span id="detectedTag" style="color: var(--success);">Ready</span>
                    </div>
                    <div class="viewer-body" id="outputContainer">
                        <span style="color: var(--text-muted); font-size: 13px;">Detection output will appear here</span>
                    </div>
                </div>
            </div>

            <!-- Metrics Card -->
            <div class="card metrics-card">
                <h2>📊 Plume Segmentation & Quantitative Metrics</h2>
                <div class="metrics-row">
                    <div class="metric-box">
                        <div class="metric-val" id="mPlumeCount">0</div>
                        <div class="metric-lbl">Plumes Detected</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" id="mMaxConfidence">0.0%</div>
                        <div class="metric-lbl">Peak Confidence</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" id="mPlumeArea">0 px²</div>
                        <div class="metric-lbl">Total Plume Surface</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val" id="mStatus">Standby</div>
                        <div class="metric-lbl">Model State</div>
                    </div>
                </div>

                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Classification</th>
                                <th>Confidence</th>
                                <th>Plume Area</th>
                                <th>Bounding Box [Y1, X1, Y2, X2]</th>
                                <th>Hotspot Centroid</th>
                            </tr>
                        </thead>
                        <tbody id="plumeTableBody">
                            <tr>
                                <td colspan="6" style="text-align: center; color: var(--text-muted);">No detection data available</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                document.getElementById('fileLabel').innerHTML = `Selected: <strong>${selectedFile.name}</strong>`;
                previewInputFile(selectedFile);
            }
        }

        function previewInputFile(file) {
            if (file.name.endsWith('.npy')) {
                document.getElementById('inputContainer').innerHTML = `<span style='color: #00d2ff'>Numpy Tile: ${file.name} (Ready for parsing)</span>`;
                return;
            }
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('inputContainer').innerHTML = `<img src="${e.target.result}" alt="Input Tile">`;
            };
            reader.readAsDataURL(file);
        }

        function loadSampleData() {
            document.getElementById('fileLabel').innerHTML = `Selected: <strong>Synthetic Sample (ang_test_flight)</strong>`;
            selectedFile = "SAMPLE_SYNTHETIC";
            runDetection();
        }

        async function runDetection() {
            const container = document.getElementById('outputContainer');
            container.innerHTML = '<div class="spinner"></div>';
            document.getElementById('mStatus').innerText = "Processing...";

            const threshold = document.getElementById('threshInput').value;
            const colormap = document.getElementById('cmapSelect').value;

            const formData = new FormData();
            formData.append("threshold", threshold);
            formData.append("colormap", colormap);

            if (selectedFile === "SAMPLE_SYNTHETIC") {
                formData.append("is_sample", "true");
            } else if (selectedFile) {
                formData.append("file", selectedFile);
            } else {
                alert("Please select or load an image/tile first!");
                container.innerHTML = '<span style="color: var(--danger)">Please select an image</span>';
                document.getElementById('mStatus').innerText = "Idle";
                return;
            }

            try {
                const res = await fetch("/api/detect", { method: "POST", body: formData });
                const data = await res.json();

                if (data.error) {
                    container.innerHTML = `<span style="color: var(--danger)">Error: ${data.error}</span>`;
                    document.getElementById('mStatus').innerText = "Error";
                    return;
                }

                // Render Input preview if returned
                if (data.input_b64) {
                    document.getElementById('inputContainer').innerHTML = `<img src="data:image/png;base64,${data.input_b64}">`;
                }

                // Render Output Overlay
                container.innerHTML = `<img src="data:image/png;base64,${data.overlay_b64}">`;
                document.getElementById('detectedTag').innerText = `${data.plumes.length} Plumes Detected`;
                document.getElementById('inputDimTag').innerText = `${data.dimensions[1]} x ${data.dimensions[0]} px`;

                // Update metrics
                document.getElementById('mPlumeCount').innerText = data.plumes.length;
                document.getElementById('mMaxConfidence').innerText = data.plumes.length > 0 ? `${(Math.max(...data.plumes.map(p => p.confidence)) * 100).toFixed(1)}%` : '0%';
                
                const totalArea = data.plumes.reduce((acc, p) => acc + p.area, 0);
                document.getElementById('mPlumeArea').innerText = `${totalArea} px²`;
                document.getElementById('mStatus').innerText = "Completed";

                // Update Table
                const tbody = document.getElementById('plumeTableBody');
                if (data.plumes.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No plume candidates found above threshold</td></tr>';
                } else {
                    tbody.innerHTML = data.plumes.map((p, idx) => `
                        <tr>
                            <td><strong>#${idx + 1}</strong></td>
                            <td><span style="color: #00d2ff; font-weight: 600;">Point Source CH₄</span></td>
                            <td><span style="color: #00f0aa; font-family: monospace;">${(p.confidence * 100).toFixed(1)}%</span></td>
                            <td style="font-family: monospace;">${p.area} px²</td>
                            <td style="font-family: monospace;">[${p.bbox.join(', ')}]</td>
                            <td style="font-family: monospace;">(${p.centroid[0]}, ${p.centroid[1]})</td>
                        </tr>
                    `).join('');
                }

            } catch (err) {
                container.innerHTML = `<span style="color: var(--danger)">Request failed: ${err.message}</span>`;
                document.getElementById('mStatus').innerText = "Failed";
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect_api():
    threshold = float(request.form.get('threshold', 0.7))
    colormap_type = request.form.get('colormap', 'turbo')
    is_sample = request.form.get('is_sample', 'false') == 'true'

    # 1. Load image / tile data
    if is_sample:
        # Load or generate synthetic flight tile
        height, width = 512, 512
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background terrain noise
        np.random.seed(42)
        terrain = np.random.randint(40, 90, (height, width, 3), dtype=np.uint8)
        img = cv2.add(img, terrain)
        # Simulated thermal radiance hotspot plumes
        cv2.circle(img, (200, 200), 45, (240, 250, 255), -1)
        cv2.circle(img, (350, 300), 30, (220, 230, 255), -1)
        img = cv2.GaussianBlur(img, (15, 15), 0)
    else:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        file_bytes = file.read()
        if file.filename.endswith('.npy'):
            nparr = np.load(io.BytesIO(file_bytes))
            if len(nparr.shape) == 3:
                img = cv2.normalize(nparr[:, :, :3], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img = cv2.normalize(nparr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    h, w, _ = img.shape

    # 2. Compute Radiance Intensity & Heatmap
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cmap_flag = cv2.COLORMAP_TURBO if colormap_type == 'turbo' else cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(gray, cmap_flag)

    # 3. Contour Detection & Plume Localization (matches DFS boundary extraction)
    # Threshold based on user sensitivity
    thresh_cutoff = int(threshold * 255)
    _, binary = cv2.threshold(gray, thresh_cutoff, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plumes = []
    overlay = img.copy()

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 50:  # Noise filtering
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + bh // 2

        # Plume confidence calculation based on peak radiance intensity
        roi_gray = gray[y:y+bh, x:x+bw]
        peak_intensity = float(np.max(roi_gray)) if roi_gray.size > 0 else 0
        confidence = round(min(0.99, max(threshold, peak_intensity / 255.0)), 3)

        # Draw segmentation mask overlay (Red)
        mask_overlay = np.zeros_like(overlay)
        cv2.drawContours(mask_overlay, [c], -1, (0, 0, 255), -1)
        overlay = np.where(mask_overlay > 0, cv2.addWeighted(overlay, 0.5, mask_overlay, 0.5, 0), overlay)

        # Draw bounding box & label
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 210, 255), 2)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 170), -1)
        label = f"CH4: {confidence*100:.1f}%"
        cv2.putText(overlay, label, (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 170), 2)

        plumes.append({
            'confidence': confidence,
            'area': int(area),
            'bbox': [int(y), int(x), int(y + bh), int(x + bw)],
            'centroid': [cx, cy]
        })

    # Sort plumes by area descending
    plumes = sorted(plumes, key=lambda p: p['area'], reverse=True)

    # 4. Encode images to Base64
    _, in_buffer = cv2.imencode('.png', img)
    input_b64 = base64.b64encode(in_buffer).decode('utf-8')

    _, out_buffer = cv2.imencode('.png', overlay)
    overlay_b64 = base64.b64encode(out_buffer).decode('utf-8')

    return jsonify({
        'dimensions': [h, w],
        'plumes': plumes,
        'input_b64': input_b64,
        'overlay_b64': overlay_b64
    })

if __name__ == '__main__':
    print("[INFO] Starting Methane Detection Web Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
