"""
label_tool.py
=============
HTML-based visual labeling tool for system/user text classification.
Uses Python's built-in http.server (zero external dependencies).

Reads recognition JSONs, NER outputs, and regex detection outputs,
shows source images with bbox overlays + crop images, and lets you
label each text block as 'system' or 'user' via keyboard or buttons.

Usage:
    python label_tool.py
    Then open http://localhost:8787 in your browser.
"""

import os
import re
import sys
import json
import glob
import mimetypes
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(BASE_DIR, "recognition_output")
NER_DIR = os.path.join(BASE_DIR, "xlm_roberta_ner_output")
REGEX_DIR = os.path.join(BASE_DIR, "regex_detection")
CROPS_DIR = os.path.join(BASE_DIR, "crops_input1")
INPUT_DIR = os.path.join(BASE_DIR, "input")
LABELS_DIR = os.path.join(BASE_DIR, "labeled_data")
LABELS_FILE = os.path.join(LABELS_DIR, "labeled_dataset.json")

PORT = 8787


def load_all_data():
    """Merge recognition + NER + regex data into unified blocks."""
    rec_files = sorted(glob.glob(os.path.join(REC_DIR, "*_final_recog.json")))
    all_blocks = []

    for rec_path in rec_files:
        with open(rec_path, "r", encoding="utf-8") as f:
            rec_data = json.load(f)

        stem = rec_data.get("image_stem", "")
        source_image = rec_data.get("source_image", "")

        # Try to load NER output
        ner_path = os.path.join(NER_DIR, f"{stem}_xlm_roberta_ner.json")
        ner_results = {}
        if os.path.exists(ner_path):
            with open(ner_path, "r", encoding="utf-8") as f:
                ner_data = json.load(f)
            for item in ner_data.get("results", []):
                ner_results[item.get("det_index")] = item

        # Try to load regex output
        regex_path = os.path.join(REGEX_DIR, f"{stem}_regex.json")
        regex_results = {}
        if os.path.exists(regex_path):
            with open(regex_path, "r", encoding="utf-8") as f:
                regex_data = json.load(f)
            for item in regex_data.get("results", []):
                regex_results[item.get("det_index")] = item

        # Find source image file (jpg or png)
        src_img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(INPUT_DIR, f"{stem}{ext}")
            if os.path.exists(candidate):
                src_img_path = f"/image/input/{stem}{ext}"
                break

        for item in rec_data.get("results", []):
            det_idx = item.get("det_index")
            bbox = item.get("bbox", [0, 0, 0, 0])
            crop_size = item.get("crop_size", [0, 0])

            # NER context
            ner_item = ner_results.get(det_idx, {})
            ner_entity = ner_item.get("entity")
            ner_entities = ner_item.get("entities", [])

            # Regex context
            regex_item = regex_results.get(det_idx, {})
            regex_type = regex_item.get("type")
            regex_signal = regex_item.get("system_or_user")
            regex_matches = regex_item.get("regex_matches", [])

            # Crop image path
            crop_filename = item.get("crop_filename", "")
            crop_img_url = f"/image/crops_input1/{stem}/{crop_filename}" if crop_filename else None

            block = {
                "id": f"{stem}__{det_idx}",
                "image_stem": stem,
                "det_index": det_idx,
                "transcribed_text": item.get("transcribed_text", ""),
                "rec_confidence": item.get("rec_confidence", 0),
                "det_score": item.get("det_score", 0),
                "bbox": bbox,
                "crop_size": crop_size,
                "crop_img_url": crop_img_url,
                "source_img_url": src_img_path,
                "ner_entity": ner_entity,
                "ner_entities": ner_entities,
                "regex_type": regex_type,
                "regex_signal": regex_signal,
                "regex_matches": regex_matches,
                # Weak label suggestion
                "suggested_label": regex_signal if regex_signal else None
            }
            all_blocks.append(block)

    return all_blocks


def load_labels():
    """Load saved labels from disk."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    """Save labels to disk."""
    os.makedirs(LABELS_DIR, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


# Pre-load data at startup
print("Loading recognition, NER, and regex data...")
ALL_BLOCKS = load_all_data()
LABELS = load_labels()
print(f"Loaded {len(ALL_BLOCKS)} text blocks across {len(set(b['image_stem'] for b in ALL_BLOCKS))} images.")
print(f"Existing labels: {len(LABELS)} / {len(ALL_BLOCKS)}")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>System/User Text Labeling Tool</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #252836;
  --border: #2e3142; --text: #e4e6f0; --text-dim: #8b8fa3;
  --accent: #6c63ff; --system-color: #3b82f6; --user-color: #f59e0b;
  --system-bg: rgba(59,130,246,0.12); --user-bg: rgba(245,158,11,0.12);
  --success: #22c55e; --danger: #ef4444;
}
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

/* Top bar */
.topbar {
  position: sticky; top: 0; z-index: 100;
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; gap: 20px;
  backdrop-filter: blur(12px);
}
.topbar h1 { font-size: 16px; font-weight: 700; white-space: nowrap; }
.progress-wrap { flex: 1; display: flex; align-items: center; gap: 10px; }
.progress-bar { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #a78bfa); transition: width 0.3s; border-radius: 3px; }
.progress-text { font-size: 13px; color: var(--text-dim); font-weight: 500; white-space: nowrap; }
.nav-buttons { display: flex; gap: 6px; }
.nav-buttons button {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-family: inherit;
  transition: all 0.15s;
}
.nav-buttons button:hover { background: var(--border); }

/* Filter tabs */
.filter-bar {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 8px 24px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
.filter-btn {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim);
  padding: 5px 14px; border-radius: 20px; cursor: pointer; font-size: 12px; font-family: inherit;
  transition: all 0.15s;
}
.filter-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.filter-btn:hover { border-color: var(--accent); color: var(--text); }
.image-select {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  padding: 5px 10px; border-radius: 6px; font-size: 12px; font-family: inherit; margin-left: auto;
}

/* Main layout */
.main { display: flex; height: calc(100vh - 90px); }

/* Left: source image */
.panel-left {
  width: 40%; border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
}
.source-img-wrap {
  flex: 1; position: relative; overflow: hidden; display: flex;
  align-items: center; justify-content: center; background: #000;
}
.source-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
.bbox-overlay {
  position: absolute; border: 2px solid var(--accent);
  background: rgba(108, 99, 255, 0.15); pointer-events: none;
  transition: all 0.3s; border-radius: 2px;
  box-shadow: 0 0 12px rgba(108, 99, 255, 0.4);
}

/* Right: details + labeling */
.panel-right { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }

.block-card {
  margin: 16px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden;
}
.block-header {
  padding: 14px 18px; background: var(--surface2);
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border);
}
.block-header .block-id { font-size: 13px; color: var(--text-dim); }
.block-header .block-nav { font-size: 13px; font-weight: 600; }

.block-body { padding: 18px; }

/* Crop image */
.crop-row { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 16px; }
.crop-img-wrap {
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  background: #000; min-width: 80px; max-width: 300px; flex-shrink: 0;
}
.crop-img-wrap img { display: block; width: 100%; height: auto; }
.text-display {
  font-size: 20px; font-weight: 600; padding: 10px 0;
  word-break: break-word; line-height: 1.4;
}

/* Info grid */
.info-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px; margin-bottom: 18px;
}
.info-chip {
  background: var(--surface2); border-radius: 8px; padding: 10px 14px;
}
.info-chip .label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.info-chip .value { font-size: 14px; font-weight: 500; }
.info-chip .value.entity-per { color: #f472b6; }
.info-chip .value.entity-loc { color: #34d399; }
.info-chip .value.entity-org { color: #60a5fa; }
.info-chip .value.entity-date { color: #fbbf24; }
.info-chip .value.regex-user { color: var(--user-color); }
.info-chip .value.regex-system { color: var(--system-color); }

/* NER entities detail */
.ner-detail { margin-bottom: 16px; }
.ner-detail h4 { font-size: 12px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 8px; }
.ner-entity-row {
  display: flex; gap: 8px; align-items: center; padding: 6px 10px;
  background: var(--surface2); border-radius: 6px; margin-bottom: 4px; font-size: 13px;
}
.ner-tag {
  padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
}
.ner-tag.PER { background: rgba(244,114,182,0.2); color: #f472b6; }
.ner-tag.LOC { background: rgba(52,211,153,0.2); color: #34d399; }
.ner-tag.ORG { background: rgba(96,165,250,0.2); color: #60a5fa; }
.ner-tag.DATE { background: rgba(251,191,36,0.2); color: #fbbf24; }

/* Label buttons */
.label-section {
  padding: 18px; border-top: 1px solid var(--border);
  display: flex; gap: 12px; align-items: center; justify-content: center;
}
.label-btn {
  padding: 12px 40px; border: 2px solid transparent; border-radius: 10px;
  font-size: 15px; font-weight: 600; cursor: pointer; font-family: inherit;
  transition: all 0.15s; position: relative;
}
.label-btn .shortcut {
  position: absolute; top: -8px; right: -8px;
  background: var(--surface); border: 1px solid var(--border);
  padding: 1px 6px; border-radius: 4px; font-size: 10px; color: var(--text-dim);
}
.label-btn.system-btn {
  background: var(--system-bg); border-color: var(--system-color); color: var(--system-color);
}
.label-btn.system-btn:hover, .label-btn.system-btn.active {
  background: var(--system-color); color: #fff;
}
.label-btn.user-btn {
  background: var(--user-bg); border-color: var(--user-color); color: var(--user-color);
}
.label-btn.user-btn:hover, .label-btn.user-btn.active {
  background: var(--user-color); color: #fff;
}
.label-btn.skip-btn {
  background: var(--surface2); border-color: var(--border); color: var(--text-dim);
  padding: 12px 24px;
}
.label-btn.skip-btn:hover { background: var(--border); color: var(--text); }

.current-label {
  font-size: 13px; padding: 4px 12px; border-radius: 6px; font-weight: 600;
}
.current-label.system { background: var(--system-bg); color: var(--system-color); }
.current-label.user { background: var(--user-bg); color: var(--user-color); }

/* Keyboard hint */
.kb-hint {
  text-align: center; padding: 8px; font-size: 11px; color: var(--text-dim);
  border-top: 1px solid var(--border);
}
kbd {
  background: var(--surface2); border: 1px solid var(--border);
  padding: 1px 5px; border-radius: 3px; font-size: 11px; font-family: inherit;
}

/* Toast */
.toast {
  position: fixed; bottom: 20px; right: 20px; z-index: 1000;
  background: var(--success); color: #fff; padding: 10px 20px;
  border-radius: 8px; font-size: 13px; font-weight: 500;
  opacity: 0; transform: translateY(10px); transition: all 0.3s;
  pointer-events: none;
}
.toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<div class="topbar">
  <h1>🏷️ Label Tool</h1>
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <span class="progress-text" id="progressText">0 / 0</span>
  </div>
  <div class="nav-buttons">
    <button onclick="navigate(-1)">← Prev</button>
    <button onclick="navigate(1)">Next →</button>
    <button onclick="jumpToUnlabeled()">Next Unlabeled</button>
  </div>
</div>

<div class="filter-bar">
  <button class="filter-btn active" data-filter="all" onclick="setFilter('all', this)">All</button>
  <button class="filter-btn" data-filter="unlabeled" onclick="setFilter('unlabeled', this)">Unlabeled</button>
  <button class="filter-btn" data-filter="system" onclick="setFilter('system', this)">System</button>
  <button class="filter-btn" data-filter="user" onclick="setFilter('user', this)">User</button>
  <select class="image-select" id="imageSelect" onchange="filterByImage(this.value)">
    <option value="all">All Images</option>
  </select>
</div>

<div class="main">
  <div class="panel-left">
    <div class="source-img-wrap" id="sourceImgWrap">
      <img id="sourceImg" src="" alt="Source image">
      <div class="bbox-overlay" id="bboxOverlay"></div>
    </div>
  </div>
  <div class="panel-right" id="panelRight">
    <div class="block-card">
      <div class="block-header">
        <span class="block-id" id="blockId">—</span>
        <span class="block-nav" id="blockNav">— / —</span>
      </div>
      <div class="block-body">
        <div class="crop-row">
          <div class="crop-img-wrap"><img id="cropImg" src="" alt="Crop"></div>
          <div class="text-display" id="textDisplay">—</div>
        </div>
        <div class="info-grid" id="infoGrid"></div>
        <div class="ner-detail" id="nerDetail"></div>
      </div>
      <div class="label-section">
        <button class="label-btn system-btn" onclick="applyLabel('system')">
          SYSTEM <span class="shortcut">S</span>
        </button>
        <button class="label-btn user-btn" onclick="applyLabel('user')">
          USER <span class="shortcut">U</span>
        </button>
        <button class="label-btn skip-btn" onclick="navigate(1)">
          SKIP <span class="shortcut">→</span>
        </button>
        <span class="current-label" id="currentLabel"></span>
      </div>
      <div class="kb-hint">
        <kbd>S</kbd> System &nbsp; <kbd>U</kbd> User &nbsp; <kbd>←</kbd> Prev &nbsp; <kbd>→</kbd> Next &nbsp; <kbd>N</kbd> Next Unlabeled
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let allBlocks = [];
let filteredIndices = [];
let labels = {};
let currentFilteredIdx = 0;
let currentFilter = 'all';
let currentImageFilter = 'all';
let imgNaturalW = 1, imgNaturalH = 1;

async function init() {
  const resp = await fetch('/api/data');
  const data = await resp.json();
  allBlocks = data.blocks;
  labels = data.labels;
  // Populate image select
  const stems = [...new Set(allBlocks.map(b => b.image_stem))];
  const sel = document.getElementById('imageSelect');
  stems.forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o); });
  applyFilters();
  renderCurrent();
}

function applyFilters() {
  filteredIndices = [];
  for (let i = 0; i < allBlocks.length; i++) {
    const b = allBlocks[i];
    const lbl = labels[b.id];
    if (currentImageFilter !== 'all' && b.image_stem !== currentImageFilter) continue;
    if (currentFilter === 'unlabeled' && lbl) continue;
    if (currentFilter === 'system' && lbl !== 'system') continue;
    if (currentFilter === 'user' && lbl !== 'user') continue;
    filteredIndices.push(i);
  }
  if (currentFilteredIdx >= filteredIndices.length) currentFilteredIdx = Math.max(0, filteredIndices.length - 1);
}

function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentFilteredIdx = 0;
  applyFilters();
  renderCurrent();
}

function filterByImage(val) {
  currentImageFilter = val;
  currentFilteredIdx = 0;
  applyFilters();
  renderCurrent();
}

function navigate(dir) {
  currentFilteredIdx = Math.max(0, Math.min(filteredIndices.length - 1, currentFilteredIdx + dir));
  renderCurrent();
}

function jumpToUnlabeled() {
  for (let fi = currentFilteredIdx + 1; fi < filteredIndices.length; fi++) {
    if (!labels[allBlocks[filteredIndices[fi]].id]) { currentFilteredIdx = fi; renderCurrent(); return; }
  }
  for (let fi = 0; fi <= currentFilteredIdx; fi++) {
    if (!labels[allBlocks[filteredIndices[fi]].id]) { currentFilteredIdx = fi; renderCurrent(); return; }
  }
  showToast('All blocks in current view are labeled!');
}

function renderCurrent() {
  updateProgress();
  if (filteredIndices.length === 0) {
    document.getElementById('textDisplay').textContent = 'No blocks match current filter.';
    return;
  }
  const block = allBlocks[filteredIndices[currentFilteredIdx]];
  // Header
  document.getElementById('blockId').textContent = `${block.image_stem} / crop_${String(block.det_index).padStart(3,'0')}`;
  document.getElementById('blockNav').textContent = `${currentFilteredIdx + 1} / ${filteredIndices.length}`;
  // Text
  document.getElementById('textDisplay').textContent = block.transcribed_text || '—';
  // Crop image
  const cropImg = document.getElementById('cropImg');
  if (block.crop_img_url) { cropImg.src = block.crop_img_url; cropImg.parentElement.style.display = ''; }
  else { cropImg.parentElement.style.display = 'none'; }
  // Source image + bbox
  const srcImg = document.getElementById('sourceImg');
  const overlay = document.getElementById('bboxOverlay');
  if (block.source_img_url) {
    srcImg.src = block.source_img_url;
    srcImg.onload = function() {
      imgNaturalW = this.naturalWidth; imgNaturalH = this.naturalHeight;
      positionBbox(block.bbox);
    };
  }
  // Info grid
  const grid = document.getElementById('infoGrid');
  const entityClass = block.ner_entity ? 'entity-' + block.ner_entity.toLowerCase() : '';
  const regexClass = block.regex_signal ? 'regex-' + block.regex_signal : '';
  grid.innerHTML = `
    <div class="info-chip"><div class="label">Confidence</div><div class="value">${(block.rec_confidence * 100).toFixed(1)}%</div></div>
    <div class="info-chip"><div class="label">Det Score</div><div class="value">${(block.det_score * 100).toFixed(1)}%</div></div>
    <div class="info-chip"><div class="label">NER Entity</div><div class="value ${entityClass}">${block.ner_entity || 'None'}</div></div>
    <div class="info-chip"><div class="label">Regex Type</div><div class="value">${block.regex_type || 'None'}</div></div>
    <div class="info-chip"><div class="label">Regex Signal</div><div class="value ${regexClass}">${block.regex_signal || 'None'}</div></div>
    <div class="info-chip"><div class="label">BBox</div><div class="value" style="font-size:11px">[${block.bbox.map(v=>Math.round(v)).join(', ')}]</div></div>
    <div class="info-chip"><div class="label">Crop Size</div><div class="value">${block.crop_size[0]}×${block.crop_size[1]}</div></div>
    <div class="info-chip"><div class="label">Suggestion</div><div class="value">${block.suggested_label || '—'}</div></div>
  `;
  // NER entities detail
  const nerDiv = document.getElementById('nerDetail');
  if (block.ner_entities && block.ner_entities.length > 0) {
    nerDiv.innerHTML = '<h4>NER Entities</h4>' + block.ner_entities.map(e => `
      <div class="ner-entity-row">
        <span class="ner-tag ${e.entity_group}">${e.entity_group}</span>
        <span>"${e.word}"</span>
        <span style="color:var(--text-dim);margin-left:auto">score: ${e.score.toFixed(4)}</span>
      </div>
    `).join('');
  } else { nerDiv.innerHTML = '<h4>NER Entities</h4><div style="font-size:13px;color:var(--text-dim)">No entities detected</div>'; }
  // Current label
  const lbl = labels[block.id];
  const lblEl = document.getElementById('currentLabel');
  if (lbl) { lblEl.textContent = lbl.toUpperCase(); lblEl.className = 'current-label ' + lbl; }
  else { lblEl.textContent = ''; lblEl.className = 'current-label'; }
}

function positionBbox(bbox) {
  const wrap = document.getElementById('sourceImgWrap');
  const img = document.getElementById('sourceImg');
  const overlay = document.getElementById('bboxOverlay');
  const dispW = img.clientWidth, dispH = img.clientHeight;
  const scaleX = dispW / imgNaturalW, scaleY = dispH / imgNaturalH;
  const imgLeft = (wrap.clientWidth - dispW) / 2, imgTop = (wrap.clientHeight - dispH) / 2;
  overlay.style.left = (imgLeft + bbox[0] * scaleX) + 'px';
  overlay.style.top = (imgTop + bbox[1] * scaleY) + 'px';
  overlay.style.width = ((bbox[2] - bbox[0]) * scaleX) + 'px';
  overlay.style.height = ((bbox[3] - bbox[1]) * scaleY) + 'px';
  overlay.style.display = 'block';
}

async function applyLabel(label) {
  if (filteredIndices.length === 0) return;
  const block = allBlocks[filteredIndices[currentFilteredIdx]];
  labels[block.id] = label;
  await fetch('/api/label', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: block.id, label: label})
  });
  showToast(`Labeled "${block.transcribed_text}" as ${label.toUpperCase()}`);
  renderCurrent();
  // Auto-advance to next
  setTimeout(() => navigate(1), 150);
}

function updateProgress() {
  const total = allBlocks.length;
  const labeled = Object.keys(labels).length;
  const pct = total > 0 ? (labeled / total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = `${labeled} / ${total} labeled (${pct.toFixed(0)}%)`;
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 's' || e.key === 'S') { e.preventDefault(); applyLabel('system'); }
  else if (e.key === 'u' || e.key === 'U') { e.preventDefault(); applyLabel('user'); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); navigate(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); navigate(-1); }
  else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); jumpToUnlabeled(); }
});

window.addEventListener('resize', () => {
  if (filteredIndices.length > 0) {
    const block = allBlocks[filteredIndices[currentFilteredIdx]];
    setTimeout(() => positionBbox(block.bbox), 50);
  }
});

init();
</script>
</body>
</html>"""


class LabelHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/data":
            self._serve_json({"blocks": ALL_BLOCKS, "labels": LABELS})
        elif path.startswith("/image/"):
            self._serve_image(path[7:])  # strip /image/
        else:
            self._send_404()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/label":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
            block_id = data.get("id")
            label = data.get("label")
            if block_id and label in ("system", "user"):
                LABELS[block_id] = label
                save_labels(LABELS)
                self._serve_json({"status": "ok", "total_labeled": len(LABELS)})
            else:
                self._serve_json({"status": "error", "message": "Invalid data"}, 400)
        else:
            self._send_404()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _serve_image(self, rel_path):
        # Normalize path separators
        rel_path = rel_path.replace("/", os.sep).replace("\\", os.sep)
        abs_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            mime, _ = mimetypes.guess_type(abs_path)
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            with open(abs_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._send_404()

    def _send_404(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not Found")


def main():
    server = HTTPServer(("0.0.0.0", PORT), LabelHandler)
    print(f"\n{'='*60}")
    print(f"  LABELING TOOL RUNNING")
    print(f"  Open http://localhost:{PORT} in your browser")
    print(f"  Keyboard: S=System, U=User, ←→=Navigate, N=Next Unlabeled")
    print(f"{'='*60}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nServer stopped. {len(LABELS)} labels saved to {LABELS_FILE}")
        server.server_close()


if __name__ == "__main__":
    main()
