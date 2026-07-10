"""Light Web Demo for the canonical 仿色 pipeline
(scripts_m2/color_reference_transfer.py) — 参考图 + 目标图 + 强度滑杆.

Design goal: let a human eyeball the C2 rule-based teacher's actual output
on real photos before deciding whether it's worth the bigger M3.7 Chroma
graft investment. Not a production service — single-process, in-memory
session cache, localhost-only by default.

Uses analyze_target()/render_from_analysis() (color_reference_transfer.py)
instead of apply_profile() directly: the expensive part (Grounding DINO +
SAM detection + per-class Lab regrade) runs ONCE per uploaded pair on
/api/analyze, and the strength slider hits the cheap /api/render endpoint
on every drag, which only re-blends + re-finishes (numpy, <100ms).

Must run under .venv-m2 (needs torch/transformers via color_reference_transfer).

Run:
  cd stage0_pipeline
  ../.venv-m2/bin/python webdemo/app.py
  open http://127.0.0.1:5057
"""
from __future__ import annotations

import io
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, render_template

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))
sys.path.insert(0, str(PIPELINE_ROOT / "scripts_m2"))

import common  # noqa: E402
from color_reference_transfer import (  # noqa: E402
    compute_style_profile, analyze_target, render_from_analysis, STRENGTH_PRESETS,
    PIPELINE_LEGACY, SUPPORTED_PIPELINES,
)

app = Flask(__name__)

MAX_SIDE = 1024
MAX_SESSIONS = 12  # small in-memory LRU-ish cap; this is a local demo, not a service
SESSIONS: dict[str, dict] = {}
SESSION_ORDER: list[str] = []

IDENTITY_PRESET = {"default": 0.0, "skin": 0.0, "neutral": 0.0, "global_base": 0.0,
                   "region_default": 0.0, "region_skin": 0.0, "region_neutral": 0.0,
                   "vibrance": 0.0, "contrast": 1.0, "sharpen": 0.0}
# Slider anchors: 0% = no-op passthrough, 33/66/100% = the three validated
# presets from the 20-image regression sweep. Anything in between is a
# linear interpolation of each numeric knob between the two bracketing
# anchors, so the slider sweeps smoothly through presets that are known-good
# rather than extrapolating past "strong".
SLIDER_ANCHORS = [
    (0, IDENTITY_PRESET),
    (33, STRENGTH_PRESETS["light"]),
    (66, STRENGTH_PRESETS["medium"]),
    (100, STRENGTH_PRESETS["strong"]),
]


def _interp_preset(pct: float) -> dict:
    pct = max(0.0, min(100.0, pct))
    for (p0, d0), (p1, d1) in zip(SLIDER_ANCHORS, SLIDER_ANCHORS[1:]):
        if p0 <= pct <= p1:
            t = 0.0 if p1 == p0 else (pct - p0) / (p1 - p0)
            return {k: d0[k] + t * (d1[k] - d0[k]) for k in d0}
    return dict(SLIDER_ANCHORS[-1][1])


def _rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 88) -> bytes:
    from PIL import Image
    out = np.clip(rgb, 0.0, 1.0)
    im = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _load_rgb_from_upload(file_storage, max_side: int = MAX_SIDE) -> np.ndarray:
    from PIL import Image, ImageOps
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _evict_if_needed() -> None:
    while len(SESSION_ORDER) > MAX_SESSIONS:
        old = SESSION_ORDER.pop(0)
        SESSIONS.pop(old, None)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    ref_file = request.files.get("ref")
    tgt_file = request.files.get("tgt")
    if not ref_file or not tgt_file:
        return jsonify({"error": "need both ref and tgt image files"}), 400

    t0 = time.time()
    try:
        ref_rgb = _load_rgb_from_upload(ref_file)
        tgt_rgb = _load_rgb_from_upload(tgt_file)
    except Exception as e:
        return jsonify({"error": f"failed to decode image(s): {e}"}), 400

    profile = compute_style_profile(ref_rgb, name="uploaded_ref")
    analysis = analyze_target(profile, tgt_rgb)
    elapsed = time.time() - t0

    token = uuid.uuid4().hex[:12]
    SESSIONS[token] = {
        "analysis": analysis,
        "ref_preview": _rgb_to_jpeg_bytes(ref_rgb),
        "tgt_preview": _rgb_to_jpeg_bytes(tgt_rgb),
        "ts": time.time(),
    }
    SESSION_ORDER.append(token)
    _evict_if_needed()

    compat = analysis["compat"]
    matched_classes = sorted(c for c, v in analysis["matched_info"].items() if v["matched"] and c != "neutral")
    ref_classes = sorted(profile["classes"])
    tgt_present = sorted(c for c, v in analysis["matched_info"].items() if v["tgt_frac"] > 0.01)

    return jsonify({
        "token": token,
        "compat": compat,
        "matched_classes": matched_classes,
        "ref_classes": ref_classes,
        "tgt_present_classes": tgt_present,
        "analyze_seconds": round(elapsed, 2),
    })


@app.route("/api/render/<token>")
def api_render(token: str):
    session = SESSIONS.get(token)
    if session is None:
        return jsonify({"error": "unknown or expired token; re-run /api/analyze"}), 404
    try:
        pct = float(request.args.get("strength", "50"))
    except ValueError:
        pct = 50.0
    pipeline = request.args.get("pipeline", PIPELINE_LEGACY)
    if pipeline not in SUPPORTED_PIPELINES:
        return jsonify({"error": f"unknown pipeline {pipeline!r}; expected one of {SUPPORTED_PIPELINES}"}), 400
    preset = _interp_preset(pct)
    out_rgb = render_from_analysis(session["analysis"], strength=preset, pipeline=pipeline)
    return send_file(io.BytesIO(_rgb_to_jpeg_bytes(out_rgb)), mimetype="image/jpeg")


@app.route("/api/preview/<token>/<kind>")
def api_preview(token: str, kind: str):
    session = SESSIONS.get(token)
    if session is None:
        return jsonify({"error": "unknown or expired token"}), 404
    key = "ref_preview" if kind == "ref" else "tgt_preview"
    return send_file(io.BytesIO(session[key]), mimetype="image/jpeg")


if __name__ == "__main__":
    print("语义调色专家 · 仿色 Web Demo")
    print("http://127.0.0.1:5057")
    app.run(host="127.0.0.1", port=5057, debug=False)
