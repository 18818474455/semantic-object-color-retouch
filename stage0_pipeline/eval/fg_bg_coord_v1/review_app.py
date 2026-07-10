"""C3-4: local web UI for scoring FG-BG-Coord-v1 samples.

Reads `manifest.jsonl` for the sample list/metadata and the per-sample
`outputs/<id>/{review_sheet.jpg, metrics.json, review.json}` already
produced by `eval_harmony.py --manifest ...`. Does NOT re-run any model —
purely a thin scoring UI over files that already exist on disk, so it
starts instantly and doesn't need GPU/torch.

Run:
    cd stage0_pipeline/eval/fg_bg_coord_v1
    ../../../.venv-m2/bin/python review_app.py
    open http://127.0.0.1:5058
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.jsonl"
OUTPUTS_ROOT = HERE / "outputs"

sys.path.insert(0, str(HERE.parent.parent / "scripts_m2"))
from eval_harmony import REVIEW_TEMPLATE, score_summary  # noqa: E402

SCORE_FIELDS = [
    ("foreground_change_visible", "前景变化是否可见（1=完全没反应，5=明显且合理）"),
    ("background_strength_natural", "背景仿色力度是否自然（1=过冲/太假，5=自然）"),
    ("fg_bg_same_tone", "前后景是否像同一张照片（1=严重脱节，5=浑然一体）"),
    ("skin_natural", "肤色是否自然（1=明显异常，5=完全自然）"),
    ("halo_free", "边界是否无光晕（1=光晕明显，5=完全无光晕）"),
    ("local_dirty_color_free", "是否无局部脏色/杂色（1=有明显脏色，5=干净）"),
    ("delivery_willingness", "整体是否愿意交付给客户（1=完全不行，5=可以直接交付）"),
]
SEVERE_FIELDS = [
    ("severe_fg_bg_disconnect", "严重前后景割裂"),
    ("severe_halo", "严重光晕"),
    ("severe_skin_error", "严重肤色异常"),
]

app = Flask(__name__)


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    recs = []
    for line in MANIFEST_PATH.read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def _review_path(sample_id: str) -> Path:
    return OUTPUTS_ROOT / sample_id / "review.json"


def _load_review(sample_id: str) -> dict:
    p = _review_path(sample_id)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(json.dumps(REVIEW_TEMPLATE))


def _load_metrics(sample_id: str) -> dict | None:
    p = OUTPUTS_ROOT / sample_id / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@app.route("/")
def index():
    manifest = _load_manifest()
    for rec in manifest:
        review = _load_review(rec["id"])
        if review.get("preferred") is None:
            return redirect(url_for("review_sample", sample_id=rec["id"]))
    if manifest:
        return redirect(url_for("review_sample", sample_id=manifest[0]["id"]))
    return "manifest.jsonl 是空的或不存在", 404


@app.route("/review/<sample_id>")
def review_sample(sample_id: str):
    manifest = _load_manifest()
    ids = [r["id"] for r in manifest]
    if sample_id not in ids:
        return f"未知样本 id: {sample_id}", 404
    idx = ids.index(sample_id)
    rec = manifest[idx]
    review = _load_review(sample_id)
    metrics = _load_metrics(sample_id)
    scored_count = sum(1 for r in manifest if _load_review(r["id"]).get("preferred") is not None)

    flags = []
    if metrics:
        flags = [f"legacy:{f}" for f in metrics.get("legacy_v0_flags", [])] + \
                [f"coherence:{f}" for f in metrics.get("coherence_v1_flags", [])]

    return render_template(
        "review.html",
        sample=rec,
        review=review,
        score_fields=SCORE_FIELDS,
        severe_fields=SEVERE_FIELDS,
        idx=idx,
        total=len(ids),
        scored_count=scored_count,
        prev_id=ids[idx - 1] if idx > 0 else None,
        next_id=ids[idx + 1] if idx < len(ids) - 1 else None,
        flags=flags,
        compat=(metrics or {}).get("compat"),
    )


@app.route("/review/<sample_id>", methods=["POST"])
def save_review(sample_id: str):
    manifest = _load_manifest()
    ids = [r["id"] for r in manifest]
    if sample_id not in ids:
        return f"未知样本 id: {sample_id}", 404

    review = {
        "preferred": request.form.get("preferred") or None,
        "scores": {},
        "severe": {},
        "notes": request.form.get("notes", ""),
    }
    for key, _ in SCORE_FIELDS:
        v = request.form.get(f"score_{key}")
        review["scores"][key] = int(v) if v else None
    for key, _ in SEVERE_FIELDS:
        review["severe"][key] = request.form.get(f"severe_{key}") == "on"

    out_dir = OUTPUTS_ROOT / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))

    idx = ids.index(sample_id)
    action = request.form.get("action", "next")
    if action == "next" and idx < len(ids) - 1:
        return redirect(url_for("review_sample", sample_id=ids[idx + 1]))
    if action == "prev" and idx > 0:
        return redirect(url_for("review_sample", sample_id=ids[idx - 1]))
    return redirect(url_for("review_sample", sample_id=sample_id))


@app.route("/img/<sample_id>/<filename>")
def serve_image(sample_id: str, filename: str):
    allowed = {"review_sheet.jpg", "reference.jpg", "target.jpg", "legacy_v0.jpg", "coherence_v1.jpg"}
    if filename not in allowed:
        return "not found", 404
    path = OUTPUTS_ROOT / sample_id / filename
    if not path.exists():
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/summary")
def summary():
    return jsonify(score_summary(OUTPUTS_ROOT))


if __name__ == "__main__":
    print("FG-BG-Coord-v1 人工评审")
    print("http://127.0.0.1:5058")
    app.run(host="127.0.0.1", port=5058, debug=False)
