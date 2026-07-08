"""Phase C: quantify how far the local color_reference_transfer algorithm is
from an actual GPT Image 2 teacher run, per semantic class.

For each matched class, compare (in Lab space, using masks computed on the
ORIGINAL target so all three images share the exact same region):
  orig  -> local   (what our algorithm did)
  orig  -> gpt     (what GPT actually did, the "teacher" target)
  local -> gpt     (the residual — how far local still is from the teacher)

A class where local's move already lands close to GPT's move is fully
localizable (no need to call GPT for that kind of content again). A class
with a big residual needs either stronger/differently-shaped local grading,
or has to stay a GPT job (e.g. content-level changes GPT makes that a
Lab-space stat matcher structurally cannot do).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from color_reference_transfer import build_classes, MIN_FRAC


def class_lab_means(lab: np.ndarray, classes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for c, m in classes.items():
        sel = m > 0.5
        if sel.sum() < 20:
            continue
        out[c] = lab[sel].mean(axis=0)
    return out


def analyze(orig_path: str, local_path: str, gpt_path: str, label: str) -> dict:
    orig = common.load_rgb(orig_path, max_side=1024)
    local = common.load_rgb(local_path, max_side=1024)
    gpt = common.load_rgb(gpt_path, max_side=1024)
    # GPT output may come back at a different resolution; resize to match orig.
    if gpt.shape[:2] != orig.shape[:2]:
        from PIL import Image
        gpt_img = Image.fromarray((np.clip(gpt, 0, 1) * 255).astype(np.uint8)).resize(
            (orig.shape[1], orig.shape[0]), Image.LANCZOS)
        gpt = np.asarray(gpt_img, dtype=np.float32) / 255.0

    classes = build_classes(orig)  # single mask set, shared across all 3 for a fair comparison
    orig_lab, local_lab, gpt_lab = common.rgb_to_lab(orig), common.rgb_to_lab(local), common.rgb_to_lab(gpt)

    orig_m = class_lab_means(orig_lab, classes)
    local_m = class_lab_means(local_lab, classes)
    gpt_m = class_lab_means(gpt_lab, classes)

    report = {}
    for c in orig_m:
        if c not in local_m or c not in gpt_m:
            continue
        frac = float(classes[c].mean())
        if frac < MIN_FRAC:
            continue
        o, l, g = orig_m[c], local_m[c], gpt_m[c]
        gpt_move = g - o          # what GPT actually changed
        local_move = l - o        # what our algorithm changed
        gpt_move_mag = float(np.linalg.norm(gpt_move))
        residual = float(np.linalg.norm(l - g))  # how far local still is from gpt
        # how much of GPT's move direction did local actually reproduce (projection)
        if gpt_move_mag > 0.5:
            captured = float(np.dot(local_move, gpt_move) / (gpt_move_mag ** 2))
        else:
            captured = None  # GPT barely moved this class; ratio not meaningful
        report[c] = {
            "frac": round(frac, 3),
            "orig_Lab": [round(float(v), 2) for v in o],
            "local_Lab": [round(float(v), 2) for v in l],
            "gpt_Lab": [round(float(v), 2) for v in g],
            "gpt_move_mag": round(gpt_move_mag, 2),
            "residual_local_vs_gpt": round(residual, 2),
            "captured_fraction": round(captured, 2) if captured is not None else None,
        }
    return {"label": label, "classes": report}


def main() -> int:
    cases = [
        ("/Volumes/未命名/大模型/原图1/DAP00641.JPG",
        "outputs/color_reference_transfer/gpt_teacher/DAP00641_medium.jpg",
        "outputs/color_transfer/DAP00641_gpt.png", "DAP00641"),
        ("/Volumes/未命名/大模型/原图1/DAP00643.JPG",
        "outputs/color_reference_transfer/gpt_teacher/DAP00643_medium.jpg",
        "outputs/color_transfer/DAP00643_gpt.png", "DAP00643"),
    ]
    results = [analyze(*c) for c in cases]
    out_path = Path("outputs/color_reference_transfer/gpt_teacher/distill_report.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for r in results:
        print(f"\n=== {r['label']} ===")
        for c, v in r["classes"].items():
            cap = f"{v['captured_fraction']*100:.0f}%" if v["captured_fraction"] is not None else "n/a"
            print(f"  {c:20s} frac={v['frac']:.2f} gpt_move={v['gpt_move_mag']:5.2f} "
                  f"residual={v['residual_local_vs_gpt']:5.2f} captured={cap}")
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
