"""C1c experiment — can a hosted Qwen3-VL tier (via API易) catch the sky
false-positives that the heuristic `_sky_plausible()` texture+saturation
gate in region_provider_v2.py lets through?

Source samples: every C2 bootstrap row (stage0_pipeline/dataset/c2/meta/*.json)
whose matched_classes already contains "sky" — i.e. rows where the heuristic
gate has ALREADY said "yes, plausible sky" (region_provider_v2.detect_classes
deletes the sky mask entirely when _sky_plausible() returns False, so any
"sky" that survived into matched_classes is, by construction, a heuristic
false-negative-proof positive... or a heuristic false positive, like the
known 058A1518 bug case in person_event).

For each row:
  1. Recompute the "sky" mask via color_reference_transfer.build_classes().
  2. Crop the mask's bounding box (+8% pad) from the target image.
  3. Ask Qwen-VL (full photo + crop) whether the crop is genuinely open
     sky/air, or something else (screen, wall, fabric, highlight, etc).
  4. Record VLM verdict next to the heuristic verdict (always "SKY" by
     construction) and the region's own std (from fit_region_params.py's
     MIN_STD degeneracy note) as a cheap proxy ground-truth signal.

Must run under .venv-m2 (needs torch/transformers for build_classes's
Grounding DINO + SAM call) with network access to api.apiyi.com.

Usage:
  ../.venv-m2/bin/python scripts_c1c/sky_gate_experiment.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))
sys.path.insert(0, str(PIPELINE_ROOT / "scripts_m2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from color_reference_transfer import build_classes  # noqa: E402
from gpt_image2_client import load_secrets  # noqa: E402
from qwen_vl_client import chat_vision  # noqa: E402

META_DIR = PIPELINE_ROOT / "dataset" / "c2" / "meta"
OUT_DIR = PIPELINE_ROOT / "outputs" / "c1c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR = OUT_DIR / "crops"
CROP_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Image 1 is a full photo. Image 2 is a close-up crop of ONE region from "
    "image 1 that an automated detector labeled \"sky\".\n\n"
    "Judge image 2 only: is it genuinely open sky/air (blue sky, clouds, "
    "sunset, overcast haze, a smooth sky-color gradient), or is it something "
    "else that just happens to be bright/pale (e.g. a white wall, projector "
    "screen, LED display, fabric backdrop, balloon, overexposed highlight, "
    "light fixture, or blown-out background)?\n\n"
    "Reply in exactly two lines:\n"
    "VERDICT: SKY  (or)  VERDICT: NOT_SKY\n"
    "REASON: <one short sentence>"
)


def _load_rows(limit: int | None) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(META_DIR / "*.json"))):
        d = json.loads(Path(p).read_text())
        if "sky" in d.get("matched_classes", []) and d.get("bucket") != "smoke":
            rows.append(d)
    if limit:
        rows = rows[:limit]
    return rows


def _bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.08) -> tuple[int, int, int, int] | None:
    sel = mask > 0.5
    if sel.sum() < 20:
        return None
    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = mask.shape
    pad_y = int((y1 - y0 + 1) * pad_frac) + 2
    pad_x = int((x1 - x0 + 1) * pad_frac) + 2
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    return int(y0), int(y1), int(x0), int(x1)


def _parse_verdict(reply: str) -> tuple[str, str]:
    verdict = "UNPARSEABLE"
    reason = reply.strip().replace("\n", " ")
    for line in reply.splitlines():
        u = line.strip().upper()
        if u.startswith("VERDICT:"):
            v = u.split(":", 1)[1].strip()
            if "NOT_SKY" in v or "NOT SKY" in v:
                verdict = "NOT_SKY"
            elif "SKY" in v:
                verdict = "SKY"
        if line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_secrets()
    rows = _load_rows(args.limit)
    print(f"{len(rows)} sample(s) with heuristic sky=plausible to re-check via VLM")

    results = []
    for i, row in enumerate(rows):
        sid = row["sample_id"]
        bucket = row.get("bucket")
        tgt_path = row["target_path"]
        try:
            tgt_rgb = common.load_rgb(tgt_path, max_side=1024)
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] {sid}: SKIP load failed ({e})")
            continue
        classes = build_classes(tgt_rgb)
        sky_mask = classes.get("sky")
        if sky_mask is None:
            print(f"[{i+1}/{len(rows)}] {sid}: SKIP no sky mask on recompute (nondeterministic detector?)")
            continue
        bbox = _bbox_from_mask(sky_mask)
        if bbox is None:
            print(f"[{i+1}/{len(rows)}] {sid}: SKIP mask too small on recompute")
            continue
        y0, y1, x0, x1 = bbox
        crop = tgt_rgb[y0:y1 + 1, x0:x1 + 1]
        sel = sky_mask > 0.5
        std_l, std_a, std_b = (common.rgb_to_lab(tgt_rgb)[..., c][sel].std() for c in range(3))

        full_path = CROP_DIR / f"{sid}_full.jpg"
        crop_path = CROP_DIR / f"{sid}_crop.jpg"
        common.save_rgb(tgt_rgb, full_path)
        common.save_rgb(crop, crop_path)

        try:
            reply = chat_vision(cfg, PROMPT, [full_path, crop_path])
            verdict, reason = _parse_verdict(reply)
        except Exception as e:
            verdict, reason = "ERROR", str(e)[:200]

        frac = float(sel.mean())
        print(f"[{i+1}/{len(rows)}] {sid:32s} bucket={bucket:14s} std(Lab)={std_l:.2f}/{std_a:.2f}/{std_b:.2f} "
              f"frac={frac:.3f} -> VLM={verdict}  ({reason[:80]})")
        results.append({
            "sample_id": sid, "bucket": bucket, "target_path": tgt_path,
            "heuristic_verdict": "SKY", "vlm_verdict": verdict, "vlm_reason": reason,
            "mask_frac": round(frac, 4),
            "std_lab": [round(float(std_l), 3), round(float(std_a), 3), round(float(std_b), 3)],
        })
        time.sleep(0.2)

    report_path = OUT_DIR / "sky_gate_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    n = len(results)
    n_not_sky = sum(1 for r in results if r["vlm_verdict"] == "NOT_SKY")
    n_err = sum(1 for r in results if r["vlm_verdict"] in ("ERROR", "UNPARSEABLE"))
    print(f"\n=== summary: {n} judged, VLM said NOT_SKY for {n_not_sky}, errors/unparseable {n_err} ===")
    by_bucket: dict[str, list[dict]] = {}
    for r in results:
        by_bucket.setdefault(r["bucket"], []).append(r)
    for b, rs in sorted(by_bucket.items()):
        flagged = [r["sample_id"] for r in rs if r["vlm_verdict"] == "NOT_SKY"]
        print(f"  {b:16s} n={len(rs):3d} NOT_SKY={len(flagged)}  {flagged}")
    print(f"\nfull report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
