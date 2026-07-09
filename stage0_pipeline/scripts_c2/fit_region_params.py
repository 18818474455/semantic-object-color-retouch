"""C2.2 — Fit per-class Lab-affine "param targets" from (target, pseudo_target)
pixel pairs, back-fit purely from data rather than copied from known formula
constants.

Why fit from pixels instead of just reading off `_grade_class_from_stats`'s
inputs: this makes the param-target extraction agnostic to *how* the edited
image was produced. Today the edited image is our own rule-based
`color_reference_transfer` output; tomorrow it could be a GPT Image 2 result
or a photographer's manual edit. All three cases reduce to the same six
numbers per class: an affine Lab transform (L_scale, L_shift, a_scale,
a_shift, b_scale, b_shift). That is the supervision target for C2.3's
PerClassHead, and it is also the quantity Phase C1 residual analysis
(`distill_vs_gpt.py`) already computes half of (via move-vectors) — this
script makes it a first-class, reusable artifact instead of a one-off report.

A second, explicitly-labeled "proxy" mapping projects the affine Lab params
onto a Chroma-slider-shaped vector (temperature/tint/saturation/vibrance).
This is a HEURISTIC APPROXIMATION for interpretability only — it has not
been validated against the real Chroma C++ renderer (not vendored in this
repo). Treat `chroma_proxy` fields as a rough sanity dial, not ground truth;
true Chroma parity must happen in the Smart Color v2 / Chroma repo (see
CHROMA_ALIGNMENT.md and the design doc's "M3.7 SCv2 嫁接" milestone).

Reads:  dataset/c2/manifest.jsonl (written by export_bootstrap_dataset.py)
Writes: dataset/c2/params/<sample_id>.json      (per-class affine + proxy)
        dataset/c2/param_targets.jsonl          (flattened, one row per class)

Run:
  cd stage0_pipeline
  ../.venv-m2/bin/python scripts_c2/fit_region_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_m2"))

import common
from color_reference_transfer import build_classes

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PIPELINE_ROOT / "dataset" / "c2"
MIN_PIXELS = 20

# Normalization constants for the Lab->Chroma-slider proxy projection.
# Rough order-of-magnitude picks (Lab a/b commonly span roughly +/-40 for
# natural photo content), NOT calibrated against the real engine.
A_NORM = 20.0
B_NORM = 20.0

# Denominator guard for the scale = std(edited)/std(orig) ratio. Found via
# the first full-dataset run (2026-07-09): one "sky" class in the
# person_event bucket is a false-positive detection (no real sky in that
# event photo — same family of bug as the project's documented "LED墙误判
# 天空" issue, resurfacing in the training pipeline instead of the render
# preview) with an almost perfectly flat original mask (std L/a/b =
# 0.15/0.10/0.26). Dividing by that near-zero std blew the scale up to 68x
# with a shift of -6719 — pure numerical noise, not a color-grading signal.
#
# First attempted fix (drop the whole class whenever std < 1.5) was too
# aggressive: legitimate clear-sky regions are *inherently* low-variance
# (smooth gradient, narrow hue) and have std in the 1.9-4.9 range — dropping
# those throws away real, usable supervision. The corrected fix only
# short-circuits the ratio when the region is truly near-constant (std below
# MIN_STD, tuned below the smallest legitimate-sky std observed and above
# the one confirmed-degenerate case), falling back to scale=1.0 (a flat
# region has no well-defined "spread change" to measure) while still using
# the always-well-defined mean-difference for shift. SCALE_CLAMP remains as
# a second safety net for the merely-noisy (not fully degenerate) middle
# ground, e.g. when the EDITED side collapses toward a near-uniform result.
MIN_STD = 0.6
SCALE_CLAMP = (0.15, 6.0)
SHIFT_CLAMP_L = 60.0
SHIFT_CLAMP_AB = 40.0


def _affine_fit(orig: np.ndarray, edited: np.ndarray, shift_clamp: float) -> tuple[float, float]:
    """Least-squares scale+shift so that edited ~= scale*orig + shift.
    Falls back to scale=1.0 when the original region is too flat for the
    std-ratio to be a meaningful estimator (see MIN_STD note above)."""
    o_mean, o_std = float(orig.mean()), float(orig.std())
    e_mean, e_std = float(edited.mean()), float(edited.std())
    if o_std < MIN_STD:
        scale = 1.0
    else:
        scale = float(np.clip(e_std / o_std, *SCALE_CLAMP))
    shift = float(np.clip(e_mean - scale * o_mean, -shift_clamp, shift_clamp))
    return scale, shift


def _chroma_proxy(a_shift: float, b_shift: float, a_scale: float, b_scale: float) -> dict:
    chroma_sat = float(np.clip(((a_scale + b_scale) / 2.0) - 1.0, -1.0, 1.0))
    return {
        "_approx": True,
        "_note": "heuristic Lab->slider projection, not validated against real Chroma renderer",
        "tint": float(np.clip(a_shift / A_NORM, -1.0, 1.0)),
        "temperature": float(np.clip(b_shift / B_NORM, -1.0, 1.0)),
        "saturation": chroma_sat,
        "vibrance": round(chroma_sat * 0.6, 4),
    }


def fit_sample(row: dict) -> dict | None:
    tgt_path = row["target_path"]
    edited_path = PIPELINE_ROOT / row["pseudo_target_path"]
    if not Path(tgt_path).is_file() or not edited_path.is_file():
        print(f"  SKIP {row['sample_id']}: missing target or pseudo_target file")
        return None

    tgt_rgb = common.load_rgb(tgt_path, max_side=1024)
    edited_rgb = common.load_rgb(str(edited_path), max_side=None)
    if edited_rgb.shape[:2] != tgt_rgb.shape[:2]:
        print(f"  SKIP {row['sample_id']}: size mismatch {tgt_rgb.shape} vs {edited_rgb.shape}")
        return None

    tgt_lab = common.rgb_to_lab(tgt_rgb)
    edited_lab = common.rgb_to_lab(edited_rgb)
    tgt_cls = build_classes(tgt_rgb)

    matched_classes = row.get("matched_classes", [])
    per_class = {}
    for c in matched_classes:
        tm = tgt_cls.get(c)
        if tm is None:
            continue
        sel = tm > 0.5
        if sel.sum() < MIN_PIXELS:
            continue
        o_L, e_L = tgt_lab[..., 0][sel], edited_lab[..., 0][sel]
        o_a, e_a = tgt_lab[..., 1][sel], edited_lab[..., 1][sel]
        o_b, e_b = tgt_lab[..., 2][sel], edited_lab[..., 2][sel]

        L_scale, L_shift = _affine_fit(o_L, e_L, SHIFT_CLAMP_L)
        a_scale, a_shift = _affine_fit(o_a, e_a, SHIFT_CLAMP_AB)
        b_scale, b_shift = _affine_fit(o_b, e_b, SHIFT_CLAMP_AB)
        if min(o_L.std(), o_a.std(), o_b.std()) < MIN_STD:
            print(f"    NOTE class '{c}' in {row['sample_id']}: near-flat original region "
                  f"(std L/a/b = {o_L.std():.2f}/{o_a.std():.2f}/{o_b.std():.2f}) — "
                  f"used scale=1.0 fallback on the flat channel(s), likely a weak/false detection")

        per_class[c] = {
            "frac": round(float(sel.mean()), 4),
            "pixels": int(sel.sum()),
            "lab_affine": {
                "L_scale": round(L_scale, 4), "L_shift": round(L_shift, 3),
                "a_scale": round(a_scale, 4), "a_shift": round(a_shift, 3),
                "b_scale": round(b_scale, 4), "b_shift": round(b_shift, 3),
            },
            "chroma_proxy": _chroma_proxy(a_shift, b_shift, a_scale, b_scale),
        }

    return {"sample_id": row["sample_id"], "bucket": row["bucket"], "classes": per_class}


def main() -> int:
    manifest_path = OUT_ROOT / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; run export_bootstrap_dataset.py first")
        return 1

    params_dir = OUT_ROOT / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    rows = common.read_jsonl(manifest_path)
    flat_rows: list[dict] = []
    n_ok = 0
    for row in rows:
        fitted = fit_sample(row)
        if fitted is None or not fitted["classes"]:
            continue
        n_ok += 1
        params_dir.joinpath(f"{fitted['sample_id']}.json").write_text(
            json.dumps(fitted, ensure_ascii=False, indent=2)
        )
        for c, v in fitted["classes"].items():
            flat_rows.append({
                "sample_id": fitted["sample_id"],
                "bucket": fitted["bucket"],
                "class": c,
                "frac": v["frac"],
                "pixels": v["pixels"],
                **{f"lab_{k}": val for k, val in v["lab_affine"].items()},
                **{f"chroma_{k}": val for k, val in v["chroma_proxy"].items() if not k.startswith("_")},
            })
        print(f"OK  {fitted['sample_id']}: {len(fitted['classes'])} classes fitted")

    targets_path = OUT_ROOT / "param_targets.jsonl"
    targets_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in flat_rows) + ("\n" if flat_rows else "")
    )
    print(f"\nfitted_samples={n_ok}/{len(rows)}  class_rows={len(flat_rows)}")
    print(f"per-sample params -> {params_dir}")
    print(f"flattened targets -> {targets_path}")
    if len(flat_rows) < 8:
        print("\nNOTE: too few class-rows for real training yet. Mount the source photo "
              "volume and re-run export_bootstrap_dataset.py (without --smoke-only) to scale up.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
