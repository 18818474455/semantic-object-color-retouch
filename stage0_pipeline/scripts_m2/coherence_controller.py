"""C3-1: global mood base — the shared, low-order "same light, same photo"
correction that runs BEFORE any per-region residual grading (C3-2+ will add
the residual layer on top of this).

Deliberately restricted to a bounded whole-image ADDITIVE Lab shift, not the
discredited global Reinhard mean/std RESCALE (`scripts/color_transfer.py`)
that was already proven to bleed a reference's strong colors across
unrelated content. A capped additive shift cannot amplify variance or smear
a strong colored light source over the whole photo — it can only nudge the
whole image's average brightness/color a bounded amount toward the
reference's average, which is exactly the "same white balance and exposure"
effect a real photographer would apply before touching any specific region.

This is intentionally the ENTIRE C3-1 deliverable: no per-region residual
logic lives here yet. `render_from_analysis(..., pipeline="coherence")` in
`color_reference_transfer.py` currently applies ONLY this global base (see
outputs/phase-c3-1-global-mood-base.md) so the effect can be visually judged
in isolation before C3-2 adds trust-controlled regional residuals on top.
"""
from __future__ import annotations

import numpy as np

# Calibrated per the upgrade design doc (仿色一致性升级方案 §三·阶段二):
# a global base is meant to be felt, not to repaint the photo — keep both
# caps well inside what a single feathered-boundary overshoot bug used to
# produce (that bug moved ΔL by ~29 on its own before this fix existed).
MAX_DELTA_L = 10.0
MAX_DELTA_AB = 9.0
# Faces get at most half the global shift: skin already has its own
# dedicated hue-locked grading path (see SKIN_HUE_LOCK in
# semantic_color_transfer.py) and shouldn't also inherit an off-tone cast
# just because the rest of the photo needs a bigger correction.
SKIN_DAMPING = 0.5

ZERO_MOOD = {"delta_L": 0.0, "delta_a": 0.0, "delta_b": 0.0}


def whole_image_lab_stats(lab: np.ndarray) -> dict:
    """Unmasked, whole-frame Lab mean — deliberately NOT per-class, this is
    the "what does this photo look like overall" signal the global base
    reasons about."""
    flat_ab = lab[..., 1:3].reshape(-1, 2)
    return {
        "l_mean": float(lab[..., 0].mean()),
        "mean_ab": [float(flat_ab[:, 0].mean()), float(flat_ab[:, 1].mean())],
    }


def compute_global_mood(profile: dict, tgt_lab: np.ndarray, compat: dict,
                        global_base_strength: float) -> dict:
    """Bounded, confidence-gated whole-image Lab shift toward the
    reference's overall mood.

    Returns an all-zero (no-op) mood if:
    - the reference profile predates this feature and has no "global" entry
      (older cached `--profile-in` JSON files), or
    - `global_base_strength` is 0 (identity slider position), or
    - the content-match gate already decided this reference doesn't suit
      this target (`compat["suitable"]` False) — the existing hard gate from
      `content_match_score`, reused here rather than re-implemented.

    Otherwise the raw reference-vs-target gap is clamped to
    (MAX_DELTA_L, MAX_DELTA_AB), then scaled by both the requested strength
    tier AND `compat["explainable_tgt_frac"]` — a marginal (but still
    "suitable") content match gets a proportionally gentler push instead of
    the same full-strength shift as a great match.
    """
    ref_global = profile.get("global")
    if ref_global is None or global_base_strength <= 0.0 or not compat.get("suitable", False):
        return dict(ZERO_MOOD)

    tgt = whole_image_lab_stats(tgt_lab)
    raw_dL = ref_global["l_mean"] - tgt["l_mean"]
    raw_da = ref_global["mean_ab"][0] - tgt["mean_ab"][0]
    raw_db = ref_global["mean_ab"][1] - tgt["mean_ab"][1]

    dL = float(np.clip(raw_dL, -MAX_DELTA_L, MAX_DELTA_L))
    ab_mag = float(np.hypot(raw_da, raw_db))
    ab_scale = min(1.0, MAX_DELTA_AB / ab_mag) if ab_mag > 1e-6 else 1.0
    da, db = raw_da * ab_scale, raw_db * ab_scale

    confidence = float(np.clip(compat.get("explainable_tgt_frac", 0.0), 0.0, 1.0))
    factor = global_base_strength * confidence
    return {"delta_L": dL * factor, "delta_a": da * factor, "delta_b": db * factor}


def apply_global_base(tgt_lab: np.ndarray, mood: dict,
                       skin_mask: np.ndarray | None = None) -> np.ndarray:
    """Add the already-bounded-and-scaled mood shift to every pixel, halving
    it inside skin regions (see SKIN_DAMPING)."""
    out = tgt_lab.copy()
    if mood["delta_L"] == 0.0 and mood["delta_a"] == 0.0 and mood["delta_b"] == 0.0:
        return out
    factor = 1.0 - SKIN_DAMPING * skin_mask if skin_mask is not None else 1.0
    out[..., 0] = out[..., 0] + mood["delta_L"] * factor
    out[..., 1] = out[..., 1] + mood["delta_a"] * factor
    out[..., 2] = out[..., 2] + mood["delta_b"] * factor
    return out
