"""Shared "coherence pipeline" architecture pieces from 仿色一致性升级方案:

- C3-1 (`compute_global_mood`/`apply_global_base`): the shared, low-order
  "same light, same photo" correction that runs BEFORE any per-region
  residual grading. Deliberately restricted to a bounded whole-image
  ADDITIVE Lab shift, not the discredited global Reinhard mean/std RESCALE
  (`scripts/color_transfer.py`) already proven to bleed a reference's
  strong colors across unrelated content. A capped additive shift cannot
  amplify variance or smear a strong colored light source over the whole
  photo — it can only nudge the whole image's average brightness/color a
  bounded amount toward the reference's average, the "same white balance
  and exposure" effect a real photographer would apply before touching any
  specific region.
- C3-3 (`edge_aware_weights`/`guided_filter`): replaces
  `common.feather_mask`'s content-agnostic fixed-radius Gaussian blur with
  a guided filter for the coherence pipeline's per-class blend weights, so
  a mask boundary "snaps" toward the target photo's own real edges (a
  treeline, a building silhouette) instead of smearing a fixed number of
  pixels across it regardless of what's actually there.

`color_reference_transfer.py`'s `_render_coherence_from_analysis` composes
these; region-residual trust math (C3-2) lives there, not here, since it
needs per-class reference stats this module has no reason to know about.
"""
from __future__ import annotations

import numpy as np

# Calibrated per the upgrade design doc (仿色一致性升级方案 §三·阶段二):
# a global base is meant to be felt, not to repaint the photo — keep both
# caps well inside what a single feathered-boundary overshoot bug used to
# produce (that bug moved ΔL by ~29 on its own before this fix existed).
MAX_DELTA_L = 12.0
MAX_DELTA_AB = 9.0
MAX_FG_LUMA_LIFT = 8.0

FG_CLASSES = frozenset({"skin", "clothing", "neutral"})
BG_CLASSES = frozenset({"sky", "grass", "tree", "water", "led screen", "led wall",
                        "stage backdrop", "spotlight", "building", "floor", "flag"})
# Faces get at most half the global *color* shift on skin: skin already has its
# dedicated hue-locked grading path. Luminance is NOT damped on skin — user
# review (2026-07-10) flagged "前景的人物没有提亮" when L and ab were both
# halved together.
SKIN_AB_DAMPING = 0.5

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
    """Add the already-bounded-and-scaled mood shift to every pixel.

    L channel applies at full strength everywhere (including skin) so
    foreground people can brighten with the rest of the photo. Only the
    a/b color axes are damped inside skin regions."""
    out = tgt_lab.copy()
    if mood["delta_L"] == 0.0 and mood["delta_a"] == 0.0 and mood["delta_b"] == 0.0:
        return out
    out[..., 0] = out[..., 0] + mood["delta_L"]
    if skin_mask is not None:
        ab_factor = 1.0 - SKIN_AB_DAMPING * skin_mask
        out[..., 1] = out[..., 1] + mood["delta_a"] * ab_factor
        out[..., 2] = out[..., 2] + mood["delta_b"] * ab_factor
    else:
        out[..., 1] = out[..., 1] + mood["delta_a"]
        out[..., 2] = out[..., 2] + mood["delta_b"]
    return out


def _combine_masks(class_masks: dict[str, np.ndarray], names) -> np.ndarray | None:
    out = None
    for c in names:
        m = class_masks.get(c)
        if m is None:
            continue
        out = m if out is None else np.maximum(out, m)
    return out


def compute_foreground_luma_lift(profile: dict, base_lab: np.ndarray,
                                 class_masks: dict[str, np.ndarray],
                                 class_names, fg_luma_lift_strength: float,
                                 compat: dict) -> float:
    """Scalar additive L to apply on foreground pixels only.

    Closes the gap between target fg-vs-bg brightness and the reference's
    fg-vs-bg brightness — directly addresses user review "亮度不统一 /
    前景没有提亮" when the global base lifted background (sky/building)
    more than people (neutral/clothing fall into weak residual caps).
    """
    if fg_luma_lift_strength <= 0.0 or not compat.get("suitable", False):
        return 0.0
    fg_present = [c for c in class_names if c in FG_CLASSES and class_masks.get(c) is not None
                  and (class_masks[c] > 0.5).sum() >= 20]
    bg_present = [c for c in class_names if c in BG_CLASSES and class_masks.get(c) is not None
                  and (class_masks[c] > 0.5).sum() >= 20]
    if not fg_present or not bg_present:
        return 0.0

    fg_mask = _combine_masks(class_masks, fg_present)
    bg_mask = _combine_masks(class_masks, bg_present)
    fg_sel, bg_sel = fg_mask > 0.5, bg_mask > 0.5
    if fg_sel.sum() < 20 or bg_sel.sum() < 20:
        return 0.0

    gap_tgt = float(base_lab[..., 0][bg_sel].mean() - base_lab[..., 0][fg_sel].mean())
    ref_fg = [profile["classes"][c]["l_mean"] for c in fg_present if c in profile.get("classes", {})]
    ref_bg = [profile["classes"][c]["l_mean"] for c in bg_present if c in profile.get("classes", {})]
    if not ref_fg or not ref_bg:
        return 0.0
    gap_ref = float(np.mean(ref_bg)) - float(np.mean(ref_fg))

    # Target fg is darker relative to bg than reference → positive lift on fg.
    raw = (gap_tgt - gap_ref) * fg_luma_lift_strength
    confidence = float(np.clip(compat.get("explainable_tgt_frac", 0.0), 0.0, 1.0))
    return float(np.clip(raw * confidence, -MAX_FG_LUMA_LIFT, MAX_FG_LUMA_LIFT))


def apply_foreground_luma_lift(base_lab: np.ndarray, lift: float,
                               fg_mask: np.ndarray,
                               skin_mask: np.ndarray | None = None) -> np.ndarray:
    if abs(lift) < 1e-3:
        return base_lab
    sel = fg_mask > 0.5
    if sel.sum() < 20:
        return base_lab
    out = base_lab.copy()
    factor = np.ones(base_lab.shape[:2], dtype=np.float32)
    if skin_mask is not None:
        factor = np.where(skin_mask > 0.5, 0.85, factor)
    out[..., 0][sel] = out[..., 0][sel] + lift * factor[sel]
    return out


def boundary_residual_damp(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Per-pixel [0,1] damp on regional residuals at mask boundaries — user
    review flagged persistent halos ("边界还是明显") even after guided filter."""
    import cv2
    grad_total = None
    for w in weights.values():
        gx = cv2.Sobel(w.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(w.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        grad_total = mag if grad_total is None else grad_total + mag
    if grad_total is None or float(grad_total.max()) < 1e-6:
        h, w = next(iter(weights.values())).shape
        return np.ones((h, w), dtype=np.float32)
    thresh = max(float(np.percentile(grad_total, 85)), 1e-3)
    return (1.0 - np.clip(grad_total / (thresh * 2.5), 0.0, 0.70)).astype(np.float32)


# ------------------------------------------------------------- C3-3: edge-aware blend weights

GUIDE_RADIUS = 8
GUIDE_EPS = 1e-2


def _box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """The O(1)-per-pixel box-filter primitive a guided filter is built
    from (He, Sun & Tang 2013). cv2 imported lazily to match the project's
    existing lazy-cv2-import convention (see scripts/face_detect.py)."""
    import cv2
    k = 2 * radius + 1
    return cv2.boxFilter(img.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = GUIDE_RADIUS,
                  eps: float = GUIDE_EPS) -> np.ndarray:
    """Single-channel guided filter: edge-aware smoothing of `src`, guided by
    `guide` (both float, same H×W shape, `guide` normalized roughly to
    [0,1]). Where `guide` is locally flat, this behaves like a plain box
    blur; where `guide` has a sharp edge, the filtered `src` snaps toward
    following that edge instead of smearing across it — exactly the "羽化
    残差图，用原图亮度作引导" behavior 仿色一致性升级方案 §三·阶段五 asks
    for, using the cheap linear-time box-filter formulation rather than
    Matting Laplacian.
    """
    mean_g = _box_filter(guide, radius)
    mean_s = _box_filter(src, radius)
    mean_gs = _box_filter(guide * src, radius)
    cov_gs = mean_gs - mean_g * mean_s
    var_g = _box_filter(guide * guide, radius) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    return mean_a * guide + mean_b


def edge_aware_weights(guide_l: np.ndarray, class_masks: dict[str, np.ndarray],
                       radius: int = GUIDE_RADIUS, eps: float = GUIDE_EPS) -> dict[str, np.ndarray]:
    """Coherence-pipeline replacement for analyze_target's Gaussian-feathered
    per-class `weights`: guided-filter each raw soft class mask using the
    target's own L channel (`guide_l`, Lab lightness in its native 0-100
    range) as the edge guide, then renormalize so every pixel's weights
    across classes still sum to ~1, same contract as the legacy weights.
    """
    guide = (guide_l / 100.0).astype(np.float32)
    smoothed = {c: np.clip(guided_filter(guide, m.astype(np.float32), radius=radius, eps=eps), 0.0, 1.0)
                for c, m in class_masks.items()}
    denom = sum(smoothed.values()) + 1e-6
    return {c: smoothed[c] / denom for c in smoothed}
