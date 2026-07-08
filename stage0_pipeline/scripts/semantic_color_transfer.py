"""Semantic (per-class) reference color transfer — the anti-串色 path, v2.

v1 used plain Reinhard mean/std matching for every class. That is a
"regression to the mean" operator: it flattens contrast and, worse, we damped
skin strength to 0.45 out of caution, so skin never picked up the reference's
brightness/glow — the result looked dull and muddy instead of "白皙通透".

v2 fixes:
  1. Percentile (5/95) tone-curve mapping for L instead of mean/std — this
     transfers actual highlight/shadow punch instead of just re-centering.
  2. Skin uses hue-locked chroma matching: L and chroma magnitude move (near)
     fully toward the reference's own skin look (brighter, cleaner), but hue
     angle is only nudged a little, so skin can't drift into a red/green cast
     even though the overall grade is being pulled toward a red-heavy scene.
  3. A global vibrance + micro-contrast + light sharpening pass after the
     per-class grade, matching the "clean / high-definition" look a reference
     photo like a lit commercial stage shot has — plain stat matching never
     reproduces that because it is a look, not just a color statistic.

Heuristic classes stand in for the real M2 masks (Grounding DINO + SAM2):
  skin | red_warm (LED/stage) | chromatic_other | neutral
Once M2 lands, swap `classify()` for real masks; the transfer math is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import common
import face_detect
from color_transfer import _delta_e

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["skin", "red_warm", "chromatic_other", "neutral"]
# How strongly each class adopts the reference's own class look. >1.0 means we
# deliberately overshoot the reference's class mean — needed because feathered
# cross-class blending always dilutes the pure per-class shift back down a bit.
CLASS_STRENGTH = {"skin": 1.15, "red_warm": 1.35, "chromatic_other": 1.15, "neutral": 0.9}
# Skin hue must not rotate much (avoid cast); chroma/L can move (brighten/clean up).
SKIN_HUE_LOCK = 0.4
MIN_FRAC = 0.004  # a class must cover >0.4% in both images to be matched


def classify(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return mutually-exclusive float{0,1} masks per class.

    Skin is gated by real face detection (YuNet), NOT a bare color threshold —
    a plain YCbCr test flagged 25-34% of these event photos as "skin" (warm
    floor tiles, red LED bleed), which silently wrecked the per-class stats.
    """
    faces = face_detect.detect_faces(rgb)
    skin = face_detect.skin_mask_from_faces(rgb, faces) > 0.5
    hsv = common.rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_warm = (~skin) & (s > 0.30) & (v > 0.15) & ((h < 25) | (h > 335))
    chromatic = (~skin) & (~red_warm) & (s > 0.22) & (v > 0.08)
    neutral = (~skin) & (~red_warm) & (~chromatic)
    return {
        "skin": skin.astype(np.float32),
        "red_warm": red_warm.astype(np.float32),
        "chromatic_other": chromatic.astype(np.float32),
        "neutral": neutral.astype(np.float32),
    }


def _percentile_stats(vals: np.ndarray, lo: float = 2, hi: float = 98):
    p_lo, p_hi = np.percentile(vals, [lo, hi])
    return float(p_lo), float(max(p_hi, p_lo + 1e-3))


def _tone_curve_L(tgt_L: np.ndarray, t_mask: np.ndarray, r_L: np.ndarray, r_mask: np.ndarray) -> np.ndarray:
    """Contrast (slope) from the percentile spread, but anchored on the MEAN
    (not the low percentile) so the class average is guaranteed to land on the
    reference class average. Anchoring on p_lo made the mapped result drift
    below target whenever a crop/class had outlier-heavy tails (a few near-black
    or near-white pixels skew p_lo/p_hi without moving where most of the mass
    sits) — the previous version undershot brightness for exactly this reason.
    """
    t_sel, r_sel = tgt_L[t_mask > 0.5], r_L[r_mask > 0.5]
    t_lo, t_hi = _percentile_stats(t_sel)
    r_lo, r_hi = _percentile_stats(r_sel)
    scale = (r_hi - r_lo) / max(t_hi - t_lo, 1e-3)
    t_mean, r_mean = float(t_sel.mean()), float(r_sel.mean())
    return (tgt_L - t_mean) * scale + r_mean


def _reinhard_ab(tgt_ab, t_mean, t_std, r_mean, r_std):
    return (tgt_ab - t_mean) * (r_std / t_std) + r_mean


def _grade_class(c: str, tgt_lab, tm, ref_lab, rm) -> np.ndarray:
    t_sel, r_sel = tm > 0.5, rm > 0.5
    out = tgt_lab.copy()
    t_mean_ab = tgt_lab[t_sel][:, 1:3].mean(axis=0)
    t_std_ab = tgt_lab[t_sel][:, 1:3].std(axis=0) + 1e-5
    r_mean_ab = ref_lab[r_sel][:, 1:3].mean(axis=0)
    r_std_ab = ref_lab[r_sel][:, 1:3].std(axis=0) + 1e-5

    out[..., 0] = _tone_curve_L(tgt_lab[..., 0], tm, ref_lab[..., 0], rm)

    if c == "skin":
        # polar decompose a,b -> chroma (radius) + hue (angle); lock hue mostly.
        t_c = np.hypot(tgt_lab[..., 1], tgt_lab[..., 2])
        t_h = np.arctan2(tgt_lab[..., 2], tgt_lab[..., 1])
        t_c_mean = np.hypot(*t_mean_ab)
        r_c_mean = np.hypot(*r_mean_ab)
        t_c_std = float(np.hypot(*t_std_ab))
        r_c_std = float(np.hypot(*r_std_ab))
        matched_c = (t_c - t_c_mean) * (r_c_std / max(t_c_std, 1e-5)) + r_c_mean
        matched_c = np.clip(matched_c, 0.0, None)
        r_h_mean = float(np.arctan2(r_mean_ab[1], r_mean_ab[0]))
        t_h_mean = float(np.arctan2(t_mean_ab[1], t_mean_ab[0]))
        d = np.arctan2(np.sin(r_h_mean - t_h_mean), np.cos(r_h_mean - t_h_mean))
        matched_h = t_h + SKIN_HUE_LOCK * d
        out[..., 1] = matched_c * np.cos(matched_h)
        out[..., 2] = matched_c * np.sin(matched_h)
    else:
        out[..., 1] = _reinhard_ab(tgt_lab[..., 1], t_mean_ab[0], t_std_ab[0], r_mean_ab[0], r_std_ab[0])
        out[..., 2] = _reinhard_ab(tgt_lab[..., 2], t_mean_ab[1], t_std_ab[1], r_mean_ab[1], r_std_ab[1])
    return out


def _vibrance_contrast_sharpen(rgb: np.ndarray, skin_mask: np.ndarray,
                               vibrance: float = 0.24, contrast: float = 1.08,
                               sharpen_amount: float = 0.3) -> np.ndarray:
    """Global 'look' finishing pass — what a plain stat-matcher never adds.

    IMPORTANT: vibrance must be applied as a Lab CHROMA boost with L held
    fixed, not an HSV saturation multiply. HSV keeps V (max channel) fixed and
    lowers the min channel(s) to raise S — for warm/red hues that lowers G,
    which luminance weighs heavily (~0.715), so "more saturated" silently also
    means "darker/dirtier" in perceptual lightness. That was the actual bug
    that made the previous version look muddy.
    """
    lab = common.rgb_to_lab(rgb)
    L = lab[..., 0]
    a, b = lab[..., 1], lab[..., 2]
    C = np.hypot(a, b) + 1e-6
    c_norm = np.clip(C / 60.0, 0.0, 1.0)  # rough normalize, Lab chroma rarely exceeds ~80
    boost = vibrance * (1.0 - c_norm) * c_norm * 4.0  # boosts mid-chroma most
    boost *= (1.0 - 0.6 * skin_mask)  # gentler on skin
    new_C = C * (1.0 + boost)
    lab[..., 1] = a / C * new_C
    lab[..., 2] = b / C * new_C

    mid = 50.0
    lab[..., 0] = np.clip((L - mid) * contrast + mid, 0.0, 100.0)
    out = np.clip(common.lab_to_rgb(lab), 0.0, 1.0)

    if sharpen_amount > 0:
        from PIL import Image, ImageFilter
        im = Image.fromarray((out * 255 + 0.5).astype(np.uint8), mode="RGB")
        blurred = im.filter(ImageFilter.GaussianBlur(radius=2.0))
        sharp = np.asarray(im, dtype=np.float32) + sharpen_amount * (
            np.asarray(im, dtype=np.float32) - np.asarray(blurred, dtype=np.float32)
        )
        out = np.clip(sharp / 255.0, 0.0, 1.0)
    return out


def semantic_transfer(ref_rgb: np.ndarray, tgt_rgb: np.ndarray,
                      strength: float = 1.0, feather: float = 3.0,
                      finish: bool = True,
                      dump_masks: Path | None = None) -> np.ndarray:
    ref_lab = common.rgb_to_lab(ref_rgb)
    tgt_lab = common.rgb_to_lab(tgt_rgb)
    ref_cls = classify(ref_rgb)
    tgt_cls = classify(tgt_rgb)

    n_tgt = tgt_rgb.shape[0] * tgt_rgb.shape[1]
    n_ref = ref_rgb.shape[0] * ref_rgb.shape[1]

    feathered = {c: common.feather_mask(tgt_cls[c], radius=feather) for c in CLASSES}
    denom = np.sum([feathered[c] for c in CLASSES], axis=0) + 1e-6
    weights = {c: feathered[c] / denom for c in CLASSES}

    acc = np.zeros_like(tgt_lab)
    for c in CLASSES:
        tm, rm = tgt_cls[c], ref_cls[c]
        matched = (tm.sum() / n_tgt > MIN_FRAC) and (rm.sum() / n_ref > MIN_FRAC)
        if matched:
            graded = _grade_class(c, tgt_lab, tm, ref_lab, rm)
            cs = CLASS_STRENGTH[c] * strength
            graded = tgt_lab * (1.0 - cs) + graded * cs
        else:
            graded = tgt_lab
        acc += graded * weights[c][..., None]

    acc[..., 0] = np.clip(acc[..., 0], 0.0, 100.0)  # overshoot strengths (>1.0) can push L out of range
    out_rgb = np.clip(common.lab_to_rgb(acc), 0.0, 1.0)
    if finish:
        out_rgb = _vibrance_contrast_sharpen(out_rgb, tgt_cls["skin"])

    if dump_masks is not None:
        vis = np.zeros((*tgt_rgb.shape[:2], 3), dtype=np.float32)
        palette = {"skin": (1, 0.8, 0.6), "red_warm": (1, 0.2, 0.2),
                   "chromatic_other": (0.3, 0.6, 1), "neutral": (0.5, 0.5, 0.5)}
        for c in CLASSES:
            for k in range(3):
                vis[..., k] += tgt_cls[c] * palette[c][k]
        common.save_rgb(np.clip(vis, 0, 1), dump_masks)

    return out_rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "color_transfer"))
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    ref = common.load_rgb(args.ref, max_side=args.max_side)
    out_dir = Path(args.out_dir)
    for tp in args.targets:
        tgt = common.load_rgb(tp, max_side=args.max_side)
        stem = Path(tp).stem.split("-")[0]
        graded = semantic_transfer(ref, tgt, strength=args.strength,
                                   dump_masks=out_dir / f"{stem}_seg.jpg")
        common.save_rgb(graded, out_dir / f"{stem}_semantic.jpg")
        print(f"{stem}: ΔE_to_ref before={_delta_e(tgt, ref):.2f} "
              f"after={_delta_e(graded, ref):.2f} -> {stem}_semantic.jpg")
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
