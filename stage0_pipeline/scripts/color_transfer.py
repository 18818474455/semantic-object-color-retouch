"""Reference-based color transfer (仿色) — local, deterministic, no GPT.

Given a reference "look" image and target photos, transfer the reference color
statistics onto the targets in CIE Lab (Reinhard mean/std matching).

Two modes demonstrate the point made in the architecture discussion:
  --mode global  : match L,a,b mean/std globally (fast, but can 串色 when the
                   reference and target have different content composition)
  --mode chroma  : match only a,b (color) + gently match L contrast, preserving
                   the target's own exposure — safer for event delivery
Skin protection (--protect-skin) fades the transfer down on skin-hued pixels so
faces don't get dragged to the reference's cast.

This is the CHEAP path. The fitted shift can later be baked into a 3D LUT for the
Chroma engine (style_lut) so a whole album is graded identically at ~0 cost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import common

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lab_stats(lab: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    flat = lab.reshape(-1, 3)
    if weights is None:
        return flat.mean(axis=0), flat.std(axis=0) + 1e-5
    w = weights.reshape(-1, 1)
    wsum = w.sum() + 1e-8
    mean = (flat * w).sum(axis=0) / wsum
    var = ((flat - mean) ** 2 * w).sum(axis=0) / wsum
    return mean, np.sqrt(var) + 1e-5


def _skin_weight(rgb: np.ndarray) -> np.ndarray:
    """Rough skin likelihood in [0,1] from normalized RGB + YCbCr gate."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5
    skin = (cr > 0.53) & (cr < 0.64) & (cb > 0.42) & (cb < 0.52) & (y > 0.25) & (y < 0.95)
    return skin.astype(np.float32)


def transfer(
    ref_rgb: np.ndarray, tgt_rgb: np.ndarray, mode: str = "chroma",
    strength: float = 1.0, protect_skin: bool = True,
) -> np.ndarray:
    ref_lab = common.rgb_to_lab(ref_rgb)
    tgt_lab = common.rgb_to_lab(tgt_rgb)
    r_mean, r_std = _lab_stats(ref_lab)
    t_mean, t_std = _lab_stats(tgt_lab)

    out = tgt_lab.copy()
    # a,b (color) always matched
    for c in (1, 2):
        out[..., c] = (tgt_lab[..., c] - t_mean[c]) * (r_std[c] / t_std[c]) + r_mean[c]
    if mode == "global":
        out[..., 0] = (tgt_lab[..., 0] - t_mean[0]) * (r_std[0] / t_std[0]) + r_mean[0]
    elif mode == "chroma":
        # keep target exposure (mean L) but adopt reference contrast (std L), softly
        contrast_ratio = 0.5 + 0.5 * (r_std[0] / t_std[0])  # damp toward 1.0
        out[..., 0] = (tgt_lab[..., 0] - t_mean[0]) * contrast_ratio + t_mean[0]

    graded = common.lab_to_rgb(out)
    graded = np.clip(graded, 0.0, 1.0)

    # blend by strength
    w = np.full(tgt_rgb.shape[:2], strength, dtype=np.float32)
    if protect_skin:
        skin = common.feather_mask(_skin_weight(tgt_rgb), radius=4.0)
        w = w * (1.0 - 0.6 * skin)  # keep 40% transfer on skin at most
    w = w[..., None]
    return np.clip(tgt_rgb * (1.0 - w) + graded * w, 0.0, 1.0)


def _delta_e(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    la, lb = common.rgb_to_lab(a_rgb), common.rgb_to_lab(b_rgb)
    return float(np.sqrt(((la - lb) ** 2).sum(axis=-1)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--mode", choices=["global", "chroma"], default="chroma")
    ap.add_argument("--strength", type=float, default=0.9)
    ap.add_argument("--no-protect-skin", action="store_true")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "color_transfer"))
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    from build_review_sheet import make_comparison

    ref = common.load_rgb(args.ref, max_side=args.max_side)
    out_dir = Path(args.out_dir)
    for tp in args.targets:
        tgt = common.load_rgb(tp, max_side=args.max_side)
        graded = transfer(ref, tgt, mode=args.mode, strength=args.strength,
                          protect_skin=not args.no_protect_skin)
        stem = Path(tp).stem.split("-")[0]
        common.save_rgb(graded, out_dir / f"{stem}_{args.mode}.jpg")
        de_before = _delta_e(tgt, ref)
        de_after = _delta_e(graded, ref)
        make_comparison(
            [("original", tgt), (f"仿色 local/{args.mode}", graded), ("reference", ref)],
            out_dir / f"{stem}_compare.jpg", panel_w=560)
        print(f"{stem}: ΔE_to_ref before={de_before:.2f} after={de_after:.2f} "
              f"(closer={de_after < de_before})")
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
