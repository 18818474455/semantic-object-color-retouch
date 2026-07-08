"""Canonical color-reference-transfer ("仿色") pipeline — Phase A productization
on top of the Phase B (Grounding DINO + SAM segmentation) validation.

Supersedes `semantic_transfer_v2.py` (kept for the historical Phase-D/expanded
test harness). Adds three things Phase A asked for:

1. Reusable style profile: analyze the reference image ONCE, cache the
   per-class Lab statistics to a small JSON file, then apply that profile to
   any number of target photos without re-running segmentation on the
   reference every time.
2. Discrete strength presets (light/medium/strong) instead of hand-edited
   magic numbers, so callers pick a tier rather than tuning floats.
3. A content-match gate: before/while applying, report how much of the
   target image is actually "explainable" by classes the reference also
   has, so a caller can warn "this reference doesn't really suit this
   photo" instead of forcing a transfer on unrelated content.

Must run under .venv-m2 (needs torch + transformers for detection; the
grading math itself is plain numpy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import face_detect
from semantic_color_transfer import _percentile_stats, _vibrance_contrast_sharpen, SKIN_HUE_LOCK
from region_provider_v2 import detect_classes

MIN_FRAC = 0.01

# Validated in the Phase B / Phase D 20-image sweep. "medium" is exactly the
# strength tier that passed the full regression + expanded-sample run.
STRENGTH_PRESETS = {
    "light":  {"default": 1.0, "skin": 0.9, "neutral": 0.25,
               "vibrance": 0.22, "contrast": 1.06, "sharpen": 0.25},
    "medium": {"default": 1.6, "skin": 1.3, "neutral": 0.45,
               "vibrance": 0.40, "contrast": 1.14, "sharpen": 0.40},
    "strong": {"default": 2.0, "skin": 1.5, "neutral": 0.55,
               "vibrance": 0.50, "contrast": 1.18, "sharpen": 0.45},
}


def build_classes(rgb: np.ndarray) -> dict[str, np.ndarray]:
    faces = face_detect.detect_faces(rgb)
    skin = face_detect.skin_mask_from_faces(rgb, faces)
    detected = detect_classes(rgb)
    classes = {"skin": skin}
    for label, m in detected.items():
        key = label.lower().strip()
        if not key:
            continue
        classes[key] = np.maximum(classes.get(key, 0), m)
    covered = np.clip(sum(classes.values()), 0.0, 1.0)
    classes["neutral"] = 1.0 - covered
    return classes


def compute_style_profile(ref_rgb: np.ndarray, name: str = "reference") -> dict:
    """Analyze a reference image once; the result is fully JSON-serializable
    and reusable for any number of future target photos."""
    ref_lab = common.rgb_to_lab(ref_rgb)
    ref_cls = build_classes(ref_rgb)
    classes = {}
    for c, rm in ref_cls.items():
        r_sel = rm > 0.5
        frac = float(rm.mean())
        if r_sel.sum() < 20 or frac <= MIN_FRAC:
            continue
        r_ab = ref_lab[r_sel][:, 1:3]
        r_L = ref_lab[..., 0][r_sel]
        l_lo, l_hi = _percentile_stats(r_L)
        mean_ab = r_ab.mean(axis=0)
        std_ab = r_ab.std(axis=0)
        classes[c] = {
            "frac": round(frac, 4),
            "mean_ab": [float(mean_ab[0]), float(mean_ab[1])],
            "std_ab": [float(std_ab[0]), float(std_ab[1])],
            "l_lo": float(l_lo), "l_hi": float(l_hi), "l_mean": float(r_L.mean()),
            "c_std": float(np.hypot(*std_ab)),
            "h_mean": float(np.arctan2(mean_ab[1], mean_ab[0])),
        }
    return {"name": name, "size": [int(ref_rgb.shape[1]), int(ref_rgb.shape[0])], "classes": classes}


def save_profile(profile: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile, ensure_ascii=False, indent=2))


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def content_match_score(profile: dict, tgt_cls: dict[str, np.ndarray]) -> dict:
    """How well does this reference's content actually cover what's in the
    target photo? Low score = 'this reference probably isn't a good fit'."""
    ref_classes = set(profile["classes"])
    tgt_present = {c: float(m.mean()) for c, m in tgt_cls.items() if float(m.mean()) > MIN_FRAC}
    tgt_classes = set(tgt_present)
    shared = ref_classes & tgt_classes
    union = ref_classes | tgt_classes
    jaccard = len(shared) / max(len(union), 1)
    explainable = sum(tgt_present[c] for c in shared if c != "neutral")
    return {
        "jaccard": round(jaccard, 3),
        "explainable_tgt_frac": round(explainable, 3),
        "shared_classes": sorted(shared - {"neutral"}),
        "suitable": bool(jaccard >= 0.15 and explainable >= 0.15),
    }


def _grade_class_from_stats(c: str, tgt_lab: np.ndarray, tm: np.ndarray, ref_stats: dict) -> np.ndarray:
    """Same math as semantic_color_transfer._grade_class, but reading
    reference-side statistics from a cached profile instead of recomputing
    them from a loaded reference image + mask every call."""
    t_sel = tm > 0.5
    out = tgt_lab.copy()
    t_ab = tgt_lab[t_sel][:, 1:3]
    t_mean_ab = t_ab.mean(axis=0)
    t_std_ab = t_ab.std(axis=0) + 1e-5
    r_mean_ab = np.array(ref_stats["mean_ab"])
    r_std_ab = np.array(ref_stats["std_ab"]) + 1e-5

    t_L = tgt_lab[..., 0]
    t_sel_L = t_L[t_sel]
    t_lo, t_hi = _percentile_stats(t_sel_L)
    r_lo, r_hi = ref_stats["l_lo"], ref_stats["l_hi"]
    scale = (r_hi - r_lo) / max(t_hi - t_lo, 1e-3)
    t_mean = float(t_sel_L.mean())
    out[..., 0] = (t_L - t_mean) * scale + ref_stats["l_mean"]

    if c == "skin":
        t_c = np.hypot(tgt_lab[..., 1], tgt_lab[..., 2])
        t_h = np.arctan2(tgt_lab[..., 2], tgt_lab[..., 1])
        t_c_mean = np.hypot(*t_mean_ab)
        r_c_mean = np.hypot(*r_mean_ab)
        t_c_std = float(np.hypot(*t_std_ab))
        r_c_std = ref_stats["c_std"]
        matched_c = np.clip((t_c - t_c_mean) * (r_c_std / max(t_c_std, 1e-5)) + r_c_mean, 0.0, None)
        t_h_mean = float(np.arctan2(t_mean_ab[1], t_mean_ab[0]))
        d = np.arctan2(np.sin(ref_stats["h_mean"] - t_h_mean), np.cos(ref_stats["h_mean"] - t_h_mean))
        matched_h = t_h + SKIN_HUE_LOCK * d
        out[..., 1] = matched_c * np.cos(matched_h)
        out[..., 2] = matched_c * np.sin(matched_h)
    else:
        out[..., 1] = (tgt_lab[..., 1] - t_mean_ab[0]) * (r_std_ab[0] / t_std_ab[0]) + r_mean_ab[0]
        out[..., 2] = (tgt_lab[..., 2] - t_mean_ab[1]) * (r_std_ab[1] / t_std_ab[1]) + r_mean_ab[1]
    return out


def apply_profile(profile: dict, tgt_rgb: np.ndarray, strength: str = "medium",
                  feather: float = 4.0) -> tuple[np.ndarray, dict, dict]:
    preset = STRENGTH_PRESETS[strength]
    tgt_lab = common.rgb_to_lab(tgt_rgb)
    tgt_cls = build_classes(tgt_rgb)
    compat = content_match_score(profile, tgt_cls)

    matched_info = {}
    class_names = set(tgt_cls) | set(profile["classes"]) | {"neutral"}
    feathered = {c: common.feather_mask(tgt_cls.get(c, np.zeros(tgt_rgb.shape[:2], np.float32)), radius=feather)
                for c in class_names}
    denom = np.sum(list(feathered.values()), axis=0) + 1e-6
    weights = {c: feathered[c] / denom for c in class_names}

    # Hard gate, not just an advisory flag: a per-class label match (same
    # string "clothing") does NOT mean the actual garments look alike. Found
    # by testing exactly this — an outdoor-crowd reference's "clothing" stats
    # got forced onto an unrelated portrait's dark coat and turned it ghost
    # white. When the whole-image content doesn't actually match the
    # reference, block every SPECIFIC labeled class instead of trusting the
    # label name alone, per the project's standing rule: low confidence ->
    # keep, don't guess.
    # "neutral" and "skin" are exempt from this gate: neutral already has its
    # own dynamic taper (see below) built specifically for the "content not
    # recognized" case, and skin comes from a reliable face detector with its
    # own hue-lock safety rather than a fuzzy text-label match — both were
    # already validated safe standalone across the 20-image sweep.
    allow_class_transfer = compat["suitable"]

    acc = np.zeros_like(tgt_lab)
    for c in class_names:
        tm = tgt_cls.get(c)
        ref_stats = profile["classes"].get(c)
        t_frac = float(tm.mean()) if tm is not None else 0.0
        r_frac = float(ref_stats["frac"]) if ref_stats is not None else 0.0
        class_allowed = allow_class_transfer or c in ("neutral", "skin")
        matched = (class_allowed and tm is not None and ref_stats is not None
                  and t_frac > MIN_FRAC and r_frac > MIN_FRAC)
        matched_info[c] = {"tgt_frac": round(t_frac, 4), "ref_frac": round(r_frac, 4), "matched": matched}
        if matched:
            graded = _grade_class_from_stats(c, tgt_lab, tm, ref_stats)
            cs = preset["skin"] if c == "skin" else (preset["neutral"] if c == "neutral" else preset["default"])
            if c == "neutral" and t_frac > 0.5:
                # Unrecognized leftover swallowing most of the frame means the
                # segmentation didn't understand this scene — taper toward a
                # no-op instead of forcing the reference's global cast on it
                # (this is the concrete "orange-wash on DAP02394_2" bug fix).
                cs *= max(0.0, 1.0 - (t_frac - 0.5) / 0.5) ** 2
            graded = tgt_lab * (1.0 - cs) + graded * cs
        else:
            graded = tgt_lab
        acc += graded * weights[c][..., None]

    acc[..., 0] = np.clip(acc[..., 0], 0.0, 100.0)
    out_rgb = np.clip(common.lab_to_rgb(acc), 0.0, 1.0)
    out_rgb = _vibrance_contrast_sharpen(out_rgb, tgt_cls["skin"], vibrance=preset["vibrance"],
                                          contrast=preset["contrast"], sharpen_amount=preset["sharpen"])
    return out_rgb, matched_info, compat


def transfer(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, strength: str = "medium") -> tuple[np.ndarray, dict, dict]:
    """Convenience one-shot call: build + apply a profile in one step."""
    profile = compute_style_profile(ref_rgb)
    return apply_profile(profile, tgt_rgb, strength=strength)


def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic color-reference transfer (仿色)")
    ap.add_argument("--ref", help="reference image path (builds a fresh style profile)")
    ap.add_argument("--profile-in", help="load a previously saved style profile JSON instead of --ref")
    ap.add_argument("--profile-out", help="save the computed style profile to this JSON path")
    ap.add_argument("--tgt", nargs="+", required=True, help="one or more target image paths")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strength", choices=list(STRENGTH_PRESETS), default="medium")
    args = ap.parse_args()

    if not args.ref and not args.profile_in:
        ap.error("need --ref or --profile-in")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_in:
        profile = load_profile(args.profile_in)
    else:
        ref_rgb = common.load_rgb(args.ref, max_side=1024)
        profile = compute_style_profile(ref_rgb, name=Path(args.ref).stem)
        if args.profile_out:
            save_profile(profile, args.profile_out)
            print(f"style profile saved -> {args.profile_out}")

    for tp in args.tgt:
        tgt_rgb = common.load_rgb(tp, max_side=1024)
        out_rgb, matched_info, compat = apply_profile(profile, tgt_rgb, strength=args.strength)
        stem = Path(tp).stem.replace(" ", "_")
        out_path = out_dir / f"{stem}_{args.strength}.jpg"
        common.save_rgb(out_rgb, out_path)
        print(f"{stem}: compat={compat} matched={[c for c, v in matched_info.items() if v['matched']]}")
        if not compat["suitable"]:
            print(f"  SKIPPED transfer (reference doesn't suit this photo): "
                  f"jaccard={compat['jaccard']}, explainable={compat['explainable_tgt_frac']}")
        print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
