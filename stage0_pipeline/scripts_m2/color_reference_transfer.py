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
from coherence_controller import compute_global_mood, apply_global_base

MIN_FRAC = 0.01

# Validated in the Phase B / Phase D 20-image sweep. "medium" is exactly the
# strength tier that passed the full regression + expanded-sample run.
# `default`/`skin`/`neutral` are consumed ONLY by the legacy pipeline
# (`_render_legacy_from_analysis`) and are untouched by C3.
#
# `global_base` (C3-1) and `region_default`/`region_skin`/`region_neutral`
# (C3-2) are consumed ONLY by the coherence pipeline
# (`_render_coherence_from_analysis`) — see coherence_controller.py and the
# residual-trust math below. They deliberately live in the SAME preset dicts
# as the legacy knobs (not a parallel table) purely so webdemo's slider
# interpolation (`_interp_preset` linearly blends every key it finds) keeps
# working for both pipelines without extra plumbing.
#
# The region_* caps are much lower than legacy's default/skin/neutral: the
# coherence pipeline multiplies its cap by a [0,1] trust score before use
# (region_strength = cap * trust), so the cap is a ceiling reached only when
# a region's pair/homogeneity/pixel/scene confidence are ALL high — legacy's
# cs_base is instead a flat per-tier constant regardless of confidence,
# which is the exact "confidence should gate strength toward 0, not toward
# 1" bug this upgrade exists to fix (see 语义物体调色专家-仿色一致性升级方案 §一).
STRENGTH_PRESETS = {
    "light":  {"default": 1.0, "skin": 0.9, "neutral": 0.25, "global_base": 0.15,
               "region_default": 1.05, "region_skin": 0.60, "region_neutral": 0.20,
               "vibrance": 0.22, "contrast": 1.06, "sharpen": 0.25},
    "medium": {"default": 1.6, "skin": 1.3, "neutral": 0.45, "global_base": 0.30,
               "region_default": 1.20, "region_skin": 0.75, "region_neutral": 0.35,
               "vibrance": 0.40, "contrast": 1.14, "sharpen": 0.40},
    "strong": {"default": 2.0, "skin": 1.5, "neutral": 0.55, "global_base": 0.45,
               "region_default": 1.25, "region_skin": 0.85, "region_neutral": 0.45,
               "vibrance": 0.50, "contrast": 1.18, "sharpen": 0.45},
}

# Full delta (before trust/weight scaling) between a region's stats-matched
# target and the global base, clamped per pixel. Prevents a degenerate
# region statistic (tiny mask, extreme std ratio) from injecting an
# arbitrarily large residual regardless of how trust is computed.
MAX_REGION_DELTA_E = 30.0

PIPELINE_LEGACY = "legacy"
PIPELINE_COHERENCE = "coherence"
SUPPORTED_PIPELINES = (PIPELINE_LEGACY, PIPELINE_COHERENCE)


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
    # C3-1: unmasked whole-image stats, consumed by coherence_controller's
    # global mood base. Older cached profile JSON files won't have this key;
    # compute_global_mood() treats that as "skip the global base", not an error.
    global_ab = ref_lab[..., 1:3].reshape(-1, 2)
    global_stats = {
        "l_mean": float(ref_lab[..., 0].mean()),
        "mean_ab": [float(global_ab[:, 0].mean()), float(global_ab[:, 1].mean())],
    }
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
    return {"name": name, "size": [int(ref_rgb.shape[1]), int(ref_rgb.shape[0])],
            "classes": classes, "global": global_stats}


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


def _grade_neutral_additive(tgt_lab: np.ndarray, tm: np.ndarray, ref_stats: dict) -> np.ndarray:
    """"neutral" (unrecognized/heterogeneous leftover content — e.g. a dense
    crowd's mixed-color clothing that no detector query matched) gets a
    constant per-channel ADDITIVE shift instead of the mean/std RESCALE
    _grade_class_from_stats uses for real semantic classes.

    Found via a Web Demo test (2026-07-10): trying to give a crowd its own
    matched "people" class and rescaling it toward the reference's mean/std
    like sky/building do washes every person's differently-colored clothing
    into one flat tint — a rescale forces convergence to ONE target
    statistic, which is fine for a visually uniform class (sky) but wrong for
    content whose whole visual identity IS "lots of different colors". An
    additive shift instead translates the whole region by a constant vector
    toward the reference's overall mood, leaving every pixel's relative hue/
    lightness difference from its neighbors completely untouched — same
    "nudge toward the reference" intent, without erasing local variety.
    render_from_analysis's existing cs blend still controls how much of this
    shift actually lands (medium neutral cs=0.45, taper for huge neutral_frac
    still applies on top).
    """
    t_sel = tm > 0.5
    out = tgt_lab.copy()
    if t_sel.sum() < 20:
        return out
    t_mean_ab = tgt_lab[t_sel][:, 1:3].mean(axis=0)
    t_mean_L = float(tgt_lab[..., 0][t_sel].mean())
    d_L = ref_stats["l_mean"] - t_mean_L
    d_a = ref_stats["mean_ab"][0] - t_mean_ab[0]
    d_b = ref_stats["mean_ab"][1] - t_mean_ab[1]
    out[..., 0] = tgt_lab[..., 0] + d_L
    out[..., 1] = tgt_lab[..., 1] + d_a
    out[..., 2] = tgt_lab[..., 2] + d_b
    return out


def _class_pair_confidence(ref_stats: dict, tgt_lab: np.ndarray, tm: np.ndarray) -> float:
    """Scalar confidence in [0,1]: does the reference's class actually look
    like a plausible target for THIS class as a whole, in absolute Lab terms?

    `_class_outlier_confidence` catches pixels atypical of their OWN class
    (a few stray leaves inside a "sky" mask) — it does nothing when EVERY
    target pixel is perfectly typical of the target's own class, but the
    target's class as a WHOLE looks nothing like the reference's same-named
    class. Found via a real case (2026-07-10): an open-vocab "building" query
    matches both a brightly-lit white mall ceiling (reference) and a dim
    grey/warm steel roof truss under mixed lighting (target) — same text
    label, physically very different things. Force-matching statistics at
    full/overshoot strength blew the truss out into an unnaturally bright
    cyan slab that visually "disconnected" from the barely-touched
    (correctly conservative) neutral crowd beneath it.

    Measures the raw absolute-Lab gap between the reference class's mean and
    the target class's OWN mean (i.e. before any grading) — deliberately NOT
    normalized by either class's internal spread, because a same-labeled
    class with a huge internal spread (e.g. "building" lumping bright
    ceiling + dark structural beams together) is itself a sign the label is
    unreliable, not a license to tolerate a bigger gap.
    """
    t_sel = tm > 0.5
    if t_sel.sum() < 20:
        return 1.0
    t_mean_L = float(tgt_lab[..., 0][t_sel].mean())
    t_mean_ab = tgt_lab[t_sel][:, 1:3].mean(axis=0)
    d_L = abs(ref_stats["l_mean"] - t_mean_L)
    d_ab = float(np.hypot(ref_stats["mean_ab"][0] - t_mean_ab[0], ref_stats["mean_ab"][1] - t_mean_ab[1]))
    # Calibrated against real photo pairs (see phase-teacher-class-mismatch-fix.md):
    # below ~15 Lab units the two same-labeled regions still plausibly read as
    # "the same kind of thing under different lighting"; past ~30 they're
    # probably different materials that only happen to share a text label.
    Z_FULL, Z_ZERO = 15.0, 30.0
    z = max(d_L, d_ab)
    return float(np.clip(1.0 - (z - Z_FULL) / (Z_ZERO - Z_FULL), 0.0, 1.0))


def _class_outlier_confidence(tgt_lab: np.ndarray, tm: np.ndarray) -> np.ndarray:
    """Per-pixel confidence in [0,1]: how well does THIS pixel's own Lab value
    match the (mean, std) of the class's own masked pixels? 1.0 = solidly
    inside the class's own distribution; tapers toward 0.0 for pixels far
    outside it (mixed/outlier/edge pixels — e.g. a few tree-branch pixels
    swept into a "sky" mask by a coarse SAM boundary, or an odd-colored
    window swept into a "building" mask).

    Found via a real bug (2026-07-10): STRENGTH_PRESETS deliberately uses
    cs > 1 ("overshoot", more than a full statistical match to the
    reference) for most classes at medium/strong, to make punchy results on
    genuinely homogeneous regions (a clean blue sky). But a flat per-class cs
    blended through a feathered mask boundary amplifies exactly the pixels
    LEAST representative of the class into a visible unnatural halo/cast —
    e.g. a bright cyan ring at a treeline-vs-sky boundary, or an oversaturated
    cast across a multi-material building facade. Verified: capping cs to 1.0
    (no overshoot at all) removes the artifact; this function lets overshoot
    stay on for pixels it's actually safe for instead of removing it globally.
    """
    t_sel = tm > 0.5
    if t_sel.sum() < 20:
        return np.ones(tgt_lab.shape[:2], dtype=np.float32)
    t_L, t_a, t_b = tgt_lab[..., 0], tgt_lab[..., 1], tgt_lab[..., 2]
    mean_L, std_L = float(t_L[t_sel].mean()), float(t_L[t_sel].std()) + 1e-5
    mean_a, std_a = float(t_a[t_sel].mean()), float(t_a[t_sel].std()) + 1e-5
    mean_b, std_b = float(t_b[t_sel].mean()), float(t_b[t_sel].std()) + 1e-5
    z = np.sqrt(((t_L - mean_L) / std_L) ** 2 + ((t_a - mean_a) / std_a) ** 2 + ((t_b - mean_b) / std_b) ** 2)
    # confidence=1.0 within 1.5 std of the class's own mean (still "typical"
    # for that class), linearly tapering to 0.0 by 3.5 std (clearly an
    # outlier/mixed pixel this class's statistics don't really describe).
    Z_FULL, Z_ZERO = 1.5, 3.5
    return np.clip(1.0 - (z - Z_FULL) / (Z_ZERO - Z_FULL), 0.0, 1.0).astype(np.float32)


def _region_homogeneity_confidence(tgt_lab: np.ndarray, tm: np.ndarray) -> float:
    """C3-2: scalar confidence in [0,1] — how visually uniform is this region
    ON ITS OWN, independent of how well it happens to line up with the
    reference? A region whose own pixels vary wildly in Lab (an open-vocab
    "building" mask that swept up a bright wall AND a dark support beam) is
    itself evidence the label lumped together different materials, and
    shouldn't be trusted with a single mean/std statistical match no matter
    how good `_class_pair_confidence`'s reference-vs-target mean comparison
    looks.

    Deliberately independent of `_class_outlier_confidence` (which flags
    individual pixels atypical of THIS region's own mean) — this is a
    single scene-level score for the region as a whole, feeding the same
    `trust = scene * pair * homogeneity * pixel` product the upgrade plan
    specifies (仿色一致性升级方案 §三·阶段三/四), not a per-pixel map.
    """
    t_sel = tm > 0.5
    if t_sel.sum() < 20:
        return 1.0
    l_std = float(tgt_lab[..., 0][t_sel].std())
    ab_std = tgt_lab[t_sel][:, 1:3].std(axis=0)
    spread = float(np.hypot(l_std, np.hypot(*ab_std)))
    # Loosely calibrated against the same "same label, different material"
    # scale _class_pair_confidence uses (Z_ZERO=30 Lab units of absolute
    # mean gap): a genuinely uniform sky/wall patch's own internal std
    # sits comfortably under 10; a region straddling two different
    # materials routinely exceeds 20.
    FULL, ZERO = 10.0, 22.0
    return float(np.clip(1.0 - (spread - FULL) / (ZERO - FULL), 0.0, 1.0))


def analyze_target(profile: dict, tgt_rgb: np.ndarray, feather: float = 4.0) -> dict:
    """Strength-INDEPENDENT half of apply_profile: segmentation, the content-
    match gate, feathered blend weights, and per-class graded Lab targets.

    Split out so a caller that wants to render the SAME (profile, target)
    pair at multiple strengths (e.g. an interactive strength slider in
    scripts_c1c/../webdemo) can pay the expensive part — Grounding DINO + SAM
    detection in build_classes(), plus the per-class Lab regrade — exactly
    once, then call render_from_analysis() repeatedly for near-instant
    re-renders. apply_profile() itself is unchanged in behavior; it now just
    calls these two halves in sequence.
    """
    tgt_lab = common.rgb_to_lab(tgt_rgb)
    tgt_cls = build_classes(tgt_rgb)
    compat = content_match_score(profile, tgt_cls)

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
    # own dynamic taper (see render_from_analysis) built specifically for the
    # "content not recognized" case, and skin comes from a reliable face
    # detector with its own hue-lock safety rather than a fuzzy text-label
    # match — both were already validated safe standalone across the
    # 20-image sweep.
    allow_class_transfer = compat["suitable"]

    matched_info = {}
    graded_by_class = {}
    confidence_by_class = {}
    class_confidence = {}
    for c in class_names:
        tm = tgt_cls.get(c)
        ref_stats = profile["classes"].get(c)
        t_frac = float(tm.mean()) if tm is not None else 0.0
        r_frac = float(ref_stats["frac"]) if ref_stats is not None else 0.0
        class_allowed = allow_class_transfer or c in ("neutral", "skin")
        matched = (class_allowed and tm is not None and ref_stats is not None
                  and t_frac > MIN_FRAC and r_frac > MIN_FRAC)
        matched_info[c] = {"tgt_frac": round(t_frac, 4), "ref_frac": round(r_frac, 4), "matched": matched}
        if not matched:
            graded_by_class[c] = tgt_lab
        elif c == "neutral":
            graded_by_class[c] = _grade_neutral_additive(tgt_lab, tm, ref_stats)
        else:
            graded_by_class[c] = _grade_class_from_stats(c, tgt_lab, tm, ref_stats)
        confidence_by_class[c] = _class_outlier_confidence(tgt_lab, tm) if matched else None
        # "neutral"/"skin" already have their own dedicated, deliberately-gentle
        # treatment (additive shift / hue-lock) — the class-pair mismatch guard
        # is specifically about detected-object classes (sky/building/floor/...)
        # whose reference stats can come from a same-labeled but physically
        # unrelated region.
        class_confidence[c] = (_class_pair_confidence(ref_stats, tgt_lab, tm)
                                if matched and c not in ("neutral", "skin") else 1.0)

    return {
        "profile": profile,
        "tgt_rgb": tgt_rgb, "tgt_lab": tgt_lab, "tgt_cls": tgt_cls, "compat": compat,
        "class_names": class_names, "weights": weights,
        "matched_info": matched_info, "graded_by_class": graded_by_class,
        "confidence_by_class": confidence_by_class, "class_confidence": class_confidence,
    }


def _render_legacy_from_analysis(analysis: dict, strength: str | dict = "medium") -> np.ndarray:
    """Strength-DEPENDENT half of apply_profile: blend the precomputed
    per-class graded targets (analyze_target) using a strength preset, then
    the global vibrance/contrast/sharpen finishing pass. No segmentation or
    per-class regrading happens here — this is the cheap call an interactive
    strength slider re-runs on every drag. `strength` may be a preset name
    ("light"/"medium"/"strong") or a raw preset dict (e.g. interpolated
    between two presets for a continuous slider).

    cs > 1 ("overshoot") is damped per-pixel by each class's outlier
    confidence map (see _class_outlier_confidence) so it stays strong on
    pixels that genuinely look like the rest of the class, and fades out on
    pixels that don't — fixes the treeline color-halo / building-cast bug
    (2026-07-10) without touching the validated preset numbers themselves.
    """
    preset = STRENGTH_PRESETS[strength] if isinstance(strength, str) else strength
    tgt_lab = analysis["tgt_lab"]
    acc = np.zeros_like(tgt_lab)
    for c in analysis["class_names"]:
        info = analysis["matched_info"][c]
        graded = analysis["graded_by_class"][c]
        if info["matched"]:
            cs_base = preset["skin"] if c == "skin" else (preset["neutral"] if c == "neutral" else preset["default"])
            if cs_base > 1.0:
                confidence = analysis["confidence_by_class"][c] * analysis["class_confidence"][c]
                cs = 1.0 + (cs_base - 1.0) * confidence  # array, per-pixel damped overshoot
            else:
                cs = cs_base  # <=1 is a plain undershoot/no-op blend, nothing to damp
            if c == "neutral" and info["tgt_frac"] > 0.5:
                # Unrecognized leftover swallowing most of the frame means the
                # segmentation didn't understand this scene — taper toward a
                # no-op instead of forcing the reference's global cast on it
                # (this is the concrete "orange-wash on DAP02394_2" bug fix).
                cs = cs * (max(0.0, 1.0 - (info["tgt_frac"] - 0.5) / 0.5) ** 2)
            cs_bcast = cs[..., None] if isinstance(cs, np.ndarray) else cs
            graded = tgt_lab * (1.0 - cs_bcast) + graded * cs_bcast
        acc += graded * analysis["weights"][c][..., None]

    acc[..., 0] = np.clip(acc[..., 0], 0.0, 100.0)
    out_rgb = np.clip(common.lab_to_rgb(acc), 0.0, 1.0)
    out_rgb = _vibrance_contrast_sharpen(out_rgb, analysis["tgt_cls"]["skin"], vibrance=preset["vibrance"],
                                          contrast=preset["contrast"], sharpen_amount=preset["sharpen"])
    return out_rgb


def _render_coherence_from_analysis(analysis: dict, strength: str | dict = "medium") -> np.ndarray:
    """C3-2 coherence renderer: global mood base (C3-1) PLUS a trust-gated
    per-region RESIDUAL, replacing C3-1's "global base only" placeholder.

    Follows 仿色一致性升级方案 §二/三 literally:

        base_lab = apply_global_base(target_lab, global_mood)
        regional_target = grade_region(base_lab, reference_stats)   # graded
                                                                     # FROM the
                                                                     # base, not
                                                                     # the raw
                                                                     # target
        regional_delta = regional_target - base_lab
        output = base_lab + trust * clamp(regional_delta)

    where ``trust = scene_confidence * pair_confidence *
    homogeneity_confidence * pixel_confidence`` and ``region_strength =
    preset_cap * trust`` — the SAME confidence terms legacy already computes
    (`_class_pair_confidence`, `_class_outlier_confidence`), now gating the
    entire residual rather than just legacy's `cs > 1` overshoot fraction.
    A same-labeled-but-different-material region (confidence -> 0) now
    converges toward "leave it at the global base", not toward "still do a
    full statistical match" — that asymmetry was the concrete bug this
    upgrade targets (see the module docstring reference in the plan doc).

    Grading against `base_lab` (already mood-shifted) rather than the raw
    target means the residual only has to cover what the global step didn't
    already close, instead of re-deriving the reference's absolute stats
    from scratch and double-counting the mood shift.
    """
    preset = STRENGTH_PRESETS[strength] if isinstance(strength, str) else strength
    tgt_lab = analysis["tgt_lab"]
    profile = analysis["profile"]

    mood = compute_global_mood(profile, tgt_lab, analysis["compat"], preset.get("global_base", 0.0))
    base_lab = apply_global_base(tgt_lab, mood, analysis["tgt_cls"].get("skin"))

    scene_confidence = float(np.clip(analysis["compat"].get("explainable_tgt_frac", 0.0), 0.0, 1.0))
    acc_delta = np.zeros_like(base_lab)
    for c in analysis["class_names"]:
        info = analysis["matched_info"][c]
        if not info["matched"]:
            continue
        tm = analysis["tgt_cls"].get(c)
        ref_stats = profile["classes"].get(c)

        if c == "neutral":
            graded = _grade_neutral_additive(base_lab, tm, ref_stats)
        else:
            graded = _grade_class_from_stats(c, base_lab, tm, ref_stats)
        delta = graded - base_lab
        delta_mag = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-6
        delta = delta * np.minimum(1.0, MAX_REGION_DELTA_E / delta_mag)

        # "neutral"/"skin" keep their own dedicated treatment (additive
        # shift / hue-lock) — the same-label-different-material guard is
        # specifically about open-vocab detected classes.
        if c in ("neutral", "skin"):
            pair_conf = 1.0
        else:
            # Recomputed against base_lab (not the precomputed tgt_lab-based
            # analysis["class_confidence"]): the global shift may already
            # have closed part of the reference-vs-target absolute gap, so
            # the residual's own trust should judge the gap that's actually
            # left, not the pre-global-shift gap.
            pair_conf = _class_pair_confidence(ref_stats, base_lab, tm)
        homog_conf = _region_homogeneity_confidence(tgt_lab, tm)
        pixel_conf = analysis["confidence_by_class"][c]
        if pixel_conf is None:
            pixel_conf = np.ones(base_lab.shape[:2], dtype=np.float32)
        trust = scene_confidence * pair_conf * homog_conf * pixel_conf

        cap = preset["region_skin"] if c == "skin" else (preset["region_neutral"] if c == "neutral" else preset["region_default"])
        region_strength = cap * trust
        if c == "neutral" and info["tgt_frac"] > 0.5:
            # Same "segmentation didn't understand this scene" taper legacy
            # uses: unrecognized leftover swallowing most of the frame
            # shouldn't get the reference's cast forced onto it.
            region_strength = region_strength * (max(0.0, 1.0 - (info["tgt_frac"] - 0.5) / 0.5) ** 2)

        w = analysis["weights"][c]
        acc_delta += delta * (region_strength * w)[..., None]

    out_lab = base_lab + acc_delta
    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out_rgb = np.clip(common.lab_to_rgb(out_lab), 0.0, 1.0)
    out_rgb = _vibrance_contrast_sharpen(out_rgb, analysis["tgt_cls"]["skin"], vibrance=preset["vibrance"],
                                          contrast=preset["contrast"], sharpen_amount=preset["sharpen"])
    return out_rgb


def render_from_analysis(analysis: dict, strength: str | dict = "medium",
                         pipeline: str = PIPELINE_LEGACY) -> np.ndarray:
    """Render an analyzed pair through an explicitly versioned pipeline.

    ``legacy`` is the C3-0-frozen previously validated implementation.
    ``coherence`` is the C3-1 global-mood-base renderer (regional residuals
    land in C3-2 — see _render_coherence_from_analysis's docstring).
    """
    if pipeline == PIPELINE_LEGACY:
        return _render_legacy_from_analysis(analysis, strength=strength)
    if pipeline == PIPELINE_COHERENCE:
        return _render_coherence_from_analysis(analysis, strength=strength)
    raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {SUPPORTED_PIPELINES}")


def apply_profile(profile: dict, tgt_rgb: np.ndarray, strength: str = "medium",
                  feather: float = 4.0, pipeline: str = PIPELINE_LEGACY
                  ) -> tuple[np.ndarray, dict, dict]:
    analysis = analyze_target(profile, tgt_rgb, feather=feather)
    out_rgb = render_from_analysis(analysis, strength=strength, pipeline=pipeline)
    return out_rgb, analysis["matched_info"], analysis["compat"]


def transfer(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, strength: str = "medium",
             pipeline: str = PIPELINE_LEGACY) -> tuple[np.ndarray, dict, dict]:
    """Convenience one-shot call: build + apply a profile in one step."""
    profile = compute_style_profile(ref_rgb)
    return apply_profile(profile, tgt_rgb, strength=strength, pipeline=pipeline)


def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic color-reference transfer (仿色)")
    ap.add_argument("--ref", help="reference image path (builds a fresh style profile)")
    ap.add_argument("--profile-in", help="load a previously saved style profile JSON instead of --ref")
    ap.add_argument("--profile-out", help="save the computed style profile to this JSON path")
    ap.add_argument("--tgt", nargs="+", required=True, help="one or more target image paths")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strength", choices=list(STRENGTH_PRESETS), default="medium")
    ap.add_argument("--pipeline", choices=SUPPORTED_PIPELINES, default=PIPELINE_LEGACY,
                    help="legacy is the frozen Teacher v0; coherence is reserved until C3-1/C3-2")
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
        out_rgb, matched_info, compat = apply_profile(
            profile, tgt_rgb, strength=args.strength, pipeline=args.pipeline
        )
        stem = Path(tp).stem.replace(" ", "_")
        out_path = out_dir / f"{stem}_{args.pipeline}_{args.strength}.jpg"
        common.save_rgb(out_rgb, out_path)
        print(f"{stem}: compat={compat} matched={[c for c, v in matched_info.items() if v['matched']]}")
        if not compat["suitable"]:
            print(f"  SKIPPED transfer (reference doesn't suit this photo): "
                  f"jaccard={compat['jaccard']}, explainable={compat['explainable_tgt_frac']}")
        print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
