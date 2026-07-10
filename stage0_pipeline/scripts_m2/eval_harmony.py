"""C3-4: automated "foreground/background harmony" diagnostic metrics.

仿色一致性升级方案 §四 is explicit these are a REGRESSION GATE, not a
substitute for the `FG-BG-Coord-v1` human review: a metric can look fine on
a photo that's subjectively ugly, and vice versa. Their job is (a) flag an
obvious numeric outlier automatically so a human reviewer knows to look
closer, and (b) give quick before/after evidence when comparing
`legacy`/`coherence` on the same pair without re-deriving it by eye every
time.

All metrics are computed purely from (analysis, out_rgb) — the SAME
`analyze_target()` output is reused for both pipelines being compared, so a
legacy-vs-coherence diff isn't confounded by different segmentation/weight
definitions between the two calls.
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
from build_review_sheet import make_comparison
from color_reference_transfer import (
    PIPELINE_LEGACY,
    PIPELINE_COHERENCE,
    SUPPORTED_PIPELINES,
    analyze_target,
    compute_style_profile,
    render_from_analysis,
)

# Matches region_provider_v2.TEXT_QUERIES's vocabulary + face_detect's "skin"
# + build_classes' "neutral" fallback. "neutral" is deliberately grouped with
# the person-ish foreground classes: per the earlier crowd-clothing bug, an
# unrecognized dense crowd falls into "neutral" and IS foreground content,
# not background.
FOREGROUND_CLASSES = {"skin", "clothing", "neutral"}
BACKGROUND_CLASSES = {"sky", "grass", "tree", "water", "led screen", "led wall",
                      "stage backdrop", "spotlight", "building", "floor", "flag"}

REVIEW_TEMPLATE = {
    "preferred": None,  # "legacy_v0" | "coherence_v1" | "tie"
    "scores": {
        "foreground_change_visible": None,
        "background_strength_natural": None,
        "fg_bg_same_tone": None,
        "skin_natural": None,
        "halo_free": None,
        "local_dirty_color_free": None,
        "delivery_willingness": None,
    },
    "severe": {
        "severe_fg_bg_disconnect": None,
        "severe_halo": None,
        "severe_skin_error": None,
    },
    "notes": "",
}


def _delta_e(delta_lab: np.ndarray) -> np.ndarray:
    return np.sqrt((delta_lab ** 2).sum(axis=-1))


def _weighted_mean_delta(delta_lab: np.ndarray, weight: np.ndarray) -> np.ndarray | None:
    total = float(weight.sum())
    if total < 1.0:
        return None
    return (delta_lab * weight[..., None]).sum(axis=(0, 1)) / total


def _group_weight(weights: dict[str, np.ndarray], class_names, group: set[str]) -> np.ndarray | None:
    present = [c for c in class_names if c in group and c in weights]
    if not present:
        return None
    return np.sum([weights[c] for c in present], axis=0)


def _boundary_mask(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels where SOME class's blend weight changes sharply — a mask
    boundary, whichever feathering method produced these weights (legacy's
    fixed-radius Gaussian or coherence's guided filter)."""
    import cv2
    grad_total = None
    for w in weights.values():
        gx = cv2.Sobel(w.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(w.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        grad_total = mag if grad_total is None else grad_total + mag
    if grad_total is None or grad_total.max() < 1e-6:
        shape = next(iter(weights.values())).shape
        return np.zeros(shape, dtype=bool)
    thresh = max(float(np.percentile(grad_total, 90)), 1e-3)
    return grad_total > thresh


def compute_harmony_metrics(analysis: dict, out_rgb: np.ndarray) -> dict:
    """All metrics 仿色一致性升级方案 §四 lists, computed from one rendered
    output against the shared `analysis` (tgt_lab/tgt_cls/weights/
    class_confidence) produced once by `analyze_target`."""
    tgt_lab = analysis["tgt_lab"]
    weights = analysis["weights"]
    class_names = analysis["class_names"]
    out_lab = common.rgb_to_lab(out_rgb)
    delta = out_lab - tgt_lab
    de = _delta_e(delta)

    metrics: dict = {}

    fg_w = _group_weight(weights, class_names, FOREGROUND_CLASSES)
    bg_w = _group_weight(weights, class_names, BACKGROUND_CLASSES)
    fg_delta = _weighted_mean_delta(delta, fg_w) if fg_w is not None else None
    bg_delta = _weighted_mean_delta(delta, bg_w) if bg_w is not None else None

    if fg_delta is not None and bg_delta is not None:
        metrics["fg_bg_luma_change_diff"] = float(abs(fg_delta[0] - bg_delta[0]))
        fg_ab, bg_ab = fg_delta[1:], bg_delta[1:]
        fg_mag, bg_mag = float(np.hypot(*fg_ab)), float(np.hypot(*bg_ab))
        # Direction consistency is only meaningful once both groups actually
        # moved a non-trivial amount; two near-zero vectors have an
        # arbitrary, noise-dominated "direction".
        metrics["fg_bg_tone_direction_cos"] = (
            float(np.dot(fg_ab, bg_ab) / (fg_mag * bg_mag)) if fg_mag > 0.3 and bg_mag > 0.3 else None
        )
    else:
        metrics["fg_bg_luma_change_diff"] = None
        metrics["fg_bg_tone_direction_cos"] = None

    neutral_w = weights.get("neutral")
    neutral_delta = _weighted_mean_delta(delta, neutral_w) if neutral_w is not None else None
    if neutral_delta is not None and bg_delta is not None:
        n_mag = float(np.hypot(*neutral_delta[1:]))
        b_mag = float(np.hypot(*bg_delta[1:]))
        metrics["neutral_vs_bg_change_ratio"] = n_mag / (b_mag + 1e-3)
    else:
        metrics["neutral_vs_bg_change_ratio"] = None

    boundary = _boundary_mask(weights)
    if boundary.sum() > 20:
        b_vals = de[boundary]
        metrics["boundary_delta_e_p95"] = float(np.percentile(b_vals, 95))
        metrics["boundary_delta_e_p99"] = float(np.percentile(b_vals, 99))
    else:
        metrics["boundary_delta_e_p95"] = None
        metrics["boundary_delta_e_p99"] = None

    skin_mask = analysis["tgt_cls"].get("skin")
    if skin_mask is not None and (skin_mask > 0.5).sum() > 20:
        sel = skin_mask > 0.5
        h0 = np.arctan2(tgt_lab[..., 2][sel], tgt_lab[..., 1][sel])
        h1 = np.arctan2(out_lab[..., 2][sel], out_lab[..., 1][sel])
        dh = np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
        metrics["skin_hue_drift_deg"] = float(np.degrees(np.abs(dh).mean()))
    else:
        metrics["skin_hue_drift_deg"] = None

    metrics["highlight_clip_frac"] = float((out_rgb >= 0.999).any(axis=-1).mean())
    metrics["shadow_clip_frac"] = float((out_rgb <= 0.001).any(axis=-1).mean())

    region_max_delta_e = {}
    region_pair_confidence = {}
    for c in class_names:
        w = weights.get(c)
        info = analysis["matched_info"].get(c, {})
        if w is None or not info.get("matched"):
            continue
        sel = w > 0.5
        if sel.sum() < 20:
            continue
        region_max_delta_e[c] = float(np.percentile(de[sel], 95))
        if c not in ("neutral", "skin"):
            region_pair_confidence[c] = analysis["class_confidence"].get(c)
    metrics["region_max_delta_e_p95"] = region_max_delta_e
    metrics["region_pair_confidence"] = region_pair_confidence

    return metrics


def _flag_summary(metrics: dict) -> list[str]:
    """Cheap, conservative "look at this one" flags — deliberately not a
    pass/fail gate, just pointers for the human reviewer."""
    flags = []
    if metrics.get("fg_bg_luma_change_diff") is not None and metrics["fg_bg_luma_change_diff"] > 15:
        flags.append("fg_bg_luma_change_diff>15 (前后景明暗变化差异大)")
    cos = metrics.get("fg_bg_tone_direction_cos")
    if cos is not None and cos < 0:
        flags.append("fg_bg_tone_direction_cos<0 (前后景色调朝相反方向变化)")
    if metrics.get("boundary_delta_e_p99") is not None and metrics["boundary_delta_e_p99"] > 25:
        flags.append("boundary_delta_e_p99>25 (边界可能有光晕)")
    if metrics.get("skin_hue_drift_deg") is not None and metrics["skin_hue_drift_deg"] > 15:
        flags.append("skin_hue_drift_deg>15 (肤色色相漂移较大)")
    if metrics.get("highlight_clip_frac", 0) > 0.05:
        flags.append("highlight_clip_frac>5% (高光裁剪偏多)")
    return flags


def run_pair(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, strength: str = "medium",
            sample_id: str = "sample") -> dict:
    profile = compute_style_profile(ref_rgb, name=sample_id)
    analysis = analyze_target(profile, tgt_rgb)
    result = {"id": sample_id, "compat": analysis["compat"]}
    outputs = {}
    for pipeline, key in ((PIPELINE_LEGACY, "legacy_v0"), (PIPELINE_COHERENCE, "coherence_v1")):
        out_rgb = render_from_analysis(analysis, strength=strength, pipeline=pipeline)
        metrics = compute_harmony_metrics(analysis, out_rgb)
        result[key] = metrics
        result[f"{key}_flags"] = _flag_summary(metrics)
        outputs[key] = out_rgb
    return result, outputs, analysis


def run_manifest(manifest_path: Path, out_root: Path, strength: str) -> None:
    lines = [l for l in manifest_path.read_text().splitlines() if l.strip()]
    ok, skipped = 0, 0
    for line in lines:
        rec = json.loads(line)
        sid = rec["id"]
        ref_path, tgt_path = Path(rec["reference_path"]), Path(rec["target_path"])
        if not ref_path.exists() or not tgt_path.exists():
            print(f"[skip] {sid}: source missing ({ref_path if not ref_path.exists() else tgt_path}) "
                  f"— is the external volume mounted?")
            skipped += 1
            continue
        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_rgb = common.load_rgb(str(ref_path), max_side=1024)
        tgt_rgb = common.load_rgb(str(tgt_path), max_side=1024)
        result, outputs, _ = run_pair(ref_rgb, tgt_rgb, strength=strength, sample_id=sid)

        common.save_rgb(ref_rgb, out_dir / "reference.jpg")
        common.save_rgb(tgt_rgb, out_dir / "target.jpg")
        common.save_rgb(outputs["legacy_v0"], out_dir / "legacy_v0.jpg")
        common.save_rgb(outputs["coherence_v1"], out_dir / "coherence_v1.jpg")
        make_comparison(
            [("reference", ref_rgb), ("target (orig)", tgt_rgb),
             ("legacy_v0", outputs["legacy_v0"]), ("coherence_v1", outputs["coherence_v1"])],
            out_dir / "review_sheet.jpg", panel_w=360,
        )
        (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        review_path = out_dir / "review.json"
        if not review_path.exists():
            review_path.write_text(json.dumps(REVIEW_TEMPLATE, ensure_ascii=False, indent=2))

        flags = [f"legacy:{f}" for f in result["legacy_v0_flags"]] + \
                [f"coherence:{f}" for f in result["coherence_v1_flags"]]
        flag_note = f" flags={flags}" if flags else ""
        print(f"[ok] {sid}: suitable={result['compat']['suitable']}{flag_note}")
        ok += 1
    print(f"\n{ok} processed, {skipped} skipped (missing source) -> {out_root}")


_SUMMARY_SCALAR_KEYS = ("fg_bg_luma_change_diff", "fg_bg_tone_direction_cos",
                       "neutral_vs_bg_change_ratio", "boundary_delta_e_p95", "boundary_delta_e_p99",
                       "skin_hue_drift_deg", "highlight_clip_frac", "shadow_clip_frac")


def summarize(out_root: Path) -> dict:
    """Aggregate every sample's metrics.json under out_root into a
    legacy_v0-vs-coherence_v1 mean-per-metric comparison table — quick
    evidence for "did C3 make this better on average", NOT a replacement
    for the manual FG-BG-Coord-v1 review (small-N means don't establish
    subjective quality; see the plan doc's acceptance criteria)."""
    per_pipeline: dict[str, dict[str, list[float]]] = {"legacy_v0": {}, "coherence_v1": {}}
    flag_counts = {"legacy_v0": 0, "coherence_v1": 0}
    n = 0
    for metrics_path in sorted(out_root.glob("*/metrics.json")):
        rec = json.loads(metrics_path.read_text())
        n += 1
        for key in ("legacy_v0", "coherence_v1"):
            m = rec.get(key, {})
            flag_counts[key] += len(rec.get(f"{key}_flags", []))
            for k in _SUMMARY_SCALAR_KEYS:
                v = m.get(k)
                if v is not None:
                    per_pipeline[key].setdefault(k, []).append(v)
    summary = {"n_samples": n, "flag_counts": flag_counts, "means": {}}
    for key, metrics in per_pipeline.items():
        summary["means"][key] = {k: round(float(np.mean(v)), 3) for k, v in metrics.items()}
    return summary


def score_summary(out_root: Path) -> dict:
    """Aggregate filled-in review.json files against 方案文档 §四's acceptance
    thresholds. Samples whose review.json is still the untouched empty
    template (preferred is None) are reported separately and excluded from
    the pass/fail ratios so an incomplete review pass doesn't silently look
    like a 100%-scored one."""
    scored, unscored = [], []
    for review_path in sorted(out_root.glob("*/review.json")):
        rec = json.loads(review_path.read_text())
        if rec.get("preferred") is None:
            unscored.append(review_path.parent.name)
        else:
            rec["_id"] = review_path.parent.name
            scored.append(rec)

    n = len(scored)
    severe_fg_bg = sum(1 for r in scored if r.get("severe", {}).get("severe_fg_bg_disconnect"))
    severe_halo = sum(1 for r in scored if r.get("severe", {}).get("severe_halo"))
    severe_skin = sum(1 for r in scored if r.get("severe", {}).get("severe_skin_error"))
    delivery_ok = sum(1 for r in scored
                      if (r.get("scores", {}).get("delivery_willingness") or 0) >= 4)
    coherence_wins = sum(1 for r in scored if r.get("preferred") == "coherence_v1")
    legacy_wins = sum(1 for r in scored if r.get("preferred") == "legacy_v0")
    ties = sum(1 for r in scored if r.get("preferred") == "tie")

    summary = {
        "n_scored": n,
        "n_unscored": len(unscored),
        "unscored_ids": unscored,
        "severe_fg_bg_disconnect_count": severe_fg_bg,
        "severe_halo_count": severe_halo,
        "severe_skin_error_count": severe_skin,
        "delivery_willingness_ge4_ratio": round(delivery_ok / n, 3) if n else None,
        "preferred": {"coherence_v1": coherence_wins, "legacy_v0": legacy_wins, "tie": ties},
        "coherence_win_rate_excl_tie": (
            round(coherence_wins / (coherence_wins + legacy_wins), 3)
            if (coherence_wins + legacy_wins) else None
        ),
        "acceptance": {
            "severe_issues_zero": severe_fg_bg == 0 and severe_halo == 0 and severe_skin == 0,
            "delivery_willingness_ge_80pct": (delivery_ok / n >= 0.8) if n else False,
            "coherence_win_rate_ge_70pct": (
                (coherence_wins / (coherence_wins + legacy_wins) >= 0.7)
                if (coherence_wins + legacy_wins) else False
            ),
            "min_30_samples_scored": n >= 30,
        },
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="C3-4 automated harmony metrics (legacy vs coherence)")
    ap.add_argument("--manifest", help="path to FG-BG-Coord-v1 manifest.jsonl; runs the whole set")
    ap.add_argument("--out-root", help="output root for --manifest/--summarize mode "
                    "(default: <manifest_dir>/outputs)")
    ap.add_argument("--summarize", action="store_true",
                    help="aggregate existing metrics.json under --out-root instead of re-rendering")
    ap.add_argument("--score-summary", action="store_true",
                    help="aggregate filled-in review.json under --out-root against the acceptance criteria")
    ap.add_argument("--ref", help="single-pair mode: reference image")
    ap.add_argument("--tgt", help="single-pair mode: target image")
    ap.add_argument("--strength", choices=("light", "medium", "strong"), default="medium")
    args = ap.parse_args()

    if args.summarize:
        if not args.out_root:
            ap.error("--summarize needs --out-root")
        print(json.dumps(summarize(Path(args.out_root)), ensure_ascii=False, indent=2))
        return 0

    if args.score_summary:
        if not args.out_root:
            ap.error("--score-summary needs --out-root")
        print(json.dumps(score_summary(Path(args.out_root)), ensure_ascii=False, indent=2))
        return 0

    if args.manifest:
        manifest_path = Path(args.manifest)
        out_root = Path(args.out_root) if args.out_root else manifest_path.parent / "outputs"
        out_root.mkdir(parents=True, exist_ok=True)
        run_manifest(manifest_path, out_root, args.strength)
        return 0

    if not args.ref or not args.tgt:
        ap.error("need --manifest, or both --ref and --tgt")

    ref_rgb = common.load_rgb(args.ref, max_side=1024)
    tgt_rgb = common.load_rgb(args.tgt, max_side=1024)
    result, _, _ = run_pair(ref_rgb, tgt_rgb, strength=args.strength, sample_id=Path(args.tgt).stem)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
