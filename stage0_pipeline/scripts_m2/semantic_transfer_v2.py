"""M2 end-to-end: real Grounding DINO + SAM masks (+ face-based skin) driving
the SAME per-class Lab grading math from semantic_color_transfer.py, in place
of the Phase-D heuristic 4-class color classifier.

Run under .venv-m2 (needs torch + transformers for detection; the grading
math itself is plain numpy, imported unchanged from the main pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import face_detect
from semantic_color_transfer import _grade_class, _vibrance_contrast_sharpen, _percentile_stats
from region_provider_v2 import detect_classes
from build_review_sheet import make_comparison

MIN_FRAC = 0.01
# Segmentation + sky-plausibility gate is now validated (Phase D re-run, 8/8
# buckets, no artifacts) — classes we actually match can push past 1.0 like
# the original heuristic pipeline's CLASS_STRENGTH did, for a bolder result.
DEFAULT_STRENGTH = 1.6
SKIN_STRENGTH = 1.3
NEUTRAL_STRENGTH = 0.45  # leftover/unclassified pixels: touch gently, not zero


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


def transfer_v2(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, feather: float = 4.0) -> tuple[np.ndarray, dict]:
    ref_lab, tgt_lab = common.rgb_to_lab(ref_rgb), common.rgb_to_lab(tgt_rgb)
    ref_cls, tgt_cls = build_classes(ref_rgb), build_classes(tgt_rgb)
    n_ref, n_tgt = ref_rgb.shape[0] * ref_rgb.shape[1], tgt_rgb.shape[0] * tgt_rgb.shape[1]

    matched_info = {}
    class_names = set(tgt_cls) | {"neutral"}
    feathered = {c: common.feather_mask(tgt_cls.get(c, np.zeros(tgt_rgb.shape[:2], np.float32)), radius=feather)
                for c in class_names}
    denom = np.sum(list(feathered.values()), axis=0) + 1e-6
    weights = {c: feathered[c] / denom for c in class_names}

    acc = np.zeros_like(tgt_lab)
    for c in class_names:
        tm = tgt_cls.get(c)
        rm = ref_cls.get(c)
        t_frac = float(tm.mean()) if tm is not None else 0.0
        r_frac = float(rm.mean()) if rm is not None else 0.0
        matched = tm is not None and rm is not None and t_frac > MIN_FRAC and r_frac > MIN_FRAC
        matched_info[c] = {"tgt_frac": round(t_frac, 4), "ref_frac": round(r_frac, 4), "matched": matched}
        if matched:
            graded = _grade_class(c, tgt_lab, tm, ref_lab, rm)
            cs = SKIN_STRENGTH if c == "skin" else (NEUTRAL_STRENGTH if c == "neutral" else DEFAULT_STRENGTH)
            if c == "neutral" and t_frac > 0.5:
                # "neutral" is the unrecognized leftover, not a real semantic
                # match. When it swallows most of the frame, the model simply
                # failed to recognize this scene's content — forcing a full
                # global cast toward the reference here is exactly the old
                # "content mismatch -> orange-wash filter" bug. Taper strength
                # to near-zero as neutral approaches the whole image so an
                # unrecognized scene stays close to untouched instead of
                # inheriting the reference's global color as a filter.
                cs *= max(0.0, 1.0 - (t_frac - 0.5) / 0.5) ** 2
            graded = tgt_lab * (1.0 - cs) + graded * cs
        else:
            graded = tgt_lab
        acc += graded * weights[c][..., None]

    acc[..., 0] = np.clip(acc[..., 0], 0.0, 100.0)
    out_rgb = np.clip(common.lab_to_rgb(acc), 0.0, 1.0)
    out_rgb = _vibrance_contrast_sharpen(out_rgb, tgt_cls["skin"],
                                          vibrance=0.4, contrast=1.14, sharpen_amount=0.4)
    return out_rgb, matched_info


# Full Phase-D re-run: same 4 buckets x 2 targets used to find the original
# bugs, now through the v2 (GD+SAM, sky-plausibility-gated) pipeline. No
# cherry-picking — every case that was tested before is tested again.
FULL_CASES = {
    "outdoor_sky": {
        "ref": "/Volumes/未命名/大模型/原图1/DAP02456(1).JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC04085(1).JPG",
                    "/Volumes/未命名/大模型/原图1/DSC05360(1).JPG"],
    },
    "person_event": {
        "ref": "/Volumes/未命名/大模型/原图1/058A1824.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC01549.JPG",
                    "/Volumes/未命名/大模型/原图1/058A1518.JPG"],
    },
    "stage_led_mixed": {
        "ref": "/Volumes/未命名/大模型/原图1/DSC06040.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC04541.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP05979.JPG"],
    },
    "difficult": {
        "ref": "/Volumes/未命名/大模型/原图1/DSC04902.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DAP05876.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP09915 2.JPG"],
    },
}

# Round 2: wider, non-cherry-picked sample from stage0_selection.jsonl — new
# targets per bucket not seen in the first Phase-D run, same bucket reference
# kept fixed so results are comparable to round 1.
EXPANDED_CASES = {
    "outdoor_sky": {
        "ref": "/Volumes/未命名/大模型/原图1/DAP02456(1).JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DAP03170(1).JPG",
                    "/Volumes/未命名/大模型/原图1/058A1908.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP06898.JPG"],
    },
    "person_event": {
        "ref": "/Volumes/未命名/大模型/原图1/058A1824.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC01533.JPG",
                    "/Volumes/未命名/大模型/原图1/058A1568.JPG",
                    "/Volumes/未命名/大模型/原图1/DSC04819.JPG"],
    },
    "stage_led_mixed": {
        "ref": "/Volumes/未命名/大模型/原图1/DSC06040.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC04543.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP08349.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP02394 2.JPG"],
    },
    "difficult": {
        "ref": "/Volumes/未命名/大模型/原图1/DSC04902.JPG",
        "targets": ["/Volumes/未命名/大模型/原图1/DSC03839.JPG",
                    "/Volumes/未命名/大模型/原图1/058A1760.JPG",
                    "/Volumes/未命名/大模型/原图1/DAP05057.JPG"],
    },
}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    cases = EXPANDED_CASES if which == "expanded" else FULL_CASES
    tag = "_r2" if which == "expanded" else ""
    OUT = Path(__file__).resolve().parents[1] / "outputs" / "m2_spike"
    OUT.mkdir(parents=True, exist_ok=True)
    for bucket, spec in cases.items():
        print(f"\n=== {bucket} ===")
        ref = common.load_rgb(spec["ref"], max_side=1024)
        panels = [("reference", ref)]
        for tp in spec["targets"]:
            tgt = common.load_rgb(tp, max_side=1024)
            graded, info = transfer_v2(ref, tgt)
            stem = Path(tp).stem.replace(" ", "_")
            print(f" -- {stem} --")
            for c, v in info.items():
                if v["matched"]:
                    print(f"    {c:12s} tgt_frac={v['tgt_frac']:.3f} ref_frac={v['ref_frac']:.3f} MATCHED")
            common.save_rgb(graded, OUT / f"{bucket}_{stem}_v2.jpg")
            panels.append(("orig " + stem, tgt))
            panels.append(("v2 " + stem, graded))
        make_comparison(panels, OUT / f"{bucket}{tag}_v2_sheet.jpg", panel_w=280)
    print(f"\noutputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
