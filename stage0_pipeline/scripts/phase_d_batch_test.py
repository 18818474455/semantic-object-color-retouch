"""Phase D: does today's semantic color-transfer algorithm generalize beyond
the 3 images we hand-tuned it on? Pull ref/target pairs from 4 different
Stage-0 buckets (outdoor_sky, person_event, stage_led_mixed, difficult) and
run the SAME code/params, unmodified, on each. No per-bucket tuning allowed —
that is the whole point of this test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from semantic_color_transfer import classify, semantic_transfer
from build_review_sheet import make_comparison

OUT = Path(__file__).resolve().parents[1] / "outputs" / "color_transfer" / "phase_d"
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
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


def main() -> int:
    for bucket, spec in CASES.items():
        print(f"\n=== {bucket} ===")
        ref = common.load_rgb(spec["ref"], max_side=1024)
        ref_cls = classify(ref)
        print("ref class frac:", {k: round(float(v.mean()), 3) for k, v in ref_cls.items()})
        panels = [("reference", ref)]
        for tp in spec["targets"]:
            tgt = common.load_rgb(tp, max_side=1024)
            tgt_cls = classify(tgt)
            print(Path(tp).stem, "tgt class frac:",
                  {k: round(float(v.mean()), 3) for k, v in tgt_cls.items()})
            graded = semantic_transfer(ref, tgt, strength=1.0)
            stem = Path(tp).stem.replace(" ", "_")
            common.save_rgb(graded, OUT / f"{bucket}_{stem}_semantic.jpg")
            panels.append(("orig " + stem, tgt))
            panels.append(("graded " + stem, graded))
        make_comparison(panels, OUT / f"{bucket}_sheet.jpg", panel_w=360)
    print(f"\noutputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
