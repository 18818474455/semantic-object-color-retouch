"""Re-run the full 20-image Phase-D + expanded regression set through the
NEW canonical color_reference_transfer module (profile-based + hard content
gate), to confirm merging/refactoring didn't change behavior for the cases
already validated, and that the new gate doesn't wrongly block legitimate
same-scene transfers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from build_review_sheet import make_comparison
from color_reference_transfer import (
    PIPELINE_LEGACY,
    SUPPORTED_PIPELINES,
    apply_profile,
    compute_style_profile,
)
from semantic_transfer_v2 import FULL_CASES, EXPANDED_CASES


def run(cases: dict, tag: str, out: Path, pipeline: str) -> None:
    for bucket, spec in cases.items():
        print(f"\n=== {bucket}{tag} ===")
        ref = common.load_rgb(spec["ref"], max_side=1024)
        profile = compute_style_profile(ref, name=bucket)
        panels = [("reference", ref)]
        for tp in spec["targets"]:
            tgt = common.load_rgb(tp, max_side=1024)
            graded, info, compat = apply_profile(
                profile, tgt, strength="medium", pipeline=pipeline
            )
            stem = Path(tp).stem.replace(" ", "_")
            status = "OK" if compat["suitable"] else "SKIPPED(gate)"
            print(f" -- {stem}: {status} jaccard={compat['jaccard']} explainable={compat['explainable_tgt_frac']}")
            common.save_rgb(graded, out / f"{bucket}_{stem}{tag}.jpg")
            panels.append(("orig " + stem, tgt))
            panels.append(("new " + stem, graded))
        make_comparison(panels, out / f"{bucket}{tag}_sheet.jpg", panel_w=280)


def main() -> int:
    ap = argparse.ArgumentParser(description="20-image color-transfer regression")
    ap.add_argument("--pipeline", choices=SUPPORTED_PIPELINES, default=PIPELINE_LEGACY)
    args = ap.parse_args()

    out = (Path(__file__).resolve().parents[1] / "outputs" /
           "color_reference_transfer" / f"regression_20_{args.pipeline}")
    out.mkdir(parents=True, exist_ok=True)
    run(FULL_CASES, "", out, pipeline=args.pipeline)
    run(EXPANDED_CASES, "_r2", out, pipeline=args.pipeline)
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
