"""Quick try: render legacy vs coherence side-by-side for any ref+tgt pair.

For when the user brings new reference/target photos to eyeball before
adding them to the eval set.

Usage:
  .venv-m2/bin/python stage0_pipeline/scripts_m2/try_pair.py \\
    --ref /path/to/reference.jpg --tgt /path/to/target.jpg \\
    --out /tmp/try_pair_out

Outputs:
  reference.jpg  target.jpg  legacy_v0.jpg  coherence_v1.jpg  review_sheet.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from build_review_sheet import make_comparison
from color_reference_transfer import (
    PIPELINE_LEGACY, PIPELINE_COHERENCE, analyze_target, compute_style_profile, render_from_analysis,
)
from eval_harmony import compute_harmony_metrics, _flag_summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick legacy vs coherence try for one ref+tgt pair")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--out", default="/tmp/try_pair_out")
    ap.add_argument("--strength", choices=("light", "medium", "strong"), default="medium")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref_rgb = common.load_rgb(args.ref, max_side=1024)
    tgt_rgb = common.load_rgb(args.tgt, max_side=1024)
    profile = compute_style_profile(ref_rgb, name=Path(args.ref).stem)
    analysis = analyze_target(profile, tgt_rgb)

    legacy = render_from_analysis(analysis, strength=args.strength, pipeline=PIPELINE_LEGACY)
    coherence = render_from_analysis(analysis, strength=args.strength, pipeline=PIPELINE_COHERENCE)

    common.save_rgb(ref_rgb, out / "reference.jpg")
    common.save_rgb(tgt_rgb, out / "target.jpg")
    common.save_rgb(legacy, out / "legacy_v0.jpg")
    common.save_rgb(coherence, out / "coherence_v1.jpg")
    make_comparison(
        [("reference", ref_rgb), ("target (orig)", tgt_rgb),
         ("legacy_v0", legacy), ("coherence_v1", coherence)],
        out / "review_sheet.jpg", panel_w=400,
    )

    metrics = {
        "compat": analysis["compat"],
        "legacy_v0": compute_harmony_metrics(analysis, legacy),
        "coherence_v1": compute_harmony_metrics(analysis, coherence),
    }
    metrics["legacy_v0_flags"] = _flag_summary(metrics["legacy_v0"])
    metrics["coherence_v1_flags"] = _flag_summary(metrics["coherence_v1"])
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"outputs -> {out}")
    print(f"suitable={analysis['compat']['suitable']}  explainable={analysis['compat']['explainable_tgt_frac']:.2f}")
    print(f"legacy flags: {metrics['legacy_v0_flags']}")
    print(f"coherence flags: {metrics['coherence_v1_flags']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
