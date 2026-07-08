"""Stage 0 orchestrator.

For each selected image, run the full no-big-model loop:
    load image
    -> region metrics + masks
    -> plan JSON (3-5 plans, gated actions, executor routing)
    -> local preview render per non-GPT plan
    -> per-image comparison sheet
    -> review row

Usage:
    ./.venv/bin/python scripts/run_stage0.py --limit 5
    ./.venv/bin/python scripts/run_stage0.py --image-id img_000578
    ./.venv/bin/python scripts/run_stage0.py            # all 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common
from build_region_metrics import build_region_metrics
from build_review_sheet import make_comparison, review_row, write_review_csv
from generate_plans import generate_plans
from region_provider import get_provider
from render_local_preview import render_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT.parent / "outputs" / "stage0" / "stage0_selection.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs"


def process_image(meta: dict, cfg: dict, provider, max_side: int, feather: float) -> dict:
    rgb = common.load_rgb(meta["source_path"], max_side=max_side)
    image_id = meta["image_id"]

    rm = build_region_metrics(rgb, meta, cfg["thresholds"], provider=provider)
    masks = rm.pop("_masks")

    # persist regions + masks
    common.dump_json(rm, OUT_DIR / "regions" / f"{image_id}.json")
    for rid, mask in masks.items():
        if rid == "global":
            continue
        common.save_mask(mask, OUT_DIR / "masks" / image_id / f"{rid}.png")

    plans = generate_plans(rm, cfg["actions"], cfg["thresholds"])
    common.dump_json(plans, OUT_DIR / "plans" / f"{image_id}.json")

    # render local previews + build comparison sheet
    panels = [("original", rgb)]
    for plan in plans["plans"]:
        if plan["executor"] == "gpt_image_2" and not plan.get("two_stage"):
            continue  # pure GPT plan: no local preview to render
        out, info = render_plan(rgb, plan, masks)
        tag = plan["name"] + ("+GPT" if info["needs_gpt"] else "")
        panels.append((f"{plan['plan_id']} {tag}", out))
        common.save_rgb(out, OUT_DIR / "previews" / image_id / f"{plan['plan_id']}_{plan['name']}.jpg")

    if len(panels) > 1:
        make_comparison(panels, OUT_DIR / "sheets" / f"{image_id}.jpg")

    return review_row(meta, rm, plans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0 semantic color pipeline")
    ap.add_argument("--selection", default=str(DEFAULT_SELECTION))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--image-id", default=None)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--feather", type=float, default=6.0)
    ap.add_argument("--provider", default="heuristic")
    args = ap.parse_args(argv)

    cfg = common.load_config()
    provider = get_provider(args.provider)
    rows = common.read_jsonl(args.selection)
    if args.image_id:
        rows = [r for r in rows if r["image_id"] == args.image_id]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No images matched.", file=sys.stderr)
        return 1

    review_rows = []
    for i, meta in enumerate(rows, 1):
        try:
            review_rows.append(process_image(meta, cfg, provider, args.max_side, args.feather))
            print(f"[{i}/{len(rows)}] {meta['image_id']} ok")
        except Exception as e:  # keep going, log failures
            print(f"[{i}/{len(rows)}] {meta['image_id']} FAILED: {e}", file=sys.stderr)

    if review_rows:
        write_review_csv(review_rows, OUT_DIR / "stage0_pipeline_review.csv")
        print(f"\nWrote {len(review_rows)} rows -> {OUT_DIR / 'stage0_pipeline_review.csv'}")
        print(f"Sheets -> {OUT_DIR / 'sheets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
