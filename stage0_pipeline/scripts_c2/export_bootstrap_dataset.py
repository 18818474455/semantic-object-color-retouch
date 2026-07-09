"""C2.1 — Export bootstrap training manifest from regression case specs.

Reads FULL_CASES / EXPANDED_CASES (semantic_transfer_v2) plus the Stage 0
100-image validation set (outputs/stage0/image_metrics.jsonl, bucketed by
build_region_metrics.py's heuristic classifier), runs the current rule-based
teacher (color_reference_transfer medium), and writes:
  dataset/c2/manifest.jsonl
  dataset/c2/profiles/<sample_id>.json
  dataset/c2/edited/<sample_id>_medium.jpg
  dataset/c2/meta/<sample_id>.json

Skips pairs whose source images are missing (e.g. external volume unmounted).
Always includes a local smoke pair if gpt_teacher/ref_small.jpg exists.

Run (.venv-m2 for detection inside apply_profile):
  cd stage0_pipeline
  ../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py
  ../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py --smoke-only
  ../.venv-m2/bin/python scripts_c2/export_bootstrap_dataset.py --no-stage0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_m2"))

import common
from color_reference_transfer import compute_style_profile, apply_profile, save_profile
from semantic_transfer_v2 import FULL_CASES, EXPANDED_CASES

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PIPELINE_ROOT / "dataset" / "c2"
STAGE0_METRICS = PIPELINE_ROOT.parent / "outputs" / "stage0" / "image_metrics.jsonl"
LOCAL_SMOKE = (
    PIPELINE_ROOT / "outputs" / "color_reference_transfer" / "gpt_teacher" / "ref_small.jpg",
    PIPELINE_ROOT / "outputs" / "color_reference_transfer" / "gpt_teacher" / "tgt_small.jpg",
)


def _iter_cases(include_expanded: bool) -> list[tuple[str, str, str, str]]:
    """Yield (bucket, tag, ref_path, tgt_path). tag '' or '_r2'."""
    rows: list[tuple[str, str, str, str]] = []
    for bucket, spec in FULL_CASES.items():
        ref = spec["ref"]
        for tgt in spec["targets"]:
            rows.append((bucket, "", ref, tgt))
    if include_expanded:
        for bucket, spec in EXPANDED_CASES.items():
            ref = spec["ref"]
            for tgt in spec["targets"]:
                rows.append((bucket, "_r2", ref, tgt))
    return rows


def _iter_stage0_cases(existing_targets: set[str]) -> list[tuple[str, str, str, str]]:
    """Yield (bucket, '_s0', ref_path, tgt_path) for the Stage 0 100-image
    validation set (bucketed by build_region_metrics.py's heuristic
    classifier), reusing the same fixed per-bucket reference as
    FULL_CASES/EXPANDED_CASES so results stay comparable. Skips images
    already used as a target or as a bucket reference."""
    if not STAGE0_METRICS.is_file():
        return []
    refs = {b: spec["ref"] for b, spec in FULL_CASES.items()}
    seen = set(existing_targets) | set(refs.values())
    rows: list[tuple[str, str, str, str]] = []
    with STAGE0_METRICS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            bucket = d.get("stage0_bucket")
            path = d.get("source_path")
            if not bucket or not path or bucket not in refs or path in seen:
                continue
            seen.add(path)
            rows.append((bucket, "_s0", refs[bucket], path))
    return rows


def _sample_id(bucket: str, tag: str, tgt_path: str) -> str:
    stem = Path(tgt_path).stem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{bucket}_{stem}{tag}"


def _export_one(
    bucket: str,
    tag: str,
    ref_path: str,
    tgt_path: str,
    strength: str = "medium",
) -> dict | None:
    ref_p, tgt_p = Path(ref_path), Path(tgt_path)
    if not ref_p.is_file() or not tgt_p.is_file():
        return None

    sid = _sample_id(bucket, tag, tgt_path)
    profiles_dir = OUT_ROOT / "profiles"
    edited_dir = OUT_ROOT / "edited"
    meta_dir = OUT_ROOT / "meta"
    for d in (profiles_dir, edited_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    ref_rgb = common.load_rgb(str(ref_p), max_side=1024)
    tgt_rgb = common.load_rgb(str(tgt_p), max_side=1024)
    profile = compute_style_profile(ref_rgb, name=bucket)
    profile_path = profiles_dir / f"{sid}.json"
    save_profile(profile, profile_path)

    graded, matched_info, compat = apply_profile(profile, tgt_rgb, strength=strength)
    edited_path = edited_dir / f"{sid}_{strength}.jpg"
    common.save_rgb(graded, edited_path)

    matched_classes = sorted(
        c for c, v in matched_info.items() if v.get("matched")
    )
    row = {
        "sample_id": sid,
        "bucket": bucket,
        "tag": tag or "full",
        "reference_path": str(ref_p.resolve()),
        "target_path": str(tgt_p.resolve()),
        "pseudo_target_path": str(edited_path.relative_to(PIPELINE_ROOT)),
        "strength": strength,
        "style_profile_path": str(profile_path.relative_to(PIPELINE_ROOT)),
        "compat": compat,
        "matched_classes": matched_classes,
        "matched_info": matched_info,
        "split": "train",
    }
    meta_dir.joinpath(f"{sid}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2)
    )
    return row


def _export_smoke(strength: str = "medium") -> dict | None:
    ref_p, tgt_p = LOCAL_SMOKE
    if not ref_p.is_file() or not tgt_p.is_file():
        print(f"smoke pair missing: {ref_p} / {tgt_p}")
        return None
    return _export_one("smoke", "", str(ref_p), str(tgt_p), strength=strength)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="only export local ref_small/tgt_small")
    ap.add_argument("--no-expanded", action="store_true", help="skip EXPANDED_CASES (_r2)")
    ap.add_argument("--no-stage0", action="store_true", help="skip the Stage 0 100-image validation set (_s0)")
    ap.add_argument("--strength", default="medium", choices=["light", "medium", "strong"])
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.jsonl"
    rows: list[dict] = []
    skipped = 0

    if args.smoke_only:
        row = _export_smoke(args.strength)
        if row:
            rows.append(row)
    else:
        base_cases = _iter_cases(include_expanded=not args.no_expanded)
        for bucket, tag, ref, tgt in base_cases:
            row = _export_one(bucket, tag, ref, tgt, strength=args.strength)
            if row:
                rows.append(row)
                print(f"OK  {row['sample_id']} suitable={row['compat']['suitable']}")
            else:
                skipped += 1

        if not args.no_stage0:
            existing_targets = {tgt for _, _, _, tgt in base_cases}
            for bucket, tag, ref, tgt in _iter_stage0_cases(existing_targets):
                row = _export_one(bucket, tag, ref, tgt, strength=args.strength)
                if row:
                    rows.append(row)
                    print(f"OK  {row['sample_id']} suitable={row['compat']['suitable']}")
                else:
                    skipped += 1

        smoke = _export_smoke(args.strength)
        if smoke and smoke["sample_id"] not in {r["sample_id"] for r in rows}:
            rows.append(smoke)
            print(f"OK  {smoke['sample_id']} (local smoke) suitable={smoke['compat']['suitable']}")

    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    )
    suitable = sum(1 for r in rows if r["compat"].get("suitable"))
    print(f"\nmanifest -> {manifest_path}")
    print(f"exported={len(rows)} suitable={suitable} skipped_missing={skipped}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
