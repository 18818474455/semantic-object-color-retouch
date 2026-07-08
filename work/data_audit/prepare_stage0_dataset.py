#!/usr/bin/env python3
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff"}


def is_resource_fork(path: Path) -> bool:
    return path.name.startswith("._")


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and not is_resource_fork(path) and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_face_manifest(root: Path):
    result = defaultdict(lambda: {"face_count": 0, "max_face_probability": 0.0, "max_face_quality": 0.0})
    manifest = root / "face_manifest.aliyun_full.csv"
    if not manifest.exists():
        return result
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get("image_path")
            if not image_name:
                continue
            rec = result[image_name]
            rec["face_count"] += 1
            try:
                rec["max_face_probability"] = max(rec["max_face_probability"], float(row.get("face_probability", 0) or 0))
            except ValueError:
                pass
            try:
                rec["max_face_quality"] = max(rec["max_face_quality"], float(row.get("quality_score", 0) or 0))
            except ValueError:
                pass
    return result


def safe_thumb(im, max_edge=320):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    scale = max_edge / max(w, h)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return im


def image_metrics(path: Path):
    with Image.open(path) as im:
        width, height = im.size
        im.draft("RGB", (192, 192))
        thumb = ImageOps.exif_transpose(im).convert("RGB")
        thumb.thumbnail((160, 160), Image.Resampling.BILINEAR)
        pixels = list(thumb.getdata())
        total = len(pixels) or 1

        lum_values = []
        sat_values = []
        top_blue_votes = 0
        top_total = 0
        high_sat = 0
        r_sum = g_sum = b_sum = 0
        clip_high = clip_low = 0

        for y in range(thumb.height):
            for x in range(thumb.width):
                r, g, b = thumb.getpixel((x, y))
                r_sum += r
                g_sum += g
                b_sum += b
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                lum_values.append(lum)
                mx = max(r, g, b)
                mn = min(r, g, b)
                sat = (mx - mn) / max(1, mx)
                sat_values.append(sat)
                if lum >= 250:
                    clip_high += 1
                if lum <= 5:
                    clip_low += 1
                if sat > 0.55:
                    high_sat += 1
                if y < thumb.height * 0.38:
                    top_total += 1
                    if b > r + 12 and b > g + 4 and lum > 65 and sat > 0.08:
                        top_blue_votes += 1

        lum_values.sort()
        def pct(p):
            idx = min(len(lum_values) - 1, max(0, int((len(lum_values) - 1) * p)))
            return lum_values[idx]

        r_mean = r_sum / total
        g_mean = g_sum / total
        b_mean = b_sum / total
        brightness = sum(lum_values) / total
        saturation = sum(sat_values) / total
        blue_top_ratio = top_blue_votes / max(1, top_total)
        color_cast_g = (g_mean - ((r_mean + b_mean) / 2)) / 255
        color_cast_warm = (r_mean - b_mean) / 255

        return {
            "width": width,
            "height": height,
            "orientation": "landscape" if width > height else "portrait" if height > width else "square",
            "brightness": round(brightness, 3),
            "lum_p01": round(pct(0.01), 3),
            "lum_p05": round(pct(0.05), 3),
            "lum_p50": round(pct(0.50), 3),
            "lum_p95": round(pct(0.95), 3),
            "lum_p99": round(pct(0.99), 3),
            "clip_high_pct": round(clip_high / total * 100, 4),
            "clip_low_pct": round(clip_low / total * 100, 4),
            "saturation": round(saturation, 4),
            "high_sat_pct": round(high_sat / total * 100, 3),
            "blue_top_ratio": round(blue_top_ratio, 4),
            "green_cast_proxy": round(color_cast_g, 4),
            "warm_cast_proxy": round(color_cast_warm, 4),
            "r_mean": round(r_mean, 2),
            "g_mean": round(g_mean, 2),
            "b_mean": round(b_mean, 2),
        }


def score_buckets(rec):
    scores = {}
    has_face = rec["face_count"] > 0
    scores["person_event"] = (3 if has_face else 0) + rec["face_count"] * 0.3 + rec["max_face_quality"] / 100
    scores["outdoor_sky"] = rec["blue_top_ratio"] * 5 + (0.4 if rec["orientation"] == "landscape" else 0)
    scores["stage_led_mixed"] = rec["high_sat_pct"] / 20 + (1.5 if rec["brightness"] < 95 else 0) + abs(rec["green_cast_proxy"]) * 2 + abs(rec["warm_cast_proxy"]) * 1.5
    scores["difficult"] = rec["clip_high_pct"] / 5 + rec["clip_low_pct"] / 8 + (1.4 if rec["brightness"] < 60 or rec["brightness"] > 190 else 0)
    return scores


def pick_bucket(records, bucket, count, used, rng):
    candidates = [r for r in records if r["image_id"] not in used]
    candidates.sort(key=lambda r: (r["bucket_scores"][bucket], rng.random()), reverse=True)
    picks = candidates[:count]
    used.update(r["image_id"] for r in picks)
    for r in picks:
        r["stage0_bucket"] = bucket
    return picks


def write_contact_sheet(records, output_png):
    thumb_w, thumb_h = 220, 160
    label_h = 48
    pad = 10
    cols = 5
    rows = math.ceil(len(records) / cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, rec in enumerate(records):
        path = Path(rec["source_path"])
        col = idx % cols
        row = idx // cols
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
                canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        except Exception:
            canvas = Image.new("RGB", (thumb_w, thumb_h), (150, 40, 40))
        sheet.paste(canvas, (x, y))
        label = f"{rec['stage0_bucket']}\n{path.name[:28]}"
        draw.multiline_text((x, y + thumb_h + 3), label, fill=(25, 25, 25), spacing=2)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_png)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare_stage0_dataset.py <photo_root> <output_dir>")
    root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260706)
    face_manifest = load_face_manifest(root)

    all_paths = sorted(iter_images(root))
    face_names = set(face_manifest.keys())
    face_paths = [p for p in all_paths if p.name in face_names]
    non_face_paths = [p for p in all_paths if p.name not in face_names]
    rng.shuffle(face_paths)
    rng.shuffle(non_face_paths)

    # Stage 0 only needs a representative validation set. Computing full metrics
    # on every image from an external drive is slow, so score a broad candidate
    # pool and still write a full lightweight manifest separately.
    candidate_paths = []
    candidate_paths.extend(face_paths[:450])
    candidate_paths.extend(non_face_paths[:750])
    seen = set()
    candidate_paths = [p for p in candidate_paths if not (str(p) in seen or seen.add(str(p)))]

    records = []
    errors = []
    for idx, path in enumerate(candidate_paths):
        try:
            metrics = image_metrics(path)
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
            continue
        rel_name = path.name
        face = face_manifest.get(rel_name, {})
        rec = {
            "image_id": f"img_{idx + 1:06d}",
            "source_path": str(path),
            "filename": rel_name,
            "face_count": int(face.get("face_count", 0)),
            "max_face_probability": round(float(face.get("max_face_probability", 0.0)), 4),
            "max_face_quality": round(float(face.get("max_face_quality", 0.0)), 3),
            **metrics,
        }
        rec["bucket_scores"] = score_buckets(rec)
        records.append(rec)

    used = set()
    selection = []
    targets = [
        ("outdoor_sky", 30),
        ("person_event", 30),
        ("stage_led_mixed", 20),
        ("difficult", 20),
    ]
    for bucket, count in targets:
        selection.extend(pick_bucket(records, bucket, count, used, rng))

    # Fill if any bucket had too few records.
    if len(selection) < 100:
        remaining = [r for r in records if r["image_id"] not in used]
        rng.shuffle(remaining)
        for rec in remaining[: 100 - len(selection)]:
            rec["stage0_bucket"] = "fill_random"
            selection.append(rec)

    clean_manifest = out_dir / "manifest_clean.jsonl"
    with clean_manifest.open("w", encoding="utf-8") as f:
        for idx, path in enumerate(all_paths):
            face = face_manifest.get(path.name, {})
            row = {
                "image_id": f"img_{idx + 1:06d}",
                "source_path": str(path),
                "filename": path.name,
                "face_count": int(face.get("face_count", 0)),
                "max_face_probability": round(float(face.get("max_face_probability", 0.0)), 4),
                "max_face_quality": round(float(face.get("max_face_quality", 0.0)), 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_jsonl = out_dir / "image_metrics.jsonl"
    with metrics_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            row = {k: v for k, v in rec.items() if k != "bucket_scores"}
            row["bucket_scores"] = rec["bucket_scores"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stage0_jsonl = out_dir / "stage0_selection.jsonl"
    with stage0_jsonl.open("w", encoding="utf-8") as f:
        for rec in selection:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (out_dir / "stage0_paths.txt").open("w", encoding="utf-8") as f:
        for rec in selection:
            f.write(rec["source_path"] + "\n")

    with (out_dir / "stage0_review.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image_id",
            "stage0_bucket",
            "source_path",
            "best_plan",
            "usable",
            "should_not_edit_but_changed",
            "face_changed",
            "text_logo_changed",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selection:
            writer.writerow({
                "image_id": rec["image_id"],
                "stage0_bucket": rec["stage0_bucket"],
                "source_path": rec["source_path"],
            })

    write_contact_sheet(selection, out_dir / "stage0_contact_sheet.png")

    bucket_counts = Counter(rec.get("stage0_bucket", "unknown") for rec in selection)
    summary = {
        "root": str(root),
        "image_count": len(all_paths),
        "candidate_metric_count": len(records),
        "error_count": len(errors),
        "stage0_count": len(selection),
        "stage0_bucket_counts": dict(bucket_counts),
        "outputs": {
            "clean_manifest": str(clean_manifest),
            "image_metrics": str(metrics_jsonl),
            "stage0_selection": str(stage0_jsonl),
            "stage0_review": str(out_dir / "stage0_review.csv"),
            "contact_sheet": str(out_dir / "stage0_contact_sheet.png"),
        },
        "errors": errors[:20],
    }
    (out_dir / "stage0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
