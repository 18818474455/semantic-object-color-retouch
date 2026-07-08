#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install pillow") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".dng", ".raw", ".arw", ".cr2", ".nef"}


def is_resource_fork(path: Path) -> bool:
    return path.name.startswith("._")


def image_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and not is_resource_fork(path) and path.suffix.lower() in IMAGE_EXTS:
            yield path


def fast_sha1(path: Path, size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(size))
    return h.hexdigest()


def classify_orientation(w: int, h: int) -> str:
    if w == h:
        return "square"
    return "landscape" if w > h else "portrait"


def light_metrics(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        thumb = im.resize((min(256, w), max(1, int(h * min(256, w) / w)))) if w >= h else im.resize((max(1, int(w * min(256, h) / h)), min(256, h)))
        stat = ImageStat.Stat(thumb)
        mean = stat.mean
        extrema = stat.extrema
        brightness = sum(mean) / 3.0
        saturation_proxy = statistics.pstdev(mean)
        clipped_low = 0
        clipped_high = 0
        pixels = list(thumb.getdata())
        for r, g, b in pixels:
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum <= 5:
                clipped_low += 1
            if lum >= 250:
                clipped_high += 1
        total = len(pixels) or 1
        return {
            "width": w,
            "height": h,
            "orientation": classify_orientation(w, h),
            "brightness": round(brightness, 2),
            "saturation_proxy": round(saturation_proxy, 2),
            "clip_low_pct": round(clipped_low / total * 100, 3),
            "clip_high_pct": round(clipped_high / total * 100, 3),
            "r_mean": round(mean[0], 2),
            "g_mean": round(mean[1], 2),
            "b_mean": round(mean[2], 2),
            "r_minmax": extrema[0],
            "g_minmax": extrema[1],
            "b_minmax": extrema[2],
        }


def read_face_manifest(root: Path):
    manifest = root / "face_manifest.aliyun_full.csv"
    if not manifest.exists():
        return None
    unique_images = set()
    rows = 0
    split_counts = Counter()
    qualities = []
    probabilities = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            image_path = row.get("image_path")
            if image_path:
                unique_images.add(image_path)
            split_counts[row.get("split", "")] += 1
            try:
                qualities.append(float(row.get("quality_score", "")))
            except ValueError:
                pass
            try:
                probabilities.append(float(row.get("face_probability", "")))
            except ValueError:
                pass
    return {
        "rows": rows,
        "unique_images": len(unique_images),
        "split_counts": dict(split_counts),
        "avg_quality_score": round(sum(qualities) / len(qualities), 2) if qualities else None,
        "avg_face_probability": round(sum(probabilities) / len(probabilities), 3) if probabilities else None,
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: audit_photo_dataset.py <photo_root> <output_dir>")
    root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(image_files(root))
    random.seed(42)
    sample = random.sample(files, min(300, len(files)))

    metrics = []
    errors = []
    for path in sample:
        try:
            m = light_metrics(path)
            m["path"] = str(path)
            m["size_bytes"] = path.stat().st_size
            m["fast_sha1"] = fast_sha1(path)
            metrics.append(m)
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})

    ext_counts = Counter(p.suffix.lower() for p in files)
    dir_counts = Counter(str(p.parent.relative_to(root)) for p in files)
    resource_forks = sum(1 for p in root.rglob("*") if p.is_file() and is_resource_fork(p))
    all_non_resource = [p for p in root.rglob("*") if p.is_file() and not is_resource_fork(p)]
    non_image_ext_counts = Counter(p.suffix.lower() or "<no_ext>" for p in all_non_resource if p.suffix.lower() not in IMAGE_EXTS)

    orientation_counts = Counter(m["orientation"] for m in metrics)
    size_buckets = Counter()
    for m in metrics:
        long_edge = max(m["width"], m["height"])
        if long_edge < 1500:
            size_buckets["lt_1500"] += 1
        elif long_edge < 2500:
            size_buckets["1500_2499"] += 1
        elif long_edge < 3500:
            size_buckets["2500_3499"] += 1
        elif long_edge < 5000:
            size_buckets["3500_4999"] += 1
        else:
            size_buckets["gte_5000"] += 1

    bright = [m["brightness"] for m in metrics]
    sat = [m["saturation_proxy"] for m in metrics]

    summary = {
        "root": str(root),
        "image_count": len(files),
        "resource_fork_count": resource_forks,
        "ext_counts": dict(ext_counts),
        "top_dirs": dict(dir_counts.most_common(20)),
        "non_image_ext_counts": dict(non_image_ext_counts),
        "sample_count": len(metrics),
        "sample_errors": errors[:20],
        "orientation_counts_sample": dict(orientation_counts),
        "long_edge_buckets_sample": dict(size_buckets),
        "brightness_sample": {
            "min": min(bright) if bright else None,
            "median": statistics.median(bright) if bright else None,
            "max": max(bright) if bright else None,
        },
        "saturation_proxy_sample": {
            "min": min(sat) if sat else None,
            "median": statistics.median(sat) if sat else None,
            "max": max(sat) if sat else None,
        },
        "face_manifest": read_face_manifest(root),
    }

    (out_dir / "dataset_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "sample_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "path",
            "width",
            "height",
            "orientation",
            "brightness",
            "saturation_proxy",
            "clip_low_pct",
            "clip_high_pct",
            "r_mean",
            "g_mean",
            "b_mean",
            "size_bytes",
            "fast_sha1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # Small balanced contact sheet sample list for later visual review.
    by_orientation = defaultdict(list)
    for row in metrics:
        by_orientation[row["orientation"]].append(row)
    contact_rows = []
    for orient, rows in by_orientation.items():
        rows = sorted(rows, key=lambda r: r["brightness"])
        picks = rows[:3] + rows[len(rows) // 2 : len(rows) // 2 + 3] + rows[-3:]
        for item in picks:
            contact_rows.append(item)
    with (out_dir / "review_sample_paths.txt").open("w", encoding="utf-8") as f:
        for row in contact_rows[:40]:
            f.write(row["path"] + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
