"""Build per-image before/after comparison sheets and review rows."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REVIEW_COLUMNS = [
    "image_id", "bucket", "scene_type", "plans_offered", "executors",
    "sky_action", "gpt_needed",
    "best_plan", "usable", "should_not_edit_but_changed",
    "face_changed", "text_logo_changed", "notes",
]


def _font(size: int = 22):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), mode="RGB")


def make_comparison(panels: list[tuple[str, np.ndarray]], out_path: str | Path, panel_w: int = 520) -> None:
    """panels: list of (label, rgb). Lays them out in a row with labels."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    font = _font(22)
    label_h = 34
    imgs = []
    for label, rgb in panels:
        im = _to_pil(rgb)
        w, h = im.size
        ph = round(panel_w * h / w)
        im = im.resize((panel_w, ph), Image.LANCZOS)
        imgs.append((label, im))
    max_h = max(im.size[1] for _, im in imgs)
    canvas = Image.new("RGB", (panel_w * len(imgs), max_h + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(imgs):
        x = i * panel_w
        draw.text((x + 8, 6), label, fill=(240, 240, 240), font=font)
        canvas.paste(im, (x, label_h))
    canvas.save(out_path, quality=90)


def write_review_csv(rows: list[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REVIEW_COLUMNS})


def review_row(meta: dict, region_metrics: dict, plans: dict) -> dict:
    plist = plans["plans"]
    sky_action = ""
    for p in plist:
        for ra in p.get("region_actions", []):
            if ra["object_type"] == "sky":
                sky_action = ra["action"]
    return {
        "image_id": meta["image_id"],
        "bucket": meta.get("stage0_bucket", ""),
        "scene_type": region_metrics["scene"]["type"],
        "plans_offered": "|".join(p["name"] for p in plist),
        "executors": "|".join(p["executor"] for p in plist),
        "sky_action": sky_action,
        "gpt_needed": "yes" if any(p["executor"] == "gpt_image_2" for p in plist) else "no",
        # human columns left blank:
        "best_plan": "", "usable": "", "should_not_edit_but_changed": "",
        "face_changed": "", "text_logo_changed": "", "notes": "",
    }
