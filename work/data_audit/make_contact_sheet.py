#!/usr/bin/env python3
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: make_contact_sheet.py <paths_txt> <output_png> <max_items>")
    paths_txt = Path(sys.argv[1])
    output_png = Path(sys.argv[2])
    max_items = int(sys.argv[3])
    paths = [Path(line.strip()) for line in paths_txt.read_text(encoding="utf-8").splitlines() if line.strip()][:max_items]

    thumb_w, thumb_h = 220, 160
    label_h = 34
    pad = 10
    cols = 5
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, path in enumerate(paths):
        col = idx % cols
        row = idx // cols
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
                ox = (thumb_w - im.width) // 2
                oy = (thumb_h - im.height) // 2
                canvas.paste(im, (ox, oy))
        except Exception:
            canvas = Image.new("RGB", (thumb_w, thumb_h), (180, 40, 40))
        sheet.paste(canvas, (x, y))
        label = path.name[:32]
        draw.text((x, y + thumb_h + 4), label, fill=(30, 30, 30))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_png)
    print(output_png)


if __name__ == "__main__":
    main()
