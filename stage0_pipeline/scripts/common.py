"""Shared utilities for the Stage 0 semantic color pipeline.

Pure numpy + Pillow. No heavy ML deps here so the skeleton runs anywhere.
Color-space helpers operate on float RGB in [0, 1] with shape (H, W, 3).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


# --------------------------------------------------------------------------- IO
def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config() -> dict[str, Any]:
    return {
        "actions": load_json(CONFIG_DIR / "actions.v1.json"),
        "object_prompts": load_json(CONFIG_DIR / "object_prompts.json"),
        "thresholds": load_json(CONFIG_DIR / "thresholds.json"),
    }


# ----------------------------------------------------------------------- images
def load_rgb(path: str | os.PathLike, max_side: int | None = 1600) -> np.ndarray:
    """Load an image as float RGB in [0,1], EXIF-oriented, optionally downscaled."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = max_side / float(max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_rgb(arr: np.ndarray, path: str | os.PathLike, quality: int = 92) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality)
    else:
        im.save(path)


def save_mask(mask: np.ndarray, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = np.clip(mask, 0.0, 1.0)
    Image.fromarray((m * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def feather_mask(mask: np.ndarray, radius: float = 6.0) -> np.ndarray:
    """Gaussian-feather a float mask via Pillow, returns float in [0,1]."""
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0)
    m = np.clip(mask, 0.0, 1.0)
    im = Image.fromarray((m * 255.0 + 0.5).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(im, dtype=np.float32) / 255.0


def morph_open(mask_bool: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple binary opening (erode then dilate) with Pillow min/max filters."""
    im = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    im = im.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    return np.asarray(im, dtype=np.uint8) > 127


# ---------------------------------------------------------------- color spaces
def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV. H in [0,360), S,V in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # red is max
    idx = mask & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    # green is max
    idx = mask & (mx == g)
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    # blue is max
    idx = mask & (mx == b)
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
    s = np.where(mx > 1e-8, diff / np.maximum(mx, 1e-8), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = np.mod(h, 360.0)
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    z = np.zeros_like(h)
    cond = (h < 60, h < 120, h < 180, h < 240, h < 300, h >= 300)
    r = np.select(cond, [c, x, z, z, x, c], default=z)
    g = np.select(cond, [x, c, c, x, z, z], default=z)
    b = np.select(cond, [z, z, x, c, c, x], default=z)
    return np.stack([r + m, g + m, b + m], axis=-1)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIE Lab (D65). L in [0,100], a/b roughly [-128,127]."""
    lin = _srgb_to_linear(rgb)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / white
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return np.stack([L, a, bb], axis=-1)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab (D65) -> sRGB [0,1]. Inverse of rgb_to_lab."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    fx3, fy3, fz3 = fx ** 3, fy ** 3, fz ** 3
    xr = np.where(fx3 > eps, fx3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, fy3, L / kappa)
    zr = np.where(fz3 > eps, fz3, (116.0 * fz - 16.0) / kappa)
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = np.stack([xr, yr, zr], axis=-1) * white
    m_inv = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]],
        dtype=np.float32,
    )
    lin = xyz @ m_inv.T
    return _linear_to_srgb(lin)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def colorfulness(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Hasler-Susstrunk colorfulness metric."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() < 4:
            return 0.0
        rg, yb = rg[sel], yb[sel]
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def sharpness_proxy(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalized gradient-energy sharpness proxy in ~[0,1]."""
    lum = luminance(rgb)
    gy, gx = np.gradient(lum)
    grad = np.sqrt(gx * gx + gy * gy)
    if mask is not None:
        sel = mask > 0.5
        if sel.sum() < 4:
            return 0.0
        grad = grad[sel]
    return float(np.clip(grad.mean() * 12.0, 0.0, 1.0))


def mean_where(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = mask > 0.5
    if sel.sum() == 0:
        return np.zeros(values.shape[-1] if values.ndim == 3 else 1, dtype=np.float32)
    if values.ndim == 3:
        return values[sel].mean(axis=0)
    return np.array([values[sel].mean()], dtype=np.float32)


def to_float_list(arr: Iterable, ndigits: int = 4) -> list:
    return [round(float(x), ndigits) for x in np.asarray(arr).ravel()]
