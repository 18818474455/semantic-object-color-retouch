"""Local masked color renderer (Python reference implementation).

Applies an action's `local_params` to pixels inside a feathered region mask,
scaled by the plan's strength. This is the Stage 0 stand-in for the BeautySDK
C++ `pe_process_image` + region blend that M4 will port to. The parameter
vocabulary here IS the contract the C++ side must match.

Supported local_params keys (all optional, value is at strength=1.0):
  RGB-space ops:
    white_balance: "gray_world"   -> neutralize cast within the region
    exposure: float               -> multiplicative exposure (+ brighter)
    contrast: float               -> contrast around mid gray
    temp: float                   -> + warm (r up / b down), - cool (b up / r down)
    tint: float                   -> + magenta (g down), - green (g up)
  HSV-space ops:
    toward_hue: {deg, amount}     -> rotate hue toward target degree
    sat_add: float                -> add to saturation
    lum_add: float                -> add to HSV value
    toward_neutral: float         -> desaturate toward gray
"""
from __future__ import annotations

import numpy as np

import common


def _apply_rgb_ops(rgb: np.ndarray, p: dict, s: float, region_mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    if p.get("white_balance") == "gray_world":
        sel = region_mask > 0.5
        if sel.sum() > 4:
            means = out[sel].reshape(-1, 3).mean(axis=0)
            gray = float(means.mean())
            factor = np.where(means > 1e-4, gray / means, 1.0)
            factor = 1.0 + s * (factor - 1.0)
            out = out * factor
    if "exposure" in p:
        out = out * (1.0 + p["exposure"] * s)
    if "contrast" in p:
        c = p["contrast"] * s
        out = (out - 0.5) * (1.0 + c) + 0.5
    if "temp" in p:
        k = 0.2 * p["temp"] * s
        out[..., 0] = out[..., 0] * (1.0 + k)
        out[..., 2] = out[..., 2] * (1.0 - k)
    if "tint" in p:
        t = 0.2 * p["tint"] * s
        out[..., 1] = out[..., 1] * (1.0 - t)
    return np.clip(out, 0.0, 1.0)


def _apply_hsv_ops(rgb: np.ndarray, p: dict, s: float) -> np.ndarray:
    needs = any(k in p for k in ("toward_hue", "sat_add", "lum_add", "toward_neutral"))
    if not needs:
        return rgb
    hsv = common.rgb_to_hsv(rgb)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if "toward_hue" in p:
        target = float(p["toward_hue"]["deg"])
        amount = float(p["toward_hue"]["amount"]) * s
        delta = ((target - h + 180.0) % 360.0) - 180.0
        h = (h + delta * amount) % 360.0
    if "toward_neutral" in p:
        sat = sat * (1.0 - float(p["toward_neutral"]) * s)
    if "sat_add" in p:
        sat = sat + float(p["sat_add"]) * s
    if "lum_add" in p:
        val = val + float(p["lum_add"]) * s
    hsv = np.stack([h, np.clip(sat, 0.0, 1.0), np.clip(val, 0.0, 1.0)], axis=-1)
    return np.clip(common.hsv_to_rgb(hsv), 0.0, 1.0)


def apply_region_action(
    rgb: np.ndarray, mask: np.ndarray, local_params: dict, strength: float,
    feather_radius: float = 6.0,
) -> np.ndarray:
    """Return rgb with local_params applied inside a feathered mask."""
    if not local_params or strength <= 0:
        return rgb
    fmask = common.feather_mask(mask, radius=feather_radius)
    adjusted = _apply_rgb_ops(rgb, local_params, strength, mask)
    adjusted = _apply_hsv_ops(adjusted, local_params, strength)
    m = fmask[..., None]
    return rgb * (1.0 - m) + adjusted * m


def render_plan(rgb: np.ndarray, plan: dict, masks: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Render a plan's local (non-GPT) actions. Returns (image, info)."""
    out = rgb.copy()
    applied, skipped = [], []
    for ra in plan.get("region_actions", []):
        if ra.get("executor") == "gpt_image_2" or not ra.get("local_params"):
            skipped.append(ra.get("action"))
            continue
        mask = masks.get(ra["region_id"])
        if mask is None:
            skipped.append(ra.get("action"))
            continue
        out = apply_region_action(out, mask, ra["local_params"], ra.get("strength", 0.5))
        applied.append(ra.get("action"))
    info = {
        "applied_actions": applied,
        "skipped_actions": skipped,
        "needs_gpt": plan.get("executor") == "gpt_image_2",
    }
    return out, info
