"""Compute per-region color metrics for one image.

Output schema (regions/<image_id>.json):
{
  "image_id": ...,
  "scene": {"type": ..., "lighting": ..., "source": "heuristic"},
  "global_metrics": {...},
  "routing_metrics": {...},
  "regions": [ {region fields + color_metrics}, ... ]
}
"""
from __future__ import annotations

import numpy as np

import common
from region_provider import RegionProvider, get_provider


def _region_color_metrics(rgb: np.ndarray, mask: np.ndarray) -> dict:
    sel = mask > 0.5
    n = int(sel.sum())
    if n < 4:
        return {"pixel_count": n}
    lum = common.luminance(rgb)
    hsv = common.rgb_to_hsv(rgb)
    lab = common.rgb_to_lab(rgb)

    lum_sel = lum[sel]
    hsv_sel = hsv[sel]
    lab_sel = lab[sel]
    rgb_sel = rgb[sel]

    pct = lambda a, q: round(float(np.percentile(a, q)), 4)
    return {
        "pixel_count": n,
        "area_frac": round(n / mask.size, 4),
        "rgb_mean": common.to_float_list(rgb_sel.mean(axis=0)),
        "lab_mean": common.to_float_list(lab_sel.mean(axis=0)),
        "hue_mean_deg": round(float(_circular_mean_deg(hsv_sel[:, 0])), 2),
        "saturation_mean": round(float(hsv_sel[:, 1].mean()), 4),
        "value_mean": round(float(hsv_sel[:, 2].mean()), 4),
        "brightness": round(float(lum_sel.mean() * 255.0), 2),
        "lum_p05": pct(lum_sel * 255.0, 5),
        "lum_p50": pct(lum_sel * 255.0, 50),
        "lum_p95": pct(lum_sel * 255.0, 95),
        "clip_high_pct": round(float((lum_sel > 0.98).mean() * 100.0), 3),
        "clip_low_pct": round(float((lum_sel < 0.02).mean() * 100.0), 3),
        "colorfulness": round(common.colorfulness(rgb, mask), 4),
        "sharpness_proxy": round(common.sharpness_proxy(rgb, mask), 4),
        "green_magenta_a": round(float(lab_sel[:, 1].mean()), 3),
        "warm_cool_b": round(float(lab_sel[:, 2].mean()), 3),
    }


def _circular_mean_deg(hues_deg: np.ndarray) -> float:
    rad = np.deg2rad(hues_deg)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def classify_scene(meta: dict, global_metrics: dict, thresholds: dict) -> dict:
    """Heuristic scene/lighting classification. Placeholder for the VLM gate."""
    g = thresholds["scene_gate_heuristics"]
    bucket = meta.get("stage0_bucket") or meta.get("bucket")
    brightness = global_metrics.get("brightness", 128.0)
    warm_b = global_metrics.get("warm_cool_b", 0.0)  # Lab b: + is warm/yellow

    lighting = "daylight"
    scene_type = "general"
    if bucket == g["stage_bucket"]:
        scene_type = "stage_led_mixed"
        lighting = "stage_mixed"
    elif brightness < g["night_brightness_below"]:
        scene_type = "night_or_dark"
        lighting = "night"
    elif warm_b > g["sunset_warm_cast_above"] * 100.0 and brightness < g["sunset_brightness_below"]:
        scene_type = "sunset"
        lighting = "sunset"
    elif bucket == "outdoor_sky":
        scene_type = "outdoor"
        lighting = "daylight"
    elif bucket == "person_event":
        scene_type = "event_portrait"
        lighting = "daylight"

    return {"type": scene_type, "lighting": lighting, "bucket": bucket, "source": "heuristic"}


def build_region_metrics(
    rgb: np.ndarray,
    meta: dict,
    thresholds: dict,
    provider: RegionProvider | None = None,
) -> dict:
    provider = provider or get_provider("heuristic")
    regions = provider.detect(rgb, meta)

    out_regions = []
    global_metrics: dict = {}
    for reg in regions:
        cm = _region_color_metrics(rgb, reg["mask"])
        entry = {k: v for k, v in reg.items() if k != "mask"}
        entry["color_metrics"] = cm
        out_regions.append(entry)
        if reg["object_type"] == "global":
            global_metrics = cm

    scene = classify_scene(meta, global_metrics, thresholds)
    routing_metrics = {
        "clip_high_pct": global_metrics.get("clip_high_pct", 0.0),
        "clip_low_pct": global_metrics.get("clip_low_pct", 0.0),
        "sharpness_proxy": global_metrics.get("sharpness_proxy", 0.0),
        "mixed_light_score": 0.8 if scene["type"] == "stage_led_mixed" else 0.1,
    }
    return {
        "image_id": meta["image_id"],
        "source_path": meta.get("source_path"),
        "scene": scene,
        "global_metrics": global_metrics,
        "routing_metrics": routing_metrics,
        "regions": out_regions,
        "_masks": {r["region_id"]: r["mask"] for r in regions},  # in-memory only
    }
