"""Region detection providers.

The pipeline depends on the abstract `RegionProvider.detect` contract only, so
the Stage 0 heuristic provider can be swapped for a Grounding DINO + SAM2
provider in M2 without touching downstream metrics / plan / render code.

A Region is a plain dict:
    {
        "region_id": str,
        "object_type": str,        # matches actions.v1.json object keys
        "role": str,
        "bbox": [x0, y0, x1, y1],  # normalized 0..1
        "mask": np.ndarray(H,W) float 0..1,   # in-memory; saved separately
        "confidence": float,
        "mask_quality": float,
        "protect_level": str,
        "source": str,             # provenance, e.g. "heuristic", "grounding_dino+sam2"
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import common


class RegionProvider(ABC):
    @abstractmethod
    def detect(self, rgb: np.ndarray, meta: dict) -> list[dict]:
        ...


def _global_region(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    return {
        "region_id": "global",
        "object_type": "global",
        "role": "whole_image",
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "mask": np.ones((h, w), dtype=np.float32),
        "confidence": 1.0,
        "mask_quality": 1.0,
        "protect_level": "none",
        "source": "constant",
    }


class HeuristicRegionProvider(RegionProvider):
    """No-ML placeholder. Detects a sky region by color + vertical prior.

    This exists so the full loop (metrics -> plan -> preview -> review) runs
    today without downloading Grounding DINO / SAM2. It is intentionally
    conservative: it only claims 'sky', and reports a mask_quality it can
    justify, so the confidence gates behave realistically.
    """

    def __init__(self, sky_top_fraction: float = 0.6, min_sky_pixels_frac: float = 0.02):
        self.sky_top_fraction = sky_top_fraction
        self.min_sky_pixels_frac = min_sky_pixels_frac

    def detect(self, rgb: np.ndarray, meta: dict) -> list[dict]:
        regions = [_global_region(rgb)]
        sky = self._detect_sky(rgb)
        if sky is not None:
            regions.append(sky)
        return regions

    def _detect_sky(self, rgb: np.ndarray) -> dict | None:
        h, w = rgb.shape[:2]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = common.luminance(rgb)

        # vertical prior: 1 at top, fading to 0 by sky_top_fraction of height
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        vprior = np.clip(1.0 - yy / self.sky_top_fraction, 0.0, 1.0)
        vprior = np.repeat(vprior, w, axis=1)

        bluish = b >= (r - 0.02)              # blue >= red (blue or neutral sky)
        bright = lum > 0.45                    # bright enough to be sky/overcast
        near_white = (r > 0.6) & (g > 0.6) & (b > 0.6)
        candidate = (bluish & bright) | near_white
        score = candidate.astype(np.float32) * vprior

        mask_bool = score > 0.15
        mask_bool = common.morph_open(mask_bool, size=3)

        frac = mask_bool.mean()
        if frac < self.min_sky_pixels_frac:
            return None

        # mask quality proxy: how top-connected and how clean the region is.
        top_band = mask_bool[: max(1, h // 20)].mean()
        quality = float(np.clip(0.4 + 0.4 * top_band + 0.2 * min(frac / 0.25, 1.0), 0.0, 1.0))
        # confidence: bluish sky is easier to trust than pure-white overcast.
        bluish_frac = float((bluish & mask_bool).mean() / max(frac, 1e-6))
        confidence = float(np.clip(0.5 + 0.4 * bluish_frac, 0.0, 0.95))

        ys, xs = np.where(mask_bool)
        bbox = [
            round(float(xs.min()) / w, 4), round(float(ys.min()) / h, 4),
            round(float(xs.max() + 1) / w, 4), round(float(ys.max() + 1) / h, 4),
        ]
        return {
            "region_id": "r_sky_001",
            "object_type": "sky",
            "role": "background",
            "bbox": bbox,
            "mask": mask_bool.astype(np.float32),
            "confidence": round(confidence, 4),
            "mask_quality": round(quality, 4),
            "protect_level": "medium",
            "source": "heuristic",
        }


def get_provider(name: str = "heuristic") -> RegionProvider:
    if name == "heuristic":
        return HeuristicRegionProvider()
    raise ValueError(
        f"Unknown region provider '{name}'. "
        f"M2 will add 'grounding_dino_sam2' here."
    )
