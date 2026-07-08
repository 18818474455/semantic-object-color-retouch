"""M2: Grounding DINO (text-prompted boxes) + SAM (box -> precise mask) region
provider. Replaces the Phase-D heuristic 4-class color classifier, which broke
on scenes it wasn't hand-tuned for (sky/cloud edge artifacts, whole-image
color-wash when one color statistically dominated a scene).

Must run under .venv-m2 (torch + transformers installed there).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TEXT_QUERIES = ("sky. grass. tree. water. clothing. led screen. led wall. stage backdrop. "
                "spotlight. building. floor. flag. skin.")

# Grounding DINO confuses saturated, high-frequency-lit indoor LED walls with
# "sky" (both are large bright regions touching the top of the frame). Real
# atmospheric sky is smooth (low local-gradient texture) and desaturated; an
# LED wall is a lit dot-pattern (high texture) and highly saturated. Measured
# on our own failing case: real sky texture_std=0.0013 sat=0.19 vs LED-wall-
# mislabeled-as-sky texture_std=0.045 sat=0.86 — >30x apart, safe margin.
SKY_MAX_TEXTURE_STD = 0.012
SKY_MAX_MEAN_SAT = 0.5


def _sky_plausible(rgb: np.ndarray, mask: np.ndarray) -> bool:
    sel = mask > 0.5
    if sel.sum() < 20:
        return False
    lum = common.luminance(rgb)
    gy, gx = np.gradient(lum)
    texture_std = float(np.sqrt(gx[sel] ** 2 + gy[sel] ** 2).std())
    mean_sat = float(common.rgb_to_hsv(rgb)[..., 1][sel].mean())
    return texture_std <= SKY_MAX_TEXTURE_STD and mean_sat <= SKY_MAX_MEAN_SAT

_cache: dict = {}


def _load():
    if not _cache:
        _cache["dino_proc"] = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        _cache["dino_model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny").to(DEVICE).eval()
        _cache["sam_proc"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
        _cache["sam_model"] = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE).eval()
    return _cache["dino_proc"], _cache["dino_model"], _cache["sam_proc"], _cache["sam_model"]


def _to_device(v):
    if torch.is_tensor(v) and v.dtype == torch.float64:
        v = v.to(torch.float32)
    return v.to(DEVICE) if torch.is_tensor(v) else v


def _segment_box(img: Image.Image, box, sam_proc, sam_model) -> np.ndarray:
    inputs = sam_proc(img, input_boxes=[[box]], return_tensors="pt")
    original_sizes = inputs.pop("original_sizes")
    reshaped_sizes = inputs.pop("reshaped_input_sizes")
    inputs = {k: _to_device(v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), original_sizes, reshaped_sizes)[0]
    scores = outputs.iou_scores.cpu()[0, 0]
    best = int(scores.argmax())
    return masks[0, best].numpy()


def detect_classes(rgb: np.ndarray, box_thresh: float = 0.30, text_thresh: float = 0.25,
                   text_queries: str = TEXT_QUERIES) -> dict[str, np.ndarray]:
    """rgb: float [0,1] HxWx3. Returns {label: float mask HxW in [0,1]}, merged
    across multiple boxes of the same label (element-wise max)."""
    dino_proc, dino_model, sam_proc, sam_model = _load()
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    inputs = dino_proc(images=img, text=text_queries, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_proc.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[img.size[::-1]],
    )[0]
    h, w = rgb.shape[:2]
    masks: dict[str, np.ndarray] = {}
    for box, label, score in zip(results["boxes"].tolist(), results["labels"], results["scores"].tolist()):
        m = _segment_box(img, box, sam_proc, sam_model).astype(np.float32)
        label = str(label).strip()
        if label in masks:
            masks[label] = np.maximum(masks[label], m)
        else:
            masks[label] = m

    if "sky" in masks and not _sky_plausible(rgb, masks["sky"]):
        del masks["sky"]  # likely a mislabeled LED wall / indoor light source
    return masks
