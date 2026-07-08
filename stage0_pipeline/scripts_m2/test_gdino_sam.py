"""M2 spike: verify Grounding DINO (text-prompted boxes) + SAM (box -> precise
mask) actually separates sky/grass/LED-stage/clothing into distinct regions on
the exact two images that broke the old 4-class color-heuristic classifier in
Phase D (outdoor_sky ground blob artifact, stage_led_mixed orange-wash).

Must run under .venv-m2 (has torch + transformers); the main pipeline venv
does not have these heavy deps installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "m2_spike"
OUT.mkdir(parents=True, exist_ok=True)

TEXT_QUERIES = "sky. grass. tree. water. clothing. led screen. building. floor. flag."

CASES = {
    "sky_grass": "/Volumes/未命名/大模型/原图1/DSC05360(1).JPG",
    "led_stage": "/Volumes/未命名/大模型/原图1/DSC04541.JPG",
}


def load_models():
    print(f"device={DEVICE}")
    dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny").to(DEVICE).eval()
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE).eval()
    return dino_proc, dino_model, sam_proc, sam_model


def detect_boxes(img: Image.Image, dino_proc, dino_model, box_thresh=0.30, text_thresh=0.25):
    inputs = dino_proc(images=img, text=TEXT_QUERIES, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_proc.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[img.size[::-1]],
    )[0]
    return list(zip(results["boxes"].tolist(), results["labels"], results["scores"].tolist()))


def segment_box(img: Image.Image, box, sam_proc, sam_model) -> np.ndarray:
    inputs = sam_proc(img, input_boxes=[[box]], return_tensors="pt")
    original_sizes = inputs.pop("original_sizes")
    reshaped_sizes = inputs.pop("reshaped_input_sizes")
    def _to_device(v):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            v = v.to(torch.float32)
        return v.to(DEVICE) if torch.is_tensor(v) else v
    inputs = {k: _to_device(v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), original_sizes, reshaped_sizes
    )[0]
    # masks: (num_boxes, num_pred_masks, H, W) bool; pick best-scoring mask per box
    scores = outputs.iou_scores.cpu()[0, 0]
    best = int(scores.argmax())
    return masks[0, best].numpy()


def main() -> int:
    dino_proc, dino_model, sam_proc, sam_model = load_models()
    for name, path in CASES.items():
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024))
        print(f"\n=== {name} ({img.size}) ===")
        dets = detect_boxes(img, dino_proc, dino_model)
        print(f"{len(dets)} boxes:")
        vis = np.array(img, dtype=np.float32) / 255.0
        h, w = vis.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        rng_colors = [(1, 0.3, 0.3), (0.3, 1, 0.3), (0.3, 0.5, 1), (1, 1, 0.3),
                      (1, 0.3, 1), (0.3, 1, 1), (1, 0.6, 0.2), (0.7, 0.7, 0.7)]
        for i, (box, label, score) in enumerate(dets):
            print(f"  {label:12s} score={score:.3f} box={[round(v) for v in box]}")
            mask = segment_box(img, box, sam_proc, sam_model)
            color = rng_colors[i % len(rng_colors)]
            for k in range(3):
                overlay[..., k] = np.where(mask, color[k], overlay[..., k])
        blended = np.clip(vis * 0.4 + overlay * 0.6, 0, 1)
        Image.fromarray((blended * 255).astype(np.uint8)).save(OUT / f"{name}_masks.jpg", quality=92)
        Image.fromarray((vis * 255).astype(np.uint8)).save(OUT / f"{name}_orig.jpg", quality=92)
    print(f"\noutputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
