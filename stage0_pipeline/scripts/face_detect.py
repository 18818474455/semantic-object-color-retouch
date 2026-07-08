"""Real face detection (OpenCV YuNet DNN) to replace the broken color-threshold
"skin" classifier.

Root-cause bug found: a pure YCbCr color threshold flagged 25-34% of pixels as
"skin" on these event photos (warm floor tiles, red LED bleed, wood tones all
pass the same threshold). That contaminated the per-class color statistics and
was the real reason the local semantic transfer looked dark/muddy — not a
tuning issue. Faces must be located geometrically first.

opencv-python-headless 5.x removed the legacy Haar CascadeClassifier bindings,
so this uses the bundled-model-free YuNet ONNX detector (~230KB, CPU, fast).
Model + weights are fetched into stage0_pipeline/assets/ (gitignored-friendly:
small, redistributable Apache-2.0 file from opencv_zoo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
MODEL_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        import cv2
        _detector = cv2.FaceDetectorYN_create(str(MODEL_PATH), "", (320, 320), score_threshold=0.6)
    return _detector


def detect_faces(rgb: np.ndarray, min_score: float = 0.6) -> list[tuple[int, int, int, int, float]]:
    """Returns list of (x, y, w, h, score) boxes in pixel coords of rgb."""
    import cv2
    det = _get_detector()
    bgr = (np.clip(rgb, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy()
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    ok, faces = det.detect(bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[-1]
        if score >= min_score:
            out.append((int(x), int(y), int(fw), int(fh), float(score)))
    return out


def _ycbcr_skin(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5
    return ((cr > 0.52) & (cr < 0.68) & (cb > 0.40) & (cb < 0.53) & (y > 0.15) & (y < 0.97)).astype(np.float32)


def skin_mask_from_faces(rgb: np.ndarray, faces: list[tuple[int, int, int, int, float]],
                         expand: float = 0.6) -> np.ndarray:
    """Skin likelihood mask: color-threshold pixels, gated to stay near a real
    detected face box (+expand for neck/forehead/ears/hairline). This prevents
    warm floors/backgrounds from ever being misread as skin.
    """
    h, w = rgb.shape[:2]
    face_gate = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh, _score) in faces:
        ex, ey = int(fw * expand), int(fh * expand)
        x0, y0 = max(0, x - ex), max(0, y - ey)
        x1, y1 = min(w, x + fw + ex), min(h, y + fh + int(ey * 1.8))  # a bit more room below for neck
        face_gate[y0:y1, x0:x1] = 1.0
    if face_gate.sum() == 0:
        return np.zeros((h, w), dtype=np.float32)
    color = _ycbcr_skin(rgb)
    return face_gate * color
