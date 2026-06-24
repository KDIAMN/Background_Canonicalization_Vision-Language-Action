"""
bg_mask_processor.py

Inference-time background canonicalization — replicates exactly the same
transform applied during fine-tuning in FT_dataset_augment.py:

  1. GroundingDINO detects foreground ("robot arm . object") bounding boxes
  2. SAM2 segments the foreground from those boxes
  3. Background pixels → grayscale + brightness normalization + Gaussian blur

Usage:
    from eval.bg_mask_processor import BgMaskProcessor
    proc = BgMaskProcessor(device="cuda")
    masked_rgb = proc.process(frame_rgb)   # frame_rgb: H×W×3 uint8 numpy array
"""

import os
import tempfile

import cv2
import numpy as np
import torch

from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── paths (same checkpoints used during training augmentation) ─────────────
GDINO_CONFIG = (
    "/path/to/VLA_Adapter/vla_adapter_env/lib/python3.10/site-packages"
    "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GDINO_WEIGHTS = "/path/to/VLA_Adapter/checkpoints/groundingdino_swint_ogc.pth"
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_WEIGHTS  = "/path/to/VLA_Adapter/checkpoints/sam2.1_hiera_large.pt"

# ── hyper-params (must match FT_dataset_augment.py exactly) ───────────────
TEXT_PROMPT       = "robot arm . object"
BOX_THRESHOLD     = 0.2
TEXT_THRESHOLD    = 0.2
BRIGHTNESS_TARGET = 59
BLUR_KERNEL_SIZE  = 7


class BgMaskProcessor:
    """Stateful processor: load models once, call process() per frame."""

    def __init__(self, device: str = "cuda"):
        self.gdino = load_model(GDINO_CONFIG, GDINO_WEIGHTS)

        sam2 = build_sam2(SAM2_CONFIG, SAM2_WEIGHTS, device=device)
        self.predictor = SAM2ImagePredictor(sam2)

    # ── internal helpers (identical logic to FT_dataset_augment.py) ────────

    def _boxes_to_pixel(self, boxes: torch.Tensor, H: int, W: int) -> np.ndarray:
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack(
            [
                (cx - bw / 2) * W,
                (cy - bh / 2) * H,
                (cx + bw / 2) * W,
                (cy + bh / 2) * H,
            ],
            dim=1,
        ).numpy()

    def _get_background_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True = background pixel."""
        H, W = frame_rgb.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        try:
            image_source, image = load_image(tmp_path)
            boxes, _, _ = predict(
                model=self.gdino,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
        finally:
            os.unlink(tmp_path)

        if len(boxes) == 0:
            # nothing detected → treat whole image as background
            return np.ones((H, W), dtype=bool)

        self.predictor.set_image(image_source)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self._boxes_to_pixel(boxes, H, W),
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        fg_mask = np.zeros((H, W), dtype=bool)
        for m in masks:
            fg_mask |= m.astype(bool)

        return ~fg_mask  # invert: True = background

    def _clean_mask(self, bg_mask: np.ndarray) -> np.ndarray:
        mask_u8 = bg_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        return mask_u8.astype(bool)

    def _apply_canonicalization(
        self, frame_rgb: np.ndarray, bg_mask: np.ndarray
    ) -> np.ndarray:
        bg_mask = self._clean_mask(bg_mask)
        fg_mask = ~bg_mask

        out = frame_rgb.copy().astype(np.float32)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)

        bg_pixels = gray[bg_mask]
        if len(bg_pixels) > 0 and bg_pixels.mean() > 0:
            scale = BRIGHTNESS_TARGET / bg_pixels.mean()
            gray_3ch = np.clip(gray_3ch * scale, 0, 255)

        bg_mean = float(gray_3ch[bg_mask].mean()) if bg_mask.any() else 128.0
        blur_input = gray_3ch.copy()
        blur_input[fg_mask] = bg_mean  # fill foreground with bg mean before blurring

        blurred = cv2.GaussianBlur(
            blur_input, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0
        )
        out[bg_mask] = blurred[bg_mask]

        return np.clip(out, 0, 255).round().astype(np.uint8)

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Apply background canonicalization to one frame.

        Args:
            frame_rgb: H×W×3 uint8 numpy array in RGB order

        Returns:
            H×W×3 uint8 numpy array — foreground unchanged, background is
            grayscale + brightness-normalized + Gaussian-blurred
        """
        bg_mask = self._get_background_mask(frame_rgb)
        return self._apply_canonicalization(frame_rgb, bg_mask)
