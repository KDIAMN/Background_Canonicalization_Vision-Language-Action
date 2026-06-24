import io
import os
import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from groundingdino.util.inference import load_model, predict, load_image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ──────────────────────────────────────────────
GDINO_CONFIG  = "/path/to/VLA_Adapter/vla_adapter_env/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "/path/to/VLA_Adapter/checkpoints/groundingdino_swint_ogc.pth"
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_WEIGHTS  = "/path/to/VLA_Adapter/checkpoints/sam2.1_hiera_large.pt"

TEXT_PROMPT    = "robot arm . object"
BOX_THRESHOLD  = 0.2
TEXT_THRESHOLD = 0.2

BRIGHTNESS_TARGET = 59
BLUR_KERNEL_SIZE  = 7

IN_DATASET_DIR  = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object/data/chunk-000/file-013.parquet")
OUT_DATASET_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/check_before_augment")

# 체크용: 처리할 parquet 파일 1개
CHECK_PARQUET = Path("data/chunk-000/file-000.parquet")
# ──────────────────────────────────────────────


def boxes_to_pixel(boxes, H, W):
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([
        (cx - bw / 2) * W, (cy - bh / 2) * H,
        (cx + bw / 2) * W, (cy + bh / 2) * H,
    ], dim=1).numpy()


def union_masks(masks, H, W):
    result = np.zeros((H, W), dtype=bool)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    for m in masks:
        result |= m.astype(bool)
    return result


def get_background_mask(frame_rgb: np.ndarray, gdino, predictor):
    H, W = frame_rgb.shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    try:
        image_source, image = load_image(tmp_path)
        boxes, _, _ = predict(
            model=gdino, image=image, caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        )
    finally:
        os.unlink(tmp_path)

    if len(boxes) == 0:
        return np.ones((H, W), dtype=bool)

    predictor.set_image(image_source)
    masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=boxes_to_pixel(boxes, H, W), multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    fg_mask = union_masks(masks, H, W)
    return ~fg_mask


def clean_mask(bg_mask: np.ndarray) -> np.ndarray:
    mask_u8 = bg_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel)
    return mask_u8.astype(bool)


def apply_bg_canonicalization(frame_rgb: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    bg_mask = clean_mask(bg_mask)
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
    blur_input[fg_mask] = bg_mean

    blurred = cv2.GaussianBlur(blur_input, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    out[bg_mask] = blurred[bg_mask]

    return np.clip(out, 0, 255).round().astype(np.uint8)


def perturb_image_bytes(raw_bytes: bytes, gdino, predictor) -> bytes:
    frame_rgb = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    bg_mask = get_background_mask(frame_rgb, gdino, predictor)
    result = apply_bg_canonicalization(frame_rgb, bg_mask)
    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    return buf.getvalue()


def process_parquet(src: Path, dst: Path, gdino, predictor):
    df = pd.read_parquet(src)
    new_images = []
    for i, entry in enumerate(df["observation.images.image"]):
        perturbed_bytes = perturb_image_bytes(entry["bytes"], gdino, predictor)
        new_images.append({"bytes": perturbed_bytes, "path": entry["path"]})
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(df)} rows")

    df["observation.images.image"] = new_images
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def main():
    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("모델 로드 중...")
    gdino     = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
    sam2      = build_sam2(SAM2_CONFIG, SAM2_WEIGHTS)
    predictor = SAM2ImagePredictor(sam2)
    print("완료\n")

    src = IN_DATASET_DIR / CHECK_PARQUET
    dst = OUT_DATASET_DIR / CHECK_PARQUET
    print(f"처리 파일: {src}")
    process_parquet(src, dst, gdino, predictor)
    print(f"\n완료 → {dst}")


if __name__ == "__main__":
    main()
