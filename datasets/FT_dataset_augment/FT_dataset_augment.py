import io
import os
import shutil
import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
import multiprocessing as mp
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

IN_DATASET_DIR  = Path("/path/to/smolVLA/lerobot/data/lerobot/libero_object_image")
OUT_DATASET_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object")

NUM_GPUS = 4
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
            print(f"      {i+1}/{len(df)} rows", flush=True)

    df["observation.images.image"] = new_images
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def worker(gpu_id: int, parquet_files: list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] 모델 로드 중... ({len(parquet_files)}개 파일 담당)", flush=True)
    gdino     = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
    sam2      = build_sam2(SAM2_CONFIG, SAM2_WEIGHTS)
    predictor = SAM2ImagePredictor(sam2)
    print(f"[GPU {gpu_id}] 모델 로드 완료", flush=True)

    for file_idx, (src, dst) in enumerate(parquet_files):
        src, dst = Path(src), Path(dst)
        print(f"[GPU {gpu_id}] [{file_idx+1}/{len(parquet_files)}] {src.name}", flush=True)
        process_parquet(src, dst, gdino, predictor)
        print(f"[GPU {gpu_id}] 저장 완료: {dst.name}", flush=True)

    print(f"[GPU {gpu_id}] 전체 완료", flush=True)


def main():
    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((IN_DATASET_DIR / "data").rglob("*.parquet"))
    print(f"총 parquet 파일 수: {len(parquet_files)}")

    # GPU별 파일 분배
    jobs_per_gpu = [[] for _ in range(NUM_GPUS)]
    for i, src in enumerate(parquet_files):
        dst = OUT_DATASET_DIR / src.relative_to(IN_DATASET_DIR)
        jobs_per_gpu[i % NUM_GPUS].append((str(src), str(dst)))

    for i, jobs in enumerate(jobs_per_gpu):
        print(f"  GPU {i}: {len(jobs)}개 파일")

    # 멀티프로세스 실행
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = ctx.Process(target=worker, args=(gpu_id, jobs_per_gpu[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # meta/, README.md 복사
    for src in IN_DATASET_DIR.rglob("*"):
        if src.is_file() and src.suffix != ".parquet":
            dst = OUT_DATASET_DIR / src.relative_to(IN_DATASET_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"\n전체 완료 → {OUT_DATASET_DIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
