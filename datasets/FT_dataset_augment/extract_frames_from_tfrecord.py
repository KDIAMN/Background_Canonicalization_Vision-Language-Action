#!/usr/bin/env python3
"""
TFRecord에서 프레임 이미지 5개 추출.

Usage:
    /path/to/VLA_Adapter/vla_adapter_env/bin/python3.10 \
        /path/to/VLA_Adapter/datasets/FT_dataset_augment/extract_frames_from_tfrecord.py
"""

from pathlib import Path
import tensorflow_datasets as tfds
from PIL import Image

RLDS_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/delete_Me_RLDS_bg_masked_libero_object")
OUT_DIR  = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/delete_Me_frames")
OUT_DIR.mkdir(exist_ok=True)

ds = tfds.load(
    "delete_me_bg_masked_libero_object",
    data_dir=str(RLDS_DIR),
    split="train",
)

saved = 0
for episode in ds.take(1):
    steps = list(episode["steps"])
    # 균등 간격으로 5프레임 선택
    n = len(steps)
    indices = [int(i * (n - 1) / 4) for i in range(5)]  # 0, 25%, 50%, 75%, 100%

    for rank, idx in enumerate(indices):
        step = steps[idx]
        img = step["observation"]["image"].numpy()
        lang = step["language_instruction"].numpy().decode()
        is_first = step["is_first"].numpy()
        is_last  = step["is_last"].numpy()

        out_path = OUT_DIR / f"frame_{rank:02d}_step{idx:03d}.png"
        Image.fromarray(img).save(out_path)
        print(f"  [{rank}] step {idx:3d}  is_first={is_first}  is_last={is_last}  → {out_path.name}")
        saved += 1

print(f"\n저장 완료: {OUT_DIR}  ({saved}개)")
print(f"task: {lang}")
