#!/usr/bin/env python3
"""
TFRecord 변환 정확성 검증.
원본 parquet 값과 TFRecord 값을 프레임 단위로 비교.

Usage:
    /path/to/VLA_Adapter/vla_adapter_env/bin/python3.10 \
        /path/to/VLA_Adapter/datasets/FT_dataset_augment/verify_rlds_conversion.py
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
from PIL import Image

PARQUET_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object")
RLDS_DIR    = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/delete_Me_RLDS_bg_masked_libero_object")
IMG_SAVE    = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/delete_Me_verify_image.png")


# ── 1. 원본 parquet에서 에피소드 0 로드 ────────────────────────────
print("=" * 60)
print("[1] 원본 parquet 읽기")

ep_meta = sorted((PARQUET_DIR / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0]
d = pq.read_table(ep_meta).to_pydict()

ep0 = {
    "chunk_index": d["data/chunk_index"][0],
    "file_index":  d["data/file_index"][0],
    "from_index":  d["dataset_from_index"][0],
    "to_index":    d["dataset_to_index"][0],
    "task":        d["tasks"][0][0],
}
n_frames_orig = ep0["to_index"] - ep0["from_index"]

parquet_path = (
    PARQUET_DIR / "data"
    / f"chunk-{ep0['chunk_index']:03d}"
    / f"file-{ep0['file_index']:03d}.parquet"
)
raw = pq.read_table(parquet_path).to_pydict()
index_to_pos = {raw["index"][i]: i for i in range(len(raw["index"]))}

pos0 = index_to_pos[ep0["from_index"]]       # 첫 번째 프레임
pos_last = index_to_pos[ep0["to_index"] - 1] # 마지막 프레임

orig_state_first  = np.array(raw["observation.state"][pos0],    dtype=np.float32)
orig_action_first = np.array(raw["action"][pos0],               dtype=np.float32)
orig_state_last   = np.array(raw["observation.state"][pos_last], dtype=np.float32)
orig_img_bytes    = raw["observation.images.image"][pos0]["bytes"]
orig_img          = np.array(Image.open(io.BytesIO(orig_img_bytes)))

print(f"  task        : {ep0['task']}")
print(f"  frames      : {n_frames_orig}")
print(f"  state[0]    : {orig_state_first}")
print(f"  action[0]   : {orig_action_first}")
print(f"  state[-1]   : {orig_state_last}")
print(f"  image shape : {orig_img.shape}  dtype={orig_img.dtype}")


# ── 2. TFRecord에서 에피소드 0 로드 ────────────────────────────────
print()
print("[2] TFRecord 읽기")

ds = tfds.load(
    "delete_me_bg_masked_libero_object",
    data_dir=str(RLDS_DIR),
    split="train",
)

step_list = list(list(ds.take(1))[0]["steps"])
n_frames_rlds = len(step_list)
first = step_list[0]
last  = step_list[-1]

rlds_eef_first     = first["observation"]["EEF_state"].numpy()
rlds_gripper_first = first["observation"]["gripper_state"].numpy()
rlds_action_first  = first["action"].numpy()
rlds_state_first   = np.concatenate([rlds_eef_first, rlds_gripper_first])  # 재결합
rlds_img           = first["observation"]["image"].numpy()
rlds_lang          = first["language_instruction"].numpy().decode()

print(f"  task        : {rlds_lang}")
print(f"  frames      : {n_frames_rlds}")
print(f"  EEF_state[0]: {rlds_eef_first}  (shape={rlds_eef_first.shape})")
print(f"  gripper[0]  : {rlds_gripper_first}  (shape={rlds_gripper_first.shape})")
print(f"  action[0]   : {rlds_action_first}  (shape={rlds_action_first.shape})")
print(f"  image shape : {rlds_img.shape}  dtype={rlds_img.dtype}")


# ── 3. 수치 비교 ────────────────────────────────────────────────────
print()
print("[3] 수치 비교")

state_ok   = np.allclose(orig_state_first, rlds_state_first, atol=1e-5)
action_ok  = np.allclose(orig_action_first, rlds_action_first, atol=1e-5)
frames_ok  = (n_frames_orig == n_frames_rlds)
is_first_ok = bool(first["is_first"].numpy())
is_last_ok  = bool(last["is_last"].numpy())

print(f"  프레임 수 일치  : {'OK' if frames_ok  else f'FAIL  orig={n_frames_orig} rlds={n_frames_rlds}'}")
print(f"  state[0] 일치   : {'OK' if state_ok   else 'FAIL'}")
if not state_ok:
    print(f"    orig  : {orig_state_first}")
    print(f"    rlds  : {rlds_state_first}")
    print(f"    diff  : {np.abs(orig_state_first - rlds_state_first)}")
print(f"  action[0] 일치  : {'OK' if action_ok  else 'FAIL'}")
if not action_ok:
    print(f"    orig  : {orig_action_first}")
    print(f"    rlds  : {rlds_action_first}")
print(f"  is_first[0]     : {'OK' if is_first_ok else 'FAIL'}  (값={is_first_ok})")
print(f"  is_last[-1]     : {'OK' if is_last_ok  else 'FAIL'}  (값={is_last_ok})")


# ── 4. 이미지 비교 및 저장 ─────────────────────────────────────────
print()
print("[4] 이미지 확인")

# TFDS가 JPEG 재인코딩하므로 픽셀이 완전히 동일하진 않음 — 평균 오차만 확인
img_diff = np.abs(orig_img.astype(float) - rlds_img.astype(float))
print(f"  원본 이미지 평균 픽셀값  : {orig_img.mean():.1f}")
print(f"  TFRecord 이미지 평균     : {rlds_img.mean():.1f}")
print(f"  픽셀 평균 절대 오차      : {img_diff.mean():.2f}  (JPEG 재인코딩으로 약간 차이 발생은 정상)")

# 원본(왼쪽)과 TFRecord(오른쪽) 나란히 저장
side_by_side = np.concatenate([orig_img, rlds_img], axis=1)
Image.fromarray(side_by_side).save(IMG_SAVE)
print(f"  이미지 저장 → {IMG_SAVE}")
print(f"  (왼쪽=원본 parquet, 오른쪽=TFRecord 복원 이미지)")


# ── 5. 최종 결과 ────────────────────────────────────────────────────
print()
print("=" * 60)
all_ok = frames_ok and state_ok and action_ok and is_first_ok and is_last_ok
if all_ok:
    print("결과: 모든 검증 통과 ✓  → 본 변환 실행 가능")
else:
    print("결과: 검증 실패 항목 있음 — 위 FAIL 항목 확인 필요")
print("=" * 60)
