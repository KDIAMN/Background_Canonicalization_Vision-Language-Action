#!/usr/bin/env python3
"""
Convert LeRobot-format parquet dataset to RLDS (TFDS) format for VLA-Adapter fine-tuning.

Input:  bg_masked_libero_object  (LeRobot v3.0 parquet)
Output: RLDS_bg_masked_libero_object  (TFDS TFRecord, ReadOnlyBuilder-compatible)

State layout (8-dim) split to match libero_object_no_noops in configs.py:
  EEF_state    (6-dim): [x, y, z, roll, pitch, yaw]
  gripper_state (2-dim): [gripper_l, gripper_r]

Wrist camera is intentionally excluded.

Usage:
    /path/to/VLA_Adapter/vla_adapter_env/bin/python3.10 \
        /path/to/VLA_Adapter/datasets/FT_dataset_augment/convert_lerobot_to_rlds.py
"""

import io
from itertools import groupby
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# ──────────────────────────────────────────────
IN_DATASET_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object")
OUT_DATA_DIR   = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/RLDS_bg_masked_libero_object")
# ──────────────────────────────────────────────


class BgMaskedLiberoObject(tfds.core.GeneratorBasedBuilder):
    """RLDS dataset: LIBERO-Object with grayscale+blur background masking (main camera only)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "LIBERO-Object with grayscale+blur background. Wrist camera excluded.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "LIBERO-Object dataset where the background of the main camera "
                "has been replaced with a grayscale+blur version. Wrist camera excluded."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        # main camera only, JPEG-encoded in TFRecord
                        "image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                        # EEF position + orientation: [x, y, z, roll, pitch, yaw]
                        "EEF_state": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                        # gripper: [gripper_l, gripper_r]
                        "gripper_state": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                    }),
                    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first":    tfds.features.Scalar(dtype=tf.bool),
                    "is_last":     tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "reward":      tfds.features.Scalar(dtype=np.float32),
                    "discount":    tfds.features.Scalar(dtype=np.float32),
                }),
            }),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"train": self._generate_examples(IN_DATASET_DIR)}

    def _generate_examples(self, data_root: Path):
        # ── 1. Load episode metadata ────────────────────────────────────────
        ep_meta_files = sorted((data_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        ep_rows = []
        for f in ep_meta_files:
            d = pq.read_table(f).to_pydict()
            for i in range(len(d["episode_index"])):
                ep_rows.append({
                    "episode_index": d["episode_index"][i],
                    "chunk_index":   d["data/chunk_index"][i],
                    "file_index":    d["data/file_index"][i],
                    "from_index":    d["dataset_from_index"][i],
                    "to_index":      d["dataset_to_index"][i],
                    "task":          d["tasks"][i][0],
                })
        ep_rows.sort(key=lambda r: (r["chunk_index"], r["file_index"], r["episode_index"]))

        total_episodes = len(ep_rows)
        processed = 0

        # ── 2. Process parquet file by file to avoid loading everything at once ─
        for (chunk_idx, file_idx), group in groupby(
            ep_rows, key=lambda r: (r["chunk_index"], r["file_index"])
        ):
            episodes_in_file = list(group)
            parquet_path = (
                data_root / "data"
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.parquet"
            )
            print(f"  Reading {parquet_path.name}  ({len(episodes_in_file)} episodes)")
            raw = pq.read_table(parquet_path).to_pydict()

            # global frame index → row position within this file
            index_to_pos = {raw["index"][i]: i for i in range(len(raw["index"]))}

            for ep in episodes_in_file:
                ep_idx   = ep["episode_index"]
                lang     = ep["task"]
                n_frames = ep["to_index"] - ep["from_index"]
                steps    = []

                for j, frame_global_idx in enumerate(range(ep["from_index"], ep["to_index"])):
                    pos = index_to_pos[frame_global_idx]

                    # Decode PNG bytes → numpy RGB array (TFDS re-encodes as JPEG)
                    png_bytes = raw["observation.images.image"][pos]["bytes"]
                    img_array = np.array(Image.open(io.BytesIO(png_bytes)))  # (256,256,3) uint8

                    state  = np.array(raw["observation.state"][pos], dtype=np.float32)
                    action = np.array(raw["action"][pos], dtype=np.float32)

                    steps.append({
                        "observation": {
                            "image": img_array,
                            "EEF_state":     state[:6],   # [x, y, z, roll, pitch, yaw]
                            "gripper_state": state[6:],   # [gripper_l, gripper_r]
                        },
                        "action":               action,
                        "language_instruction": lang,
                        "is_first":    j == 0,
                        "is_last":     j == n_frames - 1,
                        "is_terminal": j == n_frames - 1,
                        "reward":      float(j == n_frames - 1),
                        "discount":    1.0,
                    })

                processed += 1
                if processed % 50 == 0 or processed == total_episodes:
                    print(f"  [{processed}/{total_episodes}] episode {ep_idx}  "
                          f"({n_frames} frames, '{lang[:40]}')")

                yield ep_idx, {"steps": steps}


if __name__ == "__main__":
    print(f"Input  : {IN_DATASET_DIR}")
    print(f"Output : {OUT_DATA_DIR}")
    print()

    builder = BgMaskedLiberoObject(data_dir=str(OUT_DATA_DIR))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(verify_ssl=False)
    )

    print()
    print("=" * 60)
    print("Conversion complete.")
    print(f"Dataset written to:")
    print(f"  {OUT_DATA_DIR}/bg_masked_libero_object/1.0.0/")
    print()
    print("VLA-Adapter fine-tuning args:")
    print(f"  --data_root_dir {OUT_DATA_DIR}")
    print(f"  --dataset_name  bg_masked_libero_object")
    print("=" * 60)
