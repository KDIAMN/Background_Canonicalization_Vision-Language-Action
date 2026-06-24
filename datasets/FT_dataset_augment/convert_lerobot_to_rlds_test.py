#!/usr/bin/env python3
"""
단일 에피소드 변환 테스트. 결과는 delete_Me_RLDS_bg_masked_libero_object에 저장됨.

Usage:
    /path/to/VLA_Adapter/vla_adapter_env/bin/python3.10 \
        /path/to/VLA_Adapter/datasets/FT_dataset_augment/convert_lerobot_to_rlds_test.py
"""

import io
from itertools import groupby
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

IN_DATASET_DIR = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object")
OUT_DATA_DIR   = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/delete_Me_RLDS_bg_masked_libero_object")


class DeleteMeBgMaskedLiberoObject(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Test: 1 episode only."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Test conversion — 1 episode.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image":         tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                        "EEF_state":     tfds.features.Tensor(shape=(6,), dtype=np.float32),
                        "gripper_state": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                    }),
                    "action":               tfds.features.Tensor(shape=(7,), dtype=np.float32),
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

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(IN_DATASET_DIR)}

    def _generate_examples(self, data_root: Path):
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

        # 첫 번째 에피소드만 처리
        first_ep = ep_rows[0]
        chunk_idx = first_ep["chunk_index"]
        file_idx  = first_ep["file_index"]

        parquet_path = (
            data_root / "data"
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.parquet"
        )
        print(f"  Reading {parquet_path}")
        raw = pq.read_table(parquet_path).to_pydict()
        index_to_pos = {raw["index"][i]: i for i in range(len(raw["index"]))}

        ep_idx   = first_ep["episode_index"]
        lang     = first_ep["task"]
        n_frames = first_ep["to_index"] - first_ep["from_index"]
        steps    = []

        for j, frame_global_idx in enumerate(range(first_ep["from_index"], first_ep["to_index"])):
            pos = index_to_pos[frame_global_idx]

            png_bytes = raw["observation.images.image"][pos]["bytes"]
            img_array = np.array(Image.open(io.BytesIO(png_bytes)))

            state  = np.array(raw["observation.state"][pos], dtype=np.float32)
            action = np.array(raw["action"][pos], dtype=np.float32)

            steps.append({
                "observation": {
                    "image":         img_array,
                    "EEF_state":     state[:6],
                    "gripper_state": state[6:],
                },
                "action":               action,
                "language_instruction": lang,
                "is_first":    j == 0,
                "is_last":     j == n_frames - 1,
                "is_terminal": j == n_frames - 1,
                "reward":      float(j == n_frames - 1),
                "discount":    1.0,
            })

        print(f"  Episode {ep_idx}: {n_frames} frames, '{lang}'")
        yield ep_idx, {"steps": steps}


def verify(data_dir: Path):
    print("\n" + "=" * 60)
    print("검증 시작")

    ds = tfds.load(
        "delete_me_bg_masked_libero_object",
        data_dir=str(data_dir),
        split="train",
    )

    for episode in ds.take(1):
        # steps는 tf.data.Dataset — 리스트로 수집
        step_list = list(episode["steps"])
        n = len(step_list)
        first = step_list[0]
        last  = step_list[-1]

        print(f"  에피소드 프레임 수   : {n}")
        print(f"  action shape        : {first['action'].shape}  (기대: (7,))")
        print(f"  EEF_state shape     : {first['observation']['EEF_state'].shape}  (기대: (6,))")
        print(f"  gripper_state shape : {first['observation']['gripper_state'].shape}  (기대: (2,))")
        print(f"  image shape         : {first['observation']['image'].shape}  (기대: (256, 256, 3))")
        print(f"  language            : {first['language_instruction'].numpy().decode()}")
        print(f"  is_first[0]         : {first['is_first'].numpy()}  (기대: True)")
        print(f"  is_last[-1]         : {last['is_last'].numpy()}  (기대: True)")
        print(f"  EEF_state[0]        : {first['observation']['EEF_state'].numpy()}")
        print(f"  gripper_state[0]    : {first['observation']['gripper_state'].numpy()}")
        print(f"  action[0]           : {first['action'].numpy()}")

    print("=" * 60)
    print("검증 완료 — 문제 없으면 delete_Me_RLDS_bg_masked_libero_object 폴더 삭제 후 본 변환 실행")


if __name__ == "__main__":
    print(f"Input  : {IN_DATASET_DIR}")
    print(f"Output : {OUT_DATA_DIR}")
    print()

    builder = DeleteMeBgMaskedLiberoObject(data_dir=str(OUT_DATA_DIR))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(verify_ssl=False)
    )

    verify(OUT_DATA_DIR)
