"""
merge_lora_ftgdino_bg_masked.py

ftgdino_bg_masked_libero_object 로 fine-tune 된 VLA-Adapter 체크포인트의
LoRA 어댑터를 base 가중치에 머지하고, 새 디렉토리에 저장한다.

원본 디렉토리는 보존되며, 결과 디렉토리에는 평가에 필요한 모든 보조 파일
(action_head, proprio_projector, dataset_statistics, processor/tokenizer)이
함께 복사된다.

실행:
    PYTHONPATH=/path/to/VLA_Adapter/models/VLA-Adapter \
    /path/to/VLA_Adapter/vla_adapter_env/bin/python \
        /path/to/VLA_Adapter/script/merge_lora_ftgdino_bg_masked.py
"""

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

sys.path.insert(0, "/path/to/VLA_Adapter/models/VLA-Adapter")

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)


SRC_DIR_DEFAULT = (
    "/path/to/VLA_Adapter/models/"
    "base_vla_adapter+ftgdino_bg_masked_libero_object"
    "+b128+lr-0.00016+lora-r32+dropout-0.0--image_aug"
)
DST_DIR_DEFAULT = SRC_DIR_DEFAULT + "+merged"

# merged_vla.save_pretrained() 가 직접 생성하는 파일은 제외하고,
# 그 외 모든 파일을 src → dst 로 복사한다.
SKIP_FILE_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
}
SKIP_DIR_NAMES = {"lora_adapter"}


@dataclass
class MergeConfig:
    src_dir: Union[str, Path] = SRC_DIR_DEFAULT
    dst_dir: Union[str, Path] = DST_DIR_DEFAULT
    device:  str = "cuda"            # "cuda" 또는 "cpu"
    overwrite: bool = False           # dst_dir 가 이미 있을 때 덮어쓸지


def register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def copy_auxiliary_files(src_dir: Path, dst_dir: Path) -> None:
    """
    src_dir 의 보조 파일(action_head, proprio_projector, dataset_statistics,
    processor / tokenizer 관련)을 dst_dir 로 복사. lora_adapter 디렉토리와
    merged save_pretrained 가 생성하는 파일은 건너뜀.
    """
    copied = 0
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir():
            if entry.name in SKIP_DIR_NAMES:
                continue
            shutil.copytree(entry, dst_dir / entry.name, dirs_exist_ok=True)
            copied += 1
            print(f"  [dir ]  {entry.name}")
        else:
            if entry.name in SKIP_FILE_NAMES:
                continue
            # config 백업 파일(config.json.back.*) 도 굳이 복사할 필요 없음
            if entry.name.startswith("config.json.back."):
                continue
            shutil.copy2(entry, dst_dir / entry.name)
            copied += 1
            print(f"  [file]  {entry.name}")
    print(f"보조 파일 {copied} 개 복사 완료")


@draccus.wrap()
def main(cfg: MergeConfig) -> None:
    src_dir = Path(cfg.src_dir).resolve()
    dst_dir = Path(cfg.dst_dir).resolve()
    lora_dir = src_dir / "lora_adapter"

    assert src_dir.is_dir(), f"src_dir 가 존재하지 않음: {src_dir}"
    assert lora_dir.is_dir(), f"lora_adapter 디렉토리가 없음: {lora_dir}"
    assert (src_dir / "model.safetensors").is_file(), \
        f"base 가중치(model.safetensors) 가 없음: {src_dir}"

    if dst_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"dst_dir 이 이미 존재함: {dst_dir}\n"
                "덮어쓰려면 --overwrite True 로 실행"
            )
        print(f"기존 dst_dir 삭제: {dst_dir}")
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    print(f"SRC: {src_dir}")
    print(f"DST: {dst_dir}")
    print(f"LoRA adapter: {lora_dir}")

    register_openvla()

    print("\n[1/3] base VLA 로드 중 (float16)...")
    t0 = time.time()
    base_vla = AutoModelForVision2Seq.from_pretrained(
        str(src_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    print(f"  완료 ({time.time() - t0:.1f}s)")

    print("\n[2/3] LoRA 어댑터 attach 후 merge_and_unload...")
    t0 = time.time()
    peft_model = PeftModel.from_pretrained(base_vla, str(lora_dir))
    if cfg.device == "cuda" and torch.cuda.is_available():
        peft_model = peft_model.to("cuda")
    merged_vla = peft_model.merge_and_unload()
    print(f"  완료 ({time.time() - t0:.1f}s)")

    print("\n[3/3] 머지 결과 저장 + 보조 파일 복사...")
    t0 = time.time()
    merged_vla.save_pretrained(str(dst_dir))
    copy_auxiliary_files(src_dir, dst_dir)
    print(f"  완료 ({time.time() - t0:.1f}s)")

    print(f"\n저장 위치: {dst_dir}")
    print("평가 시 --pretrained_checkpoint 인자에 위 경로를 넘기면 된다.")


if __name__ == "__main__":
    main()
