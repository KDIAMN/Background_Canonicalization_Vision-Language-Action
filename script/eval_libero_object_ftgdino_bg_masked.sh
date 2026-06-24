#!/bin/bash
# 원본 LIBERO suite (libero_object) 에서 ftgdino_bg_masked 모델 평가
# - libero 자체는 컬러 원본 이미지를 제공하지만, 추론 시 BgMaskProcessor 로
#   배경 흑백 + 밝기 조절 + 블러를 적용하여 학습 분포와 정합시킴
# - task = 10, episode = 10  →  총 100 에피소드
#
# 결과 저장 위치 (기본):
#   /path/to/VLA_Adapter/eval/libero_object_eval/ftgdino_bg_libero/
#       ├── videos/
#       └── logs/
#
# 실행:
#   bash /path/to/VLA_Adapter/script/eval_libero_object_ftgdino_bg_masked.sh

set -e

CKPT_DIR="/path/to/VLA_Adapter/checkpoints/base_vla_adapter+ftgdino_bg_masked_libero_object+b128+lr-0.00016+lora-r32+dropout-0.0--image_aug"
RESULT_ROOT="/path/to/VLA_Adapter/eval/libero_object_eval/ftgdino_bg_libero"
GPU="${GPU:-3}"   # 기본 GPU 3번. 변경 시: GPU=N bash ...sh

cd /path/to/VLA_Adapter/eval/libero_pro_env_object

CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter:/path/to/VLA_Adapter/eval \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval_bg_masked.py \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --task_suite_name       libero_object \
    --num_trials_per_task   10 \
    --result_root           "${RESULT_ROOT}" \
    --seed                  7 \
    --run_id_note           ftgdino_bg_libero
