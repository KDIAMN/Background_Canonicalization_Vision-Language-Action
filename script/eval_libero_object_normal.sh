#!/bin/bash
# 원본 LIBERO-Object suite 평가 — normal_libero_object 모델
# - 학습/추론 일치 설정:
#     num_images_in_input = 1   (wrist 카메라 미사용)
#     use_proprio        = False
#     use_pro_version    = True (action_head 가 MLPResNetBlock_Pro 구조)
#     use_minivlm        = True (Qwen2.5 0.5B-extra)
# - task = 10, episode = 10  →  총 100 에피소드
#
# 결과 저장 위치 (기본):
#   /path/to/VLA_Adapter/eval/libero_object_eval/Normal_libero_object/
#       ├── videos/   ← 에피소드 .mp4
#       └── logs/     ← EVAL-libero_object-{DATE_TIME}--normal_libero_object.txt
#
# 실행:
#   bash /path/to/VLA_Adapter/script/eval_libero_object_normal.sh
#   # GPU 변경 시:  GPU=N bash ...sh

set -e

CKPT_DIR="/path/to/VLA_Adapter/checkpoints/base_vla_adapter+normal_libero_object+b128+lr-0.00016+lora-r32+dropout-0.0--image_aug"
RESULT_ROOT="/path/to/VLA_Adapter/eval/libero_object_eval/Normal_libero_object"
GPU="${GPU:-0}"

cd /path/to/VLA_Adapter/eval/libero_pro_env_object

CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval.py \
    --pretrained_checkpoint  "${CKPT_DIR}" \
    --task_suite_name        libero_object \
    --num_trials_per_task    10 \
    --num_images_in_input    1 \
    --use_proprio            False \
    --use_pro_version        True \
    --unnorm_key             normal_libero_object \
    --result_root            "${RESULT_ROOT}" \
    --seed                   7 \
    --run_id_note            normal_libero_object
