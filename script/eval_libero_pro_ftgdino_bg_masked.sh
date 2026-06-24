#!/bin/bash
# LIBERO-PRO 환경(libero_object_temp) 평가 스크립트
# bg_masked fine-tune 된 VLA-Adapter (LoRA 머지 완료) 를
# 추론 시 background canonicalization 을 적용해 평가한다.
#
# 결과 저장 위치 (기본):
#   /path/to/VLA_Adapter/eval/libero_pro_env_object/ftgdino_bg_mask_result_100/
#       ├── videos/   ← 에피소드 .mp4 (총 100개: 10 task × 10 episode)
#       └── logs/     ← EVAL-bg_masked-...-{DATE_TIME}.txt
#
# 실행:
#   bash /path/to/VLA_Adapter/script/eval_libero_pro_ftgdino_bg_masked.sh

set -e

CKPT_DIR="/path/to/VLA_Adapter/checkpoints/base_vla_adapter+ftgdino_bg_masked_libero_object+b128+lr-0.00016+lora-r32+dropout-0.0--image_aug"
RESULT_ROOT="/path/to/VLA_Adapter/eval/libero_pro_env_object/ftgdino_bg_mask_result_100"
GPU="${GPU:-1}"   # 기본 GPU 1번. 변경 시: GPU=3 bash ...sh

cd /path/to/VLA_Adapter/eval/libero_pro_env_object

CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter:/path/to/VLA_Adapter/eval \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval_bg_masked.py \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --task_suite_name       libero_object_temp \
    --num_trials_per_task   10 \
    --result_root           "${RESULT_ROOT}" \
    --seed                  7 \
    --run_id_note           ftgdino_bg_masked
