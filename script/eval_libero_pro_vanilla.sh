#!/bin/bash
# LIBERO-PRO 환경(libero_object_temp) 평가 스크립트
# vanilla VLA-Adapter (background canonicalization 없음, 원본 이미지 그대로 입력)
#
# 결과 저장 위치 (기본):
#   /path/to/VLA_Adapter/eval/libero_pro_env_object/Vanilla_libero_pro_object/
#       ├── videos/   ← 에피소드 .mp4 (총 100개: 10 task × 10 episode)
#       └── logs/     ← EVAL-...-{DATE_TIME}.txt
#
# 실행:
#   bash /path/to/VLA_Adapter/script/eval_libero_pro_vanilla.sh

set -e

CKPT_DIR="/path/to/VLA_Adapter/models/vanilla_vla_adapter"
RESULT_ROOT="/path/to/VLA_Adapter/eval/libero_pro_env_object/Vanilla_libero_pro_object"
GPU="${GPU:-2}"   # 기본 GPU 2번. 변경 시: GPU=3 bash ...sh

cd /path/to/VLA_Adapter/eval/libero_pro_env_object

CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval.py \
    --pretrained_checkpoint "${CKPT_DIR}" \
    --task_suite_name       libero_object_temp \
    --num_trials_per_task   10 \
    --result_root           "${RESULT_ROOT}" \
    --seed                  7 \
    --run_id_note           vanilla
