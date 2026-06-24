#!/bin/bash
# Fine-tune VLA-Adapter on bg_masked_libero_object (배경 흑백+블러 처리, 메인 카메라만)
#
# 실행 전 확인사항:
#   1. convert_lerobot_to_rlds.py 를 먼저 실행해서 RLDS 데이터 변환 완료
#   2. wandb_entity, wandb_project 수정
#
# 실행:
#   bash /path/to/VLA_Adapter/script/finetune_bg_masked_libero_object.sh

cd /path/to/VLA_Adapter/models/VLA-Adapter

PYTHONPATH=/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 4 \
    vla-scripts/finetune.py \
    \
    --config_file_path  /path/to/VLA_Adapter/models/base_vla_adapter \
    --vlm_path          /path/to/VLA_Adapter/models/base_vla_adapter \
    --resum_vla_path    /path/to/VLA_Adapter/models/base_vla_adapter \
    \
    --data_root_dir     /path/to/VLA_Adapter/datasets/FT_dataset_augment/RLDS_bg_masked_libero_object \
    --dataset_name      bg_masked_libero_object \
    --run_root_dir      /path/to/VLA_Adapter/runs/bg_masked_libero_object \
    \
    --num_images_in_input   1 \
    --use_minivlm           True \
    --use_proprio           False \
    --use_l1_regression     True \
    --use_lora              True \
    --lora_rank             32 \
    --image_aug             True \
    \
    --batch_size            8 \
    --learning_rate         5e-4 \
    --max_steps             50000 \
    --save_freq             5000 \
    --shuffle_buffer_size   50000 \
    \
    --wandb_entity          dummy \
    --wandb_project         vla-adapter-bg-masked \
    --run_id_note           bg_masked_libero_object
# wandb는 finetune.py 내부에서 mode="offline"으로 고정 → 계정 불필요, 로컬에만 로그 저장
