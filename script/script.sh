GPU 0 — libero_object (원본):


cd /path/to/VLA_Adapter/eval/libero_object_eval
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero/LIBERO:/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_eval.py \
    --pretrained_checkpoint /path/to/VLA_Adapter/models/vanilla_vla_adapter
GPU 1 — libero_object_temp (environment perturbation):


cd /path/to/VLA_Adapter/eval/libero_pro_env_object
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval.py \
    --pretrained_checkpoint /path/to/VLA_Adapter/models/vanilla_vla_adapter

[멀티 GPU libero-pro eval]
cd /path/to/VLA_Adapter/eval/libero_pro_env_object

MUJOCO_GL=egl \
PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter \
/path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval_multigpu.py \
    --pretrained_checkpoint /path/to/VLA_Adapter/models/vanilla_vla_adapter \
    --num_gpus 4

[rlds 변환 스크립트]

/path/to/VLA_Adapter/vla_adapter_env/bin/python3.10 \
    /path/to/VLA_Adapter/datasets/FT_dataset_augment/convert_lerobot_to_rlds.py


[lerobot libero_object dataset ]
"/path/to/smolVLA/lerobot/data/lerobot/libero_object_image"

[pro-6000 가중치 가져올 때 유의할 것]
 파인튜닝 후 저장된 .pt 파일 그대로 RTX 2080에서 로드하면 됩니다. inference 시 torch_dtype=torch.float16만 지정해주면
  되고, 체크포인트 파일 자체를 건드릴 필요는 없습니다.

[ GPU 0~3 utilization 파악]
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

[ftgdino_bd_mask libero-pro inference]
bash /path/to/VLA_Adapter/script/eval_libero_pro_ftgdino_bg_masked.sh
# 가장 최신 로그 실시간 보기
tail -f $(ls -t /path/to/VLA_Adapter/eval/libero_pro_env_object/ftgdino_bg_mask_result/logs/*.txt | head -1)

# GPU 사용률
watch -n 2 nvidia-smi

#백그라인드로 돌리기
nohup bash /path/to/VLA_Adapter/script/eval_libero_pro_ftgdino_bg_masked.sh > /tmp/eval.out 2>&1 &
echo "PID: $!"

# 진행 상황 확인
tail -f /tmp/eval.out

