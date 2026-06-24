"""
run_libero_pro_eval_multigpu.py  —  LIBERO-PRO Environment Perturbation (4 GPU 병렬)

10개 task를 4 GPU에 분배하여 병렬 평가.

Run (launcher):
    cd /path/to/VLA_Adapter/eval/libero_pro_env_object
    MUJOCO_GL=egl \
    PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter \
    /path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval_multigpu.py \
        --pretrained_checkpoint /path/to/VLA_Adapter/models/vanilla_vla_adapter \
        --num_gpus 4
"""

import json
import logging
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import imageio
import numpy as np
import tqdm

sys.path.insert(0, "/path/to/VLA_Adapter/models/VLA-Adapter")
sys.path.insert(0, "/path/to/VLA_Adapter/datasets/Libero-pro")

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class TaskSuite(str, Enum):
    LIBERO_SPATIAL    = "libero_spatial"
    LIBERO_OBJECT     = "libero_object"
    LIBERO_GOAL       = "libero_goal"
    LIBERO_10         = "libero_10"
    LIBERO_90         = "libero_90"
    LIBERO_OBJECT_TEMP = "libero_object_temp"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL:     220,
    TaskSuite.LIBERO_OBJECT:      280,
    TaskSuite.LIBERO_GOAL:        300,
    TaskSuite.LIBERO_10:          520,
    TaskSuite.LIBERO_90:          400,
    TaskSuite.LIBERO_OBJECT_TEMP: 280,
}

UNNORM_KEY_MAP = {
    "libero_object_temp": "libero_object",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = TaskSuite.LIBERO_OBJECT_TEMP
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    local_log_dir: str = "./logs"
    seed: int = 7
    save_version: str = "libero_pro_env_object_eval"

    # Multi-GPU 제어용 (worker 모드에서 사용)
    num_gpus: int = 4        # 총 GPU 수
    gpu_rank: int = -1       # -1이면 launcher 모드, 0~3이면 worker 모드
    # fmt: on


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_video_dir(gpu_rank: int) -> str:
    return f"/path/to/VLA_Adapter/eval/libero_pro_env_object/result_videos_multiGPU/gpu{gpu_rank}"


def save_video(replay_images, task_description, episode_idx, success, gpu_rank, log_file=None):
    video_dir = get_video_dir(gpu_rank)
    os.makedirs(video_dir, exist_ok=True)
    task_name = task_description.lower().replace(" ", "_").replace(".", "")[:60]
    result = "success" if success else "fail"
    path = os.path.join(video_dir, f"{task_name}_ep{episode_idx}_{result}.mp4")
    imageio.mimwrite(path, replay_images, fps=30)
    log_message(f"Video saved: {path}", log_file)


def initialize_model(cfg):
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        base_key = UNNORM_KEY_MAP.get(cfg.task_suite_name, cfg.task_suite_name)
        unnorm_key = base_key
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"unnorm_key '{unnorm_key}' not found!"
        cfg.unnorm_key = unnorm_key
    return model, action_head, proprio_projector, processor


def run_episode(cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector, initial_state, log_file):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    t = 0
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            replay_images.append(img)

            observation = {
                "full_image": resize_image_for_policy(img, resize_size),
                "wrist_image": resize_image_for_policy(wrist_img, resize_size),
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }

            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector, noisy_action_projector=None,
                    use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
    return success, replay_images


def run_worker(cfg):
    """Worker 모드: 배정된 task 범위만 평가."""
    gpu_rank = cfg.gpu_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)

    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"EVAL-gpu{gpu_rank}-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    log_path = os.path.join(cfg.local_log_dir, f"{run_id}.txt")
    log_file = open(log_path, "w")
    log_message(f"[GPU {gpu_rank}] Run ID: {run_id}", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks

    # GPU별 task 분배
    task_ids = list(range(n_tasks))
    assigned = [task_ids[i] for i in range(n_tasks) if i % cfg.num_gpus == gpu_rank]
    log_message(f"[GPU {gpu_rank}] 담당 tasks: {assigned}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(assigned, desc=f"GPU{gpu_rank}"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=task_description[:50]):
            initial_state = initial_states[episode_idx]
            log_message(f"\nTask: {task_description} | Episode {episode_idx + 1}", log_file)

            success, replay_images = run_episode(
                cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector, initial_state, log_file,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            save_video(replay_images, task_description, episode_idx + 1, success, gpu_rank, log_file)
            log_message(
                f"Result: {'SUCCESS' if success else 'FAIL'} | "
                f"Task SR: {task_successes}/{task_episodes} | "
                f"GPU{gpu_rank} Total SR: {total_successes}/{total_episodes}",
                log_file,
            )

        env.close()
        del env

    log_message(f"\n[GPU {gpu_rank}] 완료: {total_successes}/{total_episodes}", log_file)
    log_file.close()

    # 결과를 JSON으로 저장 (launcher가 집계)
    result_path = os.path.join(cfg.local_log_dir, f"gpu{gpu_rank}_result.json")
    with open(result_path, "w") as f:
        json.dump({"successes": total_successes, "episodes": total_episodes}, f)


def run_launcher(cfg):
    """Launcher 모드: num_gpus개 subprocess 실행 후 결과 집계."""
    script_path = os.path.abspath(__file__)
    env_base = os.environ.copy()
    env_base["MUJOCO_GL"] = "egl"
    env_base["PYTHONPATH"] = (
        "/path/to/VLA_Adapter/datasets/Libero-pro:"
        "/path/to/VLA_Adapter/models/VLA-Adapter"
    )

    python = "/path/to/VLA_Adapter/vla_adapter_env/bin/python"
    procs = []

    for gpu_rank in range(cfg.num_gpus):
        env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu_rank)}
        cmd = [
            python, script_path,
            "--pretrained_checkpoint", str(cfg.pretrained_checkpoint),
            "--gpu_rank", str(gpu_rank),
            "--num_gpus", str(cfg.num_gpus),
            "--num_trials_per_task", str(cfg.num_trials_per_task),
            "--task_suite_name", str(cfg.task_suite_name),
            "--num_images_in_input", str(cfg.num_images_in_input),
            "--local_log_dir", cfg.local_log_dir,
            "--seed", str(cfg.seed),
        ]
        if cfg.run_id_note:
            cmd += ["--run_id_note", cfg.run_id_note]

        print(f"[Launcher] GPU {gpu_rank} 시작: {' '.join(cmd[-6:])}")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    print(f"[Launcher] {cfg.num_gpus}개 프로세스 실행 중... 완료 대기")
    for p in procs:
        p.wait()

    # 결과 집계
    total_successes, total_episodes = 0, 0
    for gpu_rank in range(cfg.num_gpus):
        result_path = os.path.join(cfg.local_log_dir, f"gpu{gpu_rank}_result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            total_successes += r["successes"]
            total_episodes  += r["episodes"]
            print(f"  GPU {gpu_rank}: {r['successes']}/{r['episodes']}")
        else:
            print(f"  GPU {gpu_rank}: 결과 파일 없음 (오류 확인 필요)")

    final_sr = total_successes / total_episodes if total_episodes > 0 else 0
    print("\n=== Final Results ===")
    print(f"Total episodes : {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate   : {final_sr:.4f} ({final_sr * 100:.1f}%)")

    summary_path = os.path.join(cfg.local_log_dir, f"FINAL-{DATE_TIME}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total episodes : {total_episodes}\n")
        f.write(f"Total successes: {total_successes}\n")
        f.write(f"Success rate   : {final_sr:.4f} ({final_sr * 100:.1f}%)\n")
    print(f"최종 결과 저장: {summary_path}")


@draccus.wrap()
def main(cfg: GenerateConfig):
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"

    if cfg.gpu_rank >= 0:
        run_worker(cfg)
    else:
        run_launcher(cfg)


if __name__ == "__main__":
    main()
