"""
run_libero_pro_eval_bg_masked.py
LIBERO-PRO evaluation for the bg_masked fine-tuned VLA-Adapter.

Each observation frame has its background canonicalized (grayscale + brightness
normalization + Gaussian blur) before being fed to the model, matching the
distribution seen during fine-tuning on bg_masked_libero_object.

Run:
    cd /path/to/VLA_Adapter/eval/libero_pro_env_object
    MUJOCO_GL=egl \\
    PYTHONPATH=/path/to/VLA_Adapter/datasets/Libero-pro:/path/to/VLA_Adapter/models/VLA-Adapter:/path/to/VLA_Adapter/eval \\
    /path/to/VLA_Adapter/vla_adapter_env/bin/python run_libero_pro_eval_bg_masked.py \\
        --pretrained_checkpoint /path/to/VLA_Adapter/runs/bg_masked_libero_object/<checkpoint_dir> \\
        --task_suite_name libero_object_temp \\
        --num_trials_per_task 10
"""

import logging
import os
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
sys.path.insert(0, "/path/to/VLA_Adapter/eval")

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
from bg_mask_processor import BgMaskProcessor


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

# bg_masked 모델의 unnorm_key 는 학습에 사용한 dataset_name 과 일치해야 함
UNNORM_KEY_MAP = {
    "libero_object_temp": "ftgdino_bg_masked_libero_object",
    "libero_object":      "ftgdino_bg_masked_libero_object",
}

DEFAULT_RESULT_ROOT = "/path/to/VLA_Adapter/eval/libero_pro_env_object/ftgdino_bg_mask_result"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    model_family:         str   = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression:    bool  = True
    use_pro_version:      bool  = True   # action_head 의 MLPResNetBlock_Pro 사용 (학습 시 True)
    use_minivlm:          bool  = True
    num_diffusion_steps:  int   = 50
    use_film:             bool  = False
    num_images_in_input:  int   = 1
    use_proprio:          bool  = False   # fine-tuning 때 False
    center_crop:          bool  = True
    num_open_loop_steps:  int   = 8
    unnorm_key:           Union[str, Path] = ""
    load_in_8bit:         bool  = False
    load_in_4bit:         bool  = False

    task_suite_name:      str   = TaskSuite.LIBERO_OBJECT_TEMP
    num_steps_wait:       int   = 10
    num_trials_per_task:  int   = 10
    env_img_res:          int   = 256

    run_id_note:          Optional[str] = None
    result_root:          str   = DEFAULT_RESULT_ROOT   # videos/ 와 logs/ 가 이 안에 생성됨
    seed:                 int   = 7
    save_version:         str   = "libero_pro_env_object_bg_masked_eval"
    # fmt: on


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def save_video(replay_images, task_description, episode_idx, success, video_dir, log_file=None):
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
        assert unnorm_key in model.norm_stats, f"unnorm_key '{unnorm_key}' not found in norm_stats!"
        cfg.unnorm_key = unnorm_key
    return model, action_head, proprio_projector, processor


def run_episode(cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector, initial_state,
                bg_proc, log_file):
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

            img       = get_libero_image(obs)       # H×W×3 uint8 RGB
            wrist_img = get_libero_wrist_image(obs)

            replay_images.append(img)

            if len(action_queue) == 0:
                # 모델 호출 시에만 마스킹 (학습 분포 일치 + 8배 빠름)
                img_masked = bg_proc.process(img)

                observation = {
                    "full_image":  resize_image_for_policy(img_masked, resize_size),
                    "wrist_image": resize_image_for_policy(wrist_img, resize_size),
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )),
                }
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


def run_task(cfg, task_suite, task_id, model, resize_size,
             processor, action_head, proprio_projector,
             bg_proc, video_dir, total_episodes, total_successes, log_file):
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    task_episodes, task_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=task_description[:50]):
        initial_state = initial_states[episode_idx]
        log_message(f"\nTask: {task_description} | Episode {episode_idx + 1}", log_file)

        success, replay_images = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector, initial_state,
            bg_proc, log_file,
        )

        task_episodes    += 1
        total_episodes   += 1
        if success:
            task_successes  += 1
            total_successes += 1

        save_video(replay_images, task_description, episode_idx + 1, success, video_dir, log_file)
        log_message(
            f"Result: {'SUCCESS' if success else 'FAIL'} | "
            f"Task SR: {task_successes}/{task_episodes} | "
            f"Total SR: {total_successes}/{total_episodes} "
            f"({total_successes / total_episodes * 100:.1f}%)",
            log_file,
        )

    env.close()
    del env

    task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
    log_message(
        f"\n[Task {task_id}] {task_description}  →  "
        f"{task_successes}/{task_episodes} ({task_sr * 100:.1f}%)",
        log_file,
    )

    return total_episodes, total_successes, (task_id, task_description, task_episodes, task_successes)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert cfg.task_suite_name in [s.value for s in TaskSuite], \
        f"Invalid task suite: {cfg.task_suite_name}"

    set_seed_everywhere(cfg.seed)

    log_message("BgMaskProcessor 초기화 중 (GroundingDINO + SAM2)...")
    bg_proc = BgMaskProcessor(device="cuda")
    log_message("BgMaskProcessor 초기화 완료")

    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    video_dir = os.path.join(cfg.result_root, "videos")
    log_dir   = os.path.join(cfg.result_root, "logs")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    run_id = f"EVAL-bg_masked-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    log_path = os.path.join(log_dir, f"{run_id}.txt")
    log_file = open(log_path, "w")
    log_message(f"Run ID: {run_id}", log_file)
    log_message(f"Result root: {cfg.result_root}", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    log_message(f"Task suite: {cfg.task_suite_name} | Tasks: {task_suite.n_tasks}", log_file)

    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        total_episodes, total_successes, task_summary = run_task(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_projector,
            bg_proc, video_dir, total_episodes, total_successes, log_file,
        )
        task_results.append(task_summary)

    log_message("\n=== Per-Task Results ===", log_file)
    for tid, tdesc, tep, tsuc in task_results:
        sr = tsuc / tep if tep > 0 else 0.0
        log_message(
            f"Task {tid:>2d} | {tdesc[:50]:<50s} |  {tsuc:>3d}/{tep:<3d} ({sr * 100:5.1f}%)",
            log_file,
        )

    final_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    log_message("\n=== Final Results ===", log_file)
    log_message(f"Total episodes : {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Success rate   : {final_sr:.4f} ({final_sr * 100:.1f}%)", log_file)
    log_file.close()
    return final_sr


if __name__ == "__main__":
    eval_libero()
