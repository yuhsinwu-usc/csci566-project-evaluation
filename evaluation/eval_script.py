import os
import json
import argparse
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

TASK_CONFIGS = {
    "pot_on_stove": {
        "instruction": "turn on the stove and put the moka pot on it",
    },
    "mugs_on_plates": {
        "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    },
    "items_into_basket": {
        "instruction": "put both the alphabet soup and the cream cheese box in the basket",
    },
}

GENERALIST_TASKS = [
    # TODO: replace with your actual generalist eval task list
    "generalist_task_1",
    "generalist_task_2",
    "generalist_task_3",
    "generalist_task_4",
    "generalist_task_5",
]

DEFAULT_OOD_SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class EvalStats:
    mean: float
    std: float
    num_episodes: int
    scores: List[float]


@dataclass
class EvalOutput:
    checkpoint_name: str
    checkpoint_path: str
    target_task: str
    results: Dict[str, Dict[str, Any]]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_stats(binary_scores: List[int]) -> EvalStats:
    arr = np.array(binary_scores, dtype=np.float32)
    return EvalStats(
        mean=float(arr.mean()) if len(arr) > 0 else 0.0,
        std=float(arr.std()) if len(arr) > 0 else 0.0,
        num_episodes=len(binary_scores),
        scores=[float(x) for x in arr.tolist()],
    )


# ============================================================
# Model Adapter
# ============================================================

class StarVLAEvaluatorModel:
    """
    Adapter for your actual StarVLA / Qwen2.5-VL-FAST model.

    You only need to implement:
      1. __init__
      2. predict_action

    Keep the external interface unchanged.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device

        # TODO:
        # Replace this part with your actual model loading code.
        #
        # Example:
        # from your_project.models import load_starvla_model
        # self.model, self.processor = load_starvla_model(checkpoint_path, device=device)
        #
        # If your teammate already has inference code, plug it here.
        self.model = None
        self.processor = None

    def predict_action(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        """
        Input:
            obs: environment observation, usually image/state/etc.
            instruction: language instruction string

        Output:
            action: numpy array, shape depends on your env
        """

        # TODO:
        # Replace this mock with actual model inference.
        #
        # Typical flow:
        #   inputs = preprocess(obs, instruction, self.processor)
        #   pred = self.model(...)
        #   action = postprocess(pred)
        #   return action
        #
        # For now, return dummy zeros to keep the script structure valid.
        return np.zeros(7, dtype=np.float32)


# ============================================================
# Environment Adapter
# ============================================================

class EvalEnv:
    """
    Adapter for your environment.
    You only need to implement:
      1. __init__
      2. reset
      3. step

    Keep this interface stable so the evaluation loop stays the same.
    """

    def __init__(self, task_name: str, eval_mode: str, seed: Optional[int] = None):
        self.task_name = task_name
        self.eval_mode = eval_mode
        self.seed = seed if seed is not None else 0

        # TODO:
        # Replace with your actual simulator / LIBERO env creation.
        #
        # Example:
        # self.env = make_libero_env(task_name, mode=eval_mode, seed=self.seed)
        #
        # If OOD needs extra randomization, configure it here.

        self.max_steps = 100
        self.current_step = 0

    def reset(self) -> Dict[str, Any]:
        self.current_step = 0

        # TODO:
        # Replace with actual env.reset()
        # return observation dict
        obs = {
            "image": None,
            "wrist_image": None,
            "state": np.zeros(7, dtype=np.float32),
        }
        return obs

    def step(self, action: np.ndarray):
        self.current_step += 1

        # TODO:
        # Replace with actual env.step(action)
        #
        # Must return:
        #   obs, reward, done, info
        #
        # info should ideally include:
        #   info["success"] -> bool
        done = self.current_step >= self.max_steps

        # Mock success behavior just for structure
        success = bool(np.random.rand() > 0.8) if done else False

        obs = {
            "image": None,
            "wrist_image": None,
            "state": np.zeros(7, dtype=np.float32),
        }
        reward = 1.0 if success else 0.0
        info = {"success": success}
        return obs, reward, done, info


# ============================================================
# Rollout
# ============================================================

def run_one_episode(
    model: StarVLAEvaluatorModel,
    env: EvalEnv,
    instruction: str,
    verbose: bool = False,
) -> int:
    obs = env.reset()
    done = False
    info = {"success": False}

    while not done:
        action = model.predict_action(obs, instruction)
        obs, reward, done, info = env.step(action)

    success = int(bool(info.get("success", False)))

    if verbose:
        print(f"[Episode Done] success={success}")

    return success


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_id(
    model: StarVLAEvaluatorModel,
    task_name: str,
    instruction: str,
    num_episodes: int,
    verbose: bool = False,
) -> EvalStats:
    env = EvalEnv(task_name=task_name, eval_mode="id")

    scores = []
    for ep in range(num_episodes):
        score = run_one_episode(model, env, instruction, verbose=verbose)
        scores.append(score)
        if verbose:
            print(f"[ID] episode={ep + 1}/{num_episodes}, success={score}")

    return compute_stats(scores)


def evaluate_ood(
    model: StarVLAEvaluatorModel,
    task_name: str,
    instruction: str,
    seeds: List[int],
    episodes_per_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    seed_means = []
    all_scores = {}

    for seed in seeds:
        set_seed(seed)
        env = EvalEnv(task_name=task_name, eval_mode="ood", seed=seed)

        scores = []
        for ep in range(episodes_per_seed):
            score = run_one_episode(model, env, instruction, verbose=verbose)
            scores.append(score)
            if verbose:
                print(f"[OOD] seed={seed}, episode={ep + 1}/{episodes_per_seed}, success={score}")

        stats = compute_stats(scores)
        seed_means.append(stats.mean)
        all_scores[f"seed_{seed}"] = asdict(stats)

    seed_means_arr = np.array(seed_means, dtype=np.float32)
    summary = {
        "mean": float(seed_means_arr.mean()) if len(seed_means_arr) > 0 else 0.0,
        "std": float(seed_means_arr.std()) if len(seed_means_arr) > 0 else 0.0,
        "num_seeds": len(seeds),
        "episodes_per_seed": episodes_per_seed,
        "seed_results": all_scores,
    }
    return summary


def evaluate_generalist(
    model: StarVLAEvaluatorModel,
    generalist_tasks: List[str],
    episodes_per_task: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    task_results = {}
    task_means = []

    for task_name in generalist_tasks:
        instruction = TASK_CONFIGS.get(task_name, {}).get("instruction", task_name)
        env = EvalEnv(task_name=task_name, eval_mode="generalist")

        scores = []
        for ep in range(episodes_per_task):
            score = run_one_episode(model, env, instruction, verbose=verbose)
            scores.append(score)
            if verbose:
                print(
                    f"[GENERALIST] task={task_name}, "
                    f"episode={ep + 1}/{episodes_per_task}, success={score}"
                )

        stats = compute_stats(scores)
        task_results[task_name] = asdict(stats)
        task_means.append(stats.mean)

    task_means_arr = np.array(task_means, dtype=np.float32)
    summary = {
        "mean": float(task_means_arr.mean()) if len(task_means_arr) > 0 else 0.0,
        "std": float(task_means_arr.std()) if len(task_means_arr) > 0 else 0.0,
        "num_tasks": len(generalist_tasks),
        "episodes_per_task": episodes_per_task,
        "task_results": task_results,
    }
    return summary


# ============================================================
# Plot
# ============================================================

def plot_single_checkpoint_results(results: Dict[str, Any], output_path: str) -> None:
    labels = []
    values = []

    if "id" in results:
        labels.append("ID")
        values.append(results["id"]["mean"])

    if "ood" in results:
        labels.append("OOD")
        values.append(results["ood"]["mean"])

    if "generalist" in results:
        labels.append("Generalist")
        values.append(results["generalist"]["mean"])

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("Evaluation Summary")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multi_checkpoint_results(all_results: List[EvalOutput], output_path: str) -> None:
    checkpoint_names = [r.checkpoint_name for r in all_results]

    id_vals = []
    ood_vals = []
    gen_vals = []

    for r in all_results:
        id_vals.append(r.results.get("id", {}).get("mean", 0.0))
        ood_vals.append(r.results.get("ood", {}).get("mean", 0.0))
        gen_vals.append(r.results.get("generalist", {}).get("mean", 0.0))

    x = np.arange(len(checkpoint_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, id_vals, width, label="ID")
    plt.bar(x, ood_vals, width, label="OOD")
    plt.bar(x + width, gen_vals, width, label="Generalist")

    plt.xticks(x, checkpoint_names, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("Checkpoint Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
