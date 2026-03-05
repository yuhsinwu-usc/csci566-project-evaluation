from __future__ import annotations
from typing import Any, Dict, List, Optional
import random

from .metrics import EvalResult


def evaluate_backend(
    checkpoint: str,
    task: str,
    num_episodes: int,
    seed: int,
    device: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    This function is the ONLY place you will later connect to StarVLA/LIBERO.

    Return a dict that at least contains:
      - successes: List[bool] length num_episodes
      - rewards: Optional[List[float]]
      - lengths: Optional[List[int]]
    """
    random.seed(seed)

    if dry_run:
        # Fake rollout results to validate pipeline without other groups
        successes = [random.random() < 0.6 for _ in range(num_episodes)]
        rewards = [random.random() for _ in range(num_episodes)]
        lengths = [random.randint(50, 200) for _ in range(num_episodes)]
        return {"successes": successes, "rewards": rewards, "lengths": lengths}

    # TODO: Replace this with actual StarVLA + LIBERO evaluation call
    # Example pseudo-code:
    # env = libero.make_env(task, ...)
    # policy = starvla.load_policy(checkpoint, device=device)
    # successes, rewards, lengths = run_rollouts(env, policy, num_episodes, seed)
    # return {"successes": successes, "rewards": rewards, "lengths": lengths}

    raise NotImplementedError(
        "Backend evaluation not connected yet. Use dry_run=True for now."
    )


def run_evaluation(
    checkpoint: str,
    task: str,
    num_episodes: int = 50,
    seed: int = 0,
    device: str = "cuda",
    suite: str = "LIBERO",
    model: str = "StarVLA",
    notes: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> EvalResult:
    """
    Clean API for evaluation. Returns a validated EvalResult.
    """
    raw = evaluate_backend(
        checkpoint=checkpoint,
        task=task,
        num_episodes=num_episodes,
        seed=seed,
        device=device,
        dry_run=dry_run,
    )

    successes: List[bool] = raw.get("successes", [])
    if len(successes) != num_episodes:
        raise ValueError(f"Expected {num_episodes} successes, got {len(successes)}")

    success_rate = sum(1 for s in successes if s) / float(num_episodes)

    rewards = raw.get("rewards")
    lengths = raw.get("lengths")

    avg_reward = None
    if isinstance(rewards, list) and len(rewards) == num_episodes:
        avg_reward = sum(float(r) for r in rewards) / float(num_episodes)

    avg_length = None
    if isinstance(lengths, list) and len(lengths) == num_episodes:
        avg_length = sum(float(l) for l in lengths) / float(num_episodes)

    result = EvalResult(
        task=task,
        checkpoint=checkpoint,
        suite=suite,
        model=model,
        success_rate=float(success_rate),
        num_episodes=int(num_episodes),
        avg_reward=avg_reward,
        avg_length=avg_length,
        seed=seed,
        device=device,
        notes=notes,
        extra=extra,
    )
    result.validate()
    return result