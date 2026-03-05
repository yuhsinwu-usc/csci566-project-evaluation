from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import math


@dataclass
class EvalResult:
    # identifiers
    task: str
    checkpoint: str
    suite: str  # e.g., "LIBERO"
    model: str  # e.g., "StarVLA"

    # core metrics
    success_rate: float          # [0,1]
    num_episodes: int
    avg_reward: Optional[float] = None
    avg_length: Optional[float] = None

    # runtime / env metadata
    seed: Optional[int] = None
    device: Optional[str] = None
    notes: Optional[str] = None

    # raw optional payload (keep small!)
    extra: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be a non-empty string")
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("checkpoint must be a non-empty string")
        if not isinstance(self.suite, str) or not self.suite:
            raise ValueError("suite must be a non-empty string")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")

        if not isinstance(self.num_episodes, int) or self.num_episodes <= 0:
            raise ValueError("num_episodes must be a positive int")

        if not (isinstance(self.success_rate, float) or isinstance(self.success_rate, int)):
            raise ValueError("success_rate must be a number")
        self.success_rate = float(self.success_rate)
        if math.isnan(self.success_rate) or self.success_rate < 0.0 or self.success_rate > 1.0:
            raise ValueError("success_rate must be in [0,1]")

        if self.avg_reward is not None:
            self.avg_reward = float(self.avg_reward)
        if self.avg_length is not None:
            self.avg_length = float(self.avg_length)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # make sure validate runs and types are normalized
        self.validate()
        return d