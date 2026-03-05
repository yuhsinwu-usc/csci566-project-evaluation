from __future__ import annotations
import argparse
from typing import List

from .api import run_evaluation
from .logger import save_json, save_csv


def main():
    parser = argparse.ArgumentParser(description="StarVLA + LIBERO Evaluation Runner")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--format", type=str, choices=["json", "csv", "both"], default="both")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Generate fake results without backend")

    args = parser.parse_args()

    result = run_evaluation(
        checkpoint=args.checkpoint,
        task=args.task,
        num_episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        notes=args.notes,
        dry_run=args.dry_run,
    )

    records: List[dict] = [result.to_dict()]

    saved = []
    if args.format in ("json", "both"):
        saved.append(save_json(records, out_dir=args.out_dir))
    if args.format in ("csv", "both"):
        saved.append(save_csv(records, out_dir=args.out_dir))

    print("Evaluation finished.")
    print(f"success_rate={result.success_rate:.4f}, episodes={result.num_episodes}")
    for p in saved:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()