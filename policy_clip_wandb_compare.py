"""Run one LunarLander PPO policy-clipping comparison condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import main as training
from config import Advantage_Policy, TrainingConfig
from utils import training_run_dir


def ratio_label(value: float) -> str:
    """Return a filesystem- and W&B-friendly decimal label."""
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-clip-ratio",
        type=float,
        choices=(0.1, 0.15, 0.3),
        required=True,
    )
    parser.add_argument("--value-clip-range", type=float, default=0.3)
    parser.add_argument("--env-name", default="LunarLander-v3")
    parser.add_argument("--device", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_label = ratio_label(args.policy_clip_ratio)
    value_label = ratio_label(args.value_clip_range)
    condition = f"policy-clip-{policy_label}__value-clip-{value_label}"
    run_name = f"{args.env_name}-{condition}"
    save_root = training_run_dir(
        output_root=args.output_root,
        env_name=args.env_name,
        experiment_name=args.group,
        run_name=condition,
        create=True,
    )

    config = TrainingConfig(
        env_name=args.env_name,
        steps_per_epoch=8192,
        num_envs=8,
        epochs=100,
        policy_target=Advantage_Policy.STANDARD_PPO,
        use_adv_normalizer=True,
        use_clip=True,
        clip_ratio=args.policy_clip_ratio,
        use_value_clip=True,
        value_clip_range=args.value_clip_range,
        train_iters=10,
        batch_size=256,
        use_wandb=True,
        wandb_entity="crb_1411",
        wandb_project="seek_rl",
        wandb_group=args.group,
        output_root=args.output_root,
        experiment_name=args.group,
        run_name=run_name,
        seed=20260814,
        device=args.device,
        save_root=save_root,
        checkpoint_dir=Path("checkpoints"),
    )
    training.ppo_train(config)


if __name__ == "__main__":
    main()
