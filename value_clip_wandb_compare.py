"""Run one condition of the PPO value-clipping W&B comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import main as training
from config import Advantage_Policy, TrainingConfig
from utils import training_run_dir


VARIANTS = {
    "no_clip": (False, 0.0),
    "clip_3": (True, 3.0),
    "clip_1": (True, 1.0),
    "clip_0p5": (True, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--env-name", default="CartPole-v1")
    parser.add_argument("--device", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_value_clip, value_clip_range = VARIANTS[args.variant]
    condition = (
        "no-value-clip"
        if not use_value_clip
        else f"value-clip-{value_clip_range:g}".replace(".", "p")
    )
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
        clip_ratio=0.2,
        use_value_clip=use_value_clip,
        value_clip_range=value_clip_range,
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
