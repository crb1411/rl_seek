"""Command-line launcher for PPO training experiments."""

from __future__ import annotations

import argparse
from datetime import datetime

from config import Advantage_Policy, TrainingConfig
from main import ppo_train


POLICY_TARGETS = {
    policy.name.lower().replace("_", "-"): policy
    for policy in Advantage_Policy
}


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    policy_clip = (
        f"policy-clip-{_number_label(args.clip_ratio)}"
        if args.policy_clip
        else "no-policy-clip"
    )
    value_clip = (
        f"value-clip-{_number_label(args.value_clip_range)}"
        if args.value_clip
        else "no-value-clip"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}__{policy_clip}__{value_clip}__seed-{args.seed}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an actor-critic policy. With --policy-target standard-ppo, "
            "--no-policy-clip uses the unclipped ratio*advantage objective."
        )
    )
    parser.add_argument("--env-name", default="CartPole-v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=8192,
        help="Total transitions across all rollout environments.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    parser.add_argument(
        "--policy-target",
        choices=POLICY_TARGETS,
        default="standard-ppo",
    )
    parser.add_argument(
        "--adv-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--policy-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the PPO policy-ratio clip.",
    )
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--value-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable critic value clipping.",
    )
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default="manual")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--resume-path")
    parser.add_argument("--video-episodes", type=int, default=5)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-entity", default="crb_1411")
    parser.add_argument("--wandb-project", default="seek_rl")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--render-test",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    run_name = args.run_name or _default_run_name(args)
    wandb_group = args.wandb_group or args.experiment_name
    return TrainingConfig(
        env_name=args.env_name,
        steps_per_epoch=args.steps_per_epoch,
        num_envs=args.num_envs,
        epochs=args.epochs,
        gamma=args.gamma,
        lam=args.lam,
        pi_lr=args.learning_rate,
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        render_test=args.render_test,
        policy_target=POLICY_TARGETS[args.policy_target],
        use_adv_normalizer=args.adv_normalization,
        use_clip=args.policy_clip,
        clip_ratio=args.clip_ratio,
        use_value_clip=args.value_clip,
        value_clip_range=args.value_clip_range,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        checkpoint_freq=args.checkpoint_freq,
        resume_path=args.resume_path,
        video_episodes=args.video_episodes,
        use_wandb=args.wandb,
        run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=wandb_group,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ppo_train(config_from_args(args))


if __name__ == "__main__":
    main()
