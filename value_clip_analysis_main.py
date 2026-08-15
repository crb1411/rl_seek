"""Measure how PPO value clipping affects critic sample gradients."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from advantage_normalizer import AdvantageNormalizer
from config import Advantage_Policy
from inference import evaluate_policy
from main import RolloutBuffer
from models import ActorCritic
from utils import analysis_run_dir


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    epochs: int = 100
    steps_per_epoch: int = 2000
    train_iters: int = 10
    batch_size: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 3e-4
    policy_clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_episodes: int = 3
    seed: int = 20260814
    device: str = "cuda:1"
    output_root: str = "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-clip-ratio", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", default=ExperimentConfig.output_root)
    return parser.parse_args()


def configure_logger(path: Path, value_clip_ratio: float) -> logging.Logger:
    name = f"value-clip-analysis.{value_clip_ratio:.2f}"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")

    for handler in (
        logging.FileHandler(path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def collect_rollout(
    env,
    model: ActorCritic,
    buffer: RolloutBuffer,
    steps_per_epoch: int,
    device: torch.device,
) -> None:
    buffer.reset()
    steps_collected = 0
    while steps_collected < steps_per_epoch:
        obs, _ = env.reset()
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, old_log_prob, old_value = model.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            timeout_bootstrap_value = 0.0
            if truncated and not terminated:
                next_obs_tensor = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    timeout_bootstrap_value = model.get_value(
                        next_obs_tensor
                    ).item()

            buffer.store(
                obs,
                action,
                old_log_prob.item(),
                reward,
                terminated,
                truncated,
                old_value.item(),
                timeout_bootstrap_value=timeout_bootstrap_value,
            )
            obs = next_obs
            steps_collected += 1
            if terminated or truncated:
                break


def safe_percentage(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def train_one_epoch(
    *,
    epoch: int,
    config: ExperimentConfig,
    env,
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
) -> dict[str, float | int | str]:
    collect_rollout(env, model, buffer, config.steps_per_epoch, device)
    buffer.compute_returns_and_advantages(
        gamma=config.gamma,
        lam=config.lam,
        strategy=Advantage_Policy.STANDARD_PPO,
    )
    (
        obs,
        actions,
        old_log_probs,
        returns,
        advantages,
        old_values,
    ) = (tensor.to(device) for tensor in buffer.get())

    sample_count = obs.shape[0]
    shuffled_indices = np.arange(sample_count)
    ever_value_clipped = torch.zeros(sample_count, dtype=torch.bool, device=device)
    ever_gradient_blocked = torch.zeros(
        sample_count, dtype=torch.bool, device=device
    )

    visit_count = torch.zeros((), dtype=torch.long, device=device)
    value_clipped_visits = torch.zeros((), dtype=torch.long, device=device)
    gradient_blocked_visits = torch.zeros((), dtype=torch.long, device=device)
    clipped_but_active_visits = torch.zeros((), dtype=torch.long, device=device)
    abs_value_change_sum = torch.zeros((), device=device)
    selected_value_loss_sum = torch.zeros((), device=device)
    unclipped_value_loss_sum = torch.zeros((), device=device)
    clipped_value_loss_sum = torch.zeros((), device=device)
    max_abs_value_change = torch.zeros((), device=device)

    for _ in range(config.train_iters):
        np.random.shuffle(shuffled_indices)
        for start in range(0, sample_count, config.batch_size):
            mb_indices = torch.as_tensor(
                shuffled_indices[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            new_log_probs, entropy, new_values = model.evaluate_actions(
                obs[mb_indices], actions[mb_indices]
            )
            mb_advantages = advantages[mb_indices].detach()
            ratio = torch.exp(new_log_probs - old_log_probs[mb_indices])
            surrogate = ratio * mb_advantages
            clipped_surrogate = torch.clamp(
                ratio,
                1.0 - config.policy_clip_ratio,
                1.0 + config.policy_clip_ratio,
            ) * mb_advantages
            actor_loss = -torch.minimum(surrogate, clipped_surrogate).mean()

            mb_old_values = old_values[mb_indices]
            mb_returns = returns[mb_indices]
            value_change = new_values - mb_old_values
            clipped_values = mb_old_values + torch.clamp(
                value_change,
                -config.value_clip_ratio,
                config.value_clip_ratio,
            )
            value_loss_unclipped = (new_values - mb_returns) ** 2
            value_loss_clipped = (clipped_values - mb_returns) ** 2
            selected_value_losses = torch.maximum(
                value_loss_unclipped, value_loss_clipped
            )

            # Outside the clamp interval, V_clipped is locally constant with
            # respect to V_new. If maximum selects that larger loss, this
            # sample contributes zero gradient to the critic value output.
            value_clipped = value_change.abs() > config.value_clip_ratio
            gradient_blocked = value_clipped & (
                value_loss_clipped > value_loss_unclipped
            )
            clipped_but_active = value_clipped & ~gradient_blocked

            with torch.no_grad():
                batch_size = value_change.numel()
                visit_count += batch_size
                value_clipped_visits += value_clipped.sum()
                gradient_blocked_visits += gradient_blocked.sum()
                clipped_but_active_visits += clipped_but_active.sum()
                abs_value_change_sum += value_change.abs().sum()
                selected_value_loss_sum += selected_value_losses.sum()
                unclipped_value_loss_sum += value_loss_unclipped.sum()
                clipped_value_loss_sum += value_loss_clipped.sum()
                max_abs_value_change = torch.maximum(
                    max_abs_value_change, value_change.abs().max()
                )
                ever_value_clipped[mb_indices] |= value_clipped
                ever_gradient_blocked[mb_indices] |= gradient_blocked

            critic_loss = selected_value_losses.mean()
            total_loss = (
                actor_loss
                + config.value_loss_coef * critic_loss
                - config.entropy_coef * entropy.mean()
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

    eval_reward = evaluate_policy(
        model,
        env,
        episodes=config.eval_episodes,
        render=False,
        device=device,
    )

    return {
        "epoch": epoch + 1,
        "device": config.device,
        "value_clip_ratio": config.value_clip_ratio,
        "eval_reward": float(eval_reward),
        "rollout_samples": int(sample_count),
        "visit_count": int(visit_count.item()),
        "value_clipped_visits": int(value_clipped_visits.item()),
        "gradient_blocked_visits": int(gradient_blocked_visits.item()),
        "clipped_but_active_visits": int(clipped_but_active_visits.item()),
        "unique_value_clipped": int(ever_value_clipped.sum().item()),
        "unique_gradient_blocked": int(ever_gradient_blocked.sum().item()),
        "abs_value_change_sum": float(abs_value_change_sum.item()),
        "selected_value_loss_sum": float(selected_value_loss_sum.item()),
        "unclipped_value_loss_sum": float(unclipped_value_loss_sum.item()),
        "clipped_value_loss_sum": float(clipped_value_loss_sum.item()),
        "max_abs_value_change": float(max_abs_value_change.item()),
    }


def format_epoch(record: dict[str, float | int | str]) -> str:
    return (
        f"epoch={record['epoch']:03d} reward={record['eval_reward']:6.1f} | "
        f"value-clipped visits="
        f"{safe_percentage(record['value_clipped_visits'], record['visit_count']):6.2f}% "
        f"unique={safe_percentage(record['unique_value_clipped'], record['rollout_samples']):6.2f}% | "
        f"critic-gradient-blocked visits="
        f"{safe_percentage(record['gradient_blocked_visits'], record['visit_count']):6.2f}% "
        f"among-clipped="
        f"{safe_percentage(record['gradient_blocked_visits'], record['value_clipped_visits']):6.2f}% "
        f"unique={safe_percentage(record['unique_gradient_blocked'], record['rollout_samples']):6.2f}% | "
        f"|dV| mean/max={record['abs_value_change_sum'] / record['visit_count']:.4f}/"
        f"{record['max_abs_value_change']:.4f} | "
        f"selected-vloss={record['selected_value_loss_sum'] / record['visit_count']:.4f}"
    )


def aggregate_period(
    records: list[dict[str, float | int | str]],
    start_epoch: int,
    end_epoch: int,
) -> dict[str, float | int | str]:
    selected = [
        record
        for record in records
        if start_epoch <= int(record["epoch"]) <= end_epoch
    ]
    sum_fields = (
        "rollout_samples",
        "visit_count",
        "value_clipped_visits",
        "gradient_blocked_visits",
        "clipped_but_active_visits",
        "unique_value_clipped",
        "unique_gradient_blocked",
        "abs_value_change_sum",
        "selected_value_loss_sum",
        "unclipped_value_loss_sum",
        "clipped_value_loss_sum",
    )
    row: dict[str, float | int | str] = {
        field: sum(record[field] for record in selected) for field in sum_fields
    }
    row.update(
        {
            "period": f"{start_epoch:02d}-{end_epoch:03d}",
            "epochs": len(selected),
            "eval_reward_mean": float(
                np.mean([record["eval_reward"] for record in selected])
            ),
            "eval_reward_min": float(
                np.min([record["eval_reward"] for record in selected])
            ),
            "max_abs_value_change": max(
                record["max_abs_value_change"] for record in selected
            ),
        }
    )
    return row


def format_period(config: ExperimentConfig, row: dict[str, float | int | str]) -> str:
    return (
        f"eps_v={config.value_clip_ratio:.2f} {row['period']} {config.device} | "
        f"reward mean/min={row['eval_reward_mean']:7.2f}/{row['eval_reward_min']:6.1f} | "
        f"value-clipped visits="
        f"{safe_percentage(row['value_clipped_visits'], row['visit_count']):6.2f}% "
        f"unique={safe_percentage(row['unique_value_clipped'], row['rollout_samples']):6.2f}% | "
        f"gradient-blocked visits="
        f"{safe_percentage(row['gradient_blocked_visits'], row['visit_count']):6.2f}% "
        f"among-clipped="
        f"{safe_percentage(row['gradient_blocked_visits'], row['value_clipped_visits']):6.2f}% "
        f"unique={safe_percentage(row['unique_gradient_blocked'], row['rollout_samples']):6.2f}% | "
        f"clipped-but-active="
        f"{safe_percentage(row['clipped_but_active_visits'], row['visit_count']):6.2f}% | "
        f"|dV| mean/max={row['abs_value_change_sum'] / row['visit_count']:.4f}/"
        f"{row['max_abs_value_change']:.4f} | "
        f"selected-vloss={row['selected_value_loss_sum'] / row['visit_count']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.value_clip_ratio <= 0:
        raise ValueError("--value-clip-ratio must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    config = ExperimentConfig(
        value_clip_ratio=args.value_clip_ratio,
        device=args.device,
        output_root=args.output_root,
    )
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    label = f"eps_{config.value_clip_ratio:.2f}".replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analysis_run_dir(
        output_root=config.output_root,
        analysis_name="value-clip",
        run_name=f"{timestamp}_{label}_gpu{device.index}",
        create=True,
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger = configure_logger(output_dir / "training.log", config.value_clip_ratio)
    logger.info("START config=%s gpu=%s", asdict(config), torch.cuda.get_device_name(device))

    env = gym.make(config.env_name)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env.spec.max_episode_steps or config.steps_per_epoch
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = RolloutBuffer(
        config.steps_per_epoch + max_episode_steps,
        obs_dim,
        normalizer=AdvantageNormalizer(momentum=0),
    )

    records: list[dict[str, float | int | str]] = []
    jsonl_path = output_dir / "epochs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for epoch in range(config.epochs):
            record = train_one_epoch(
                epoch=epoch,
                config=config,
                env=env,
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                device=device,
            )
            records.append(record)
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            current_epoch = epoch + 1
            if (
                current_epoch == 1
                or current_epoch % 10 == 0
                or current_epoch in (33, 34, 66, 67)
            ):
                logger.info(format_epoch(record))
    env.close()

    rows = [
        aggregate_period(records, start, end)
        for start, end in ((1, 33), (34, 66), (67, 100))
    ]
    summary_lines = [
        "PPO critic value-clipping analysis",
        "value-clipped: abs(V_new - V_old) > eps_v",
        "gradient-blocked: value-clipped and clipped_loss > unclipped_loss",
        "",
        *(format_period(config, row) for row in rows),
    ]
    summary = "\n".join(summary_lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + summary)
    print(f"Detailed results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
