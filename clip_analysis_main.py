"""Compare PPO clipping with two unclipped policy objectives.

This is an isolated experiment entry point. It imports the model and rollout
buffer from the project but does not change the behavior of ``main.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
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


class ActorObjective(str, Enum):
    PPO_CLIP = "ppo_clip"
    RATIO_UNCLIPPED = "ratio_unclipped"
    LOGPROB = "logprob"


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
    clip_ratio: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_episodes: int = 3
    seed: int = 20260814
    output_root: str = "clip_analysis_results"


def configure_logger(log_path: Path, mode: ActorObjective) -> logging.Logger:
    logger = logging.getLogger(f"clip-analysis.{mode.value}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
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


def policy_loss(
    mode: ActorObjective,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(new_log_probs - old_log_probs)

    if mode == ActorObjective.PPO_CLIP:
        surrogate = ratio * advantages
        clipped_surrogate = torch.clamp(
            ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
        ) * advantages
        loss = -torch.minimum(surrogate, clipped_surrogate).mean()
    elif mode == ActorObjective.RATIO_UNCLIPPED:
        loss = -(ratio * advantages).mean()
    elif mode == ActorObjective.LOGPROB:
        loss = -(new_log_probs * advantages).mean()
    else:
        raise ValueError(f"Unknown actor objective: {mode}")

    return loss, ratio


def train_one_epoch(
    *,
    epoch: int,
    mode: ActorObjective,
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
    indices = np.arange(sample_count)
    positive_unique = (advantages > 0).cpu().numpy()
    ever_positive_upper_clipped = np.zeros(sample_count, dtype=bool)

    visit_count = 0
    positive_visits = 0
    negative_visits = 0
    positive_upper_clipped_visits = 0
    negative_lower_clipped_visits = 0
    old_prob_clipped_sum = 0.0
    old_prob_clipped_count = 0
    old_prob_positive_kept_sum = 0.0
    old_prob_positive_kept_count = 0
    ratio_sum = 0.0
    ratio_max = 0.0

    for _ in range(config.train_iters):
        np.random.shuffle(indices)
        for start in range(0, sample_count, config.batch_size):
            mb_indices = indices[start : start + config.batch_size]
            new_log_probs, entropy, new_values = model.evaluate_actions(
                obs[mb_indices], actions[mb_indices]
            )
            mb_advantages = advantages[mb_indices].detach()
            mb_old_log_probs = old_log_probs[mb_indices]

            actor_loss, ratio = policy_loss(
                mode,
                new_log_probs,
                mb_old_log_probs,
                mb_advantages,
                config.clip_ratio,
            )

            with torch.no_grad():
                positive = mb_advantages > 0
                negative = mb_advantages < 0
                positive_upper_clipped = positive & (
                    ratio > 1.0 + config.clip_ratio
                )
                negative_lower_clipped = negative & (
                    ratio < 1.0 - config.clip_ratio
                )
                positive_kept = positive & ~positive_upper_clipped
                old_action_probs = mb_old_log_probs.exp()

                visit_count += ratio.numel()
                positive_visits += positive.sum().item()
                negative_visits += negative.sum().item()
                positive_upper_clipped_visits += (
                    positive_upper_clipped.sum().item()
                )
                negative_lower_clipped_visits += (
                    negative_lower_clipped.sum().item()
                )
                old_prob_clipped_sum += old_action_probs[
                    positive_upper_clipped
                ].sum().item()
                old_prob_clipped_count += positive_upper_clipped.sum().item()
                old_prob_positive_kept_sum += old_action_probs[
                    positive_kept
                ].sum().item()
                old_prob_positive_kept_count += positive_kept.sum().item()
                ratio_sum += ratio.sum().item()
                ratio_max = max(ratio_max, ratio.max().item())

                mb_indices_np = np.asarray(mb_indices)
                ever_positive_upper_clipped[mb_indices_np] |= (
                    positive_upper_clipped.cpu().numpy()
                )

            mb_old_values = old_values[mb_indices]
            clipped_values = mb_old_values + torch.clamp(
                new_values - mb_old_values,
                -config.value_clip_range,
                config.value_clip_range,
            )
            value_loss = torch.maximum(
                (new_values - returns[mb_indices]) ** 2,
                (clipped_values - returns[mb_indices]) ** 2,
            ).mean()
            total_loss = (
                actor_loss
                + config.value_loss_coef * value_loss
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
    unique_positive_count = int(positive_unique.sum())
    unique_positive_clipped_count = int(
        (ever_positive_upper_clipped & positive_unique).sum()
    )

    return {
        "mode": mode.value,
        "epoch": epoch + 1,
        "eval_reward": float(eval_reward),
        "rollout_samples": int(sample_count),
        "visit_count": int(visit_count),
        "positive_visits": int(positive_visits),
        "negative_visits": int(negative_visits),
        "positive_upper_clipped_visits": int(positive_upper_clipped_visits),
        "negative_lower_clipped_visits": int(negative_lower_clipped_visits),
        "unique_positive_count": unique_positive_count,
        "unique_positive_clipped_count": unique_positive_clipped_count,
        "old_prob_clipped_sum": float(old_prob_clipped_sum),
        "old_prob_clipped_count": int(old_prob_clipped_count),
        "old_prob_positive_kept_sum": float(old_prob_positive_kept_sum),
        "old_prob_positive_kept_count": int(old_prob_positive_kept_count),
        "ratio_sum": float(ratio_sum),
        "ratio_max": float(ratio_max),
    }


def percentage(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def safe_mean(total: float, count: int) -> float:
    return total / count if count else float("nan")


def format_epoch(record: dict[str, float | int | str]) -> str:
    return (
        f"epoch={record['epoch']:03d} reward={record['eval_reward']:6.1f} | "
        f"A>0 upper-clip visits="
        f"{percentage(record['positive_upper_clipped_visits'], record['positive_visits']):6.2f}% "
        f"unique={percentage(record['unique_positive_clipped_count'], record['unique_positive_count']):6.2f}% | "
        f"A<0 lower-clip visits="
        f"{percentage(record['negative_lower_clipped_visits'], record['negative_visits']):6.2f}% | "
        f"oldP(A+ clipped/kept)="
        f"{safe_mean(record['old_prob_clipped_sum'], record['old_prob_clipped_count']):.4f}/"
        f"{safe_mean(record['old_prob_positive_kept_sum'], record['old_prob_positive_kept_count']):.4f} | "
        f"ratio mean/max={record['ratio_sum'] / record['visit_count']:.4f}/"
        f"{record['ratio_max']:.4f}"
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
    summed_fields = (
        "visit_count",
        "positive_visits",
        "negative_visits",
        "positive_upper_clipped_visits",
        "negative_lower_clipped_visits",
        "unique_positive_count",
        "unique_positive_clipped_count",
        "old_prob_clipped_sum",
        "old_prob_clipped_count",
        "old_prob_positive_kept_sum",
        "old_prob_positive_kept_count",
        "ratio_sum",
    )
    aggregate: dict[str, float | int | str] = {
        field: sum(record[field] for record in selected) for field in summed_fields
    }
    aggregate.update(
        {
            "period": f"{start_epoch:02d}-{end_epoch:03d}",
            "epochs": len(selected),
            "eval_reward_mean": float(
                np.mean([record["eval_reward"] for record in selected])
            ),
            "eval_reward_min": float(
                np.min([record["eval_reward"] for record in selected])
            ),
            "ratio_max": max(record["ratio_max"] for record in selected),
        }
    )
    return aggregate


def format_period(mode: ActorObjective, row: dict[str, float | int | str]) -> str:
    return (
        f"{mode.value:17s} {row['period']} | "
        f"reward mean/min={row['eval_reward_mean']:7.2f}/{row['eval_reward_min']:6.1f} | "
        f"A>0 upper-clip visits="
        f"{percentage(row['positive_upper_clipped_visits'], row['positive_visits']):6.2f}% "
        f"unique={percentage(row['unique_positive_clipped_count'], row['unique_positive_count']):6.2f}% | "
        f"A<0 lower-clip visits="
        f"{percentage(row['negative_lower_clipped_visits'], row['negative_visits']):6.2f}% | "
        f"oldP clipped/kept="
        f"{safe_mean(row['old_prob_clipped_sum'], row['old_prob_clipped_count']):.4f}/"
        f"{safe_mean(row['old_prob_positive_kept_sum'], row['old_prob_positive_kept_count']):.4f} | "
        f"ratio mean/max={row['ratio_sum'] / row['visit_count']:.4f}/"
        f"{row['ratio_max']:.4f}"
    )


def run_experiment(
    mode: ActorObjective,
    config: ExperimentConfig,
    output_dir: Path,
) -> list[dict[str, float | int | str]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cpu")
    env = gym.make(config.env_name)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env.spec.max_episode_steps or config.steps_per_epoch

    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    normalizer = AdvantageNormalizer(momentum=0)
    buffer = RolloutBuffer(
        config.steps_per_epoch + max_episode_steps,
        obs_dim,
        normalizer=normalizer,
    )

    log_path = output_dir / f"{mode.value}.log"
    jsonl_path = output_dir / f"{mode.value}.jsonl"
    logger = configure_logger(log_path, mode)
    logger.info("START mode=%s config=%s", mode.value, asdict(config))

    records: list[dict[str, float | int | str]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for epoch in range(config.epochs):
            record = train_one_epoch(
                epoch=epoch,
                mode=mode,
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
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-ratio",
        type=float,
        default=ExperimentConfig.clip_ratio,
        help="PPO clip epsilon and the hypothetical threshold for unclipped modes.",
    )
    parser.add_argument(
        "--output-root",
        default=ExperimentConfig.output_root,
        help="Directory under which the timestamped result folder is created.",
    )
    return parser.parse_args()


def main() -> None:
    # These networks use tiny matrix multiplications; a large BLAS thread pool
    # costs far more in scheduling than it saves in compute.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args = parse_args()
    if not 0.0 < args.clip_ratio < 1.0:
        raise ValueError("--clip-ratio must be between 0 and 1")
    config = ExperimentConfig(
        clip_ratio=args.clip_ratio,
        output_root=args.output_root,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_label = f"clip_{config.clip_ratio:.2f}".replace(".", "p")
    output_dir = Path(config.output_root) / f"{timestamp}_{ratio_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_records: dict[ActorObjective, list[dict[str, float | int | str]]] = {}
    for mode in ActorObjective:
        all_records[mode] = run_experiment(mode, config, output_dir)

    periods = ((1, 33), (34, 66), (67, 100))
    summary_lines = [
        "Actor clip analysis summary",
        "A>0 upper-clip means ratio > 1 + clip_ratio.",
        "For PPO_CLIP these visits have zero actor-gradient contribution;",
        "for unclipped modes the same condition is hypothetical only.",
        "",
    ]
    summary_rows = []
    for mode, records in all_records.items():
        for start_epoch, end_epoch in periods:
            row = aggregate_period(records, start_epoch, end_epoch)
            summary_rows.append({"mode": mode.value, **row})
            summary_lines.append(format_period(mode, row))

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + summary_text)
    print(f"Detailed results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
