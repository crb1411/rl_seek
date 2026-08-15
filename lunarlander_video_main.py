"""Render comparable LunarLander videos from the final value-clip checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from models import ActorCritic


RUN_DIRS = {
    "no_clip": "no-value-clip",
    "clip_0p5": "value-clip-0p5",
    "clip_1": "value-clip-1",
    "clip_3": "value-clip-3",
}
DEFAULT_EXPERIMENT_ROOT = Path(
    "outputs/training/LunarLander-v3/"
    "LunarLander-no-value-clip-20260814-v1"
)
DEFAULT_SEEDS = (31001, 31002, 31003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    parser.add_argument(
        "--video-set", default="fixed-seeds-20260814"
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = ActorCritic(obs_dim=8, act_dim=4).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def record_variant(
    *,
    variant: str,
    checkpoint_path: Path,
    output_dir: Path,
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, int | float | str | bool]]:
    variant_dir = output_dir
    variant_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path, device)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(variant_dir),
        episode_trigger=lambda _: True,
        name_prefix=variant,
        disable_logger=True,
    )

    records: list[dict[str, int | float | str | bool]] = []
    for episode_index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        video_path = variant_dir / f"{variant}-episode-{episode_index}.mp4"
        record = {
            "variant": variant,
            "seed": seed,
            "reward": total_reward,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "video": str(video_path.resolve()),
        }
        records.append(record)
        print(
            f"{variant:9s} seed={seed} reward={total_reward:8.2f} "
            f"steps={steps:4d} video={video_path}"
        )

    env.close()
    return records


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    all_records: list[dict[str, int | float | str | bool]] = []
    for variant, run_dir_name in RUN_DIRS.items():
        run_dir = args.experiment_root / run_dir_name
        checkpoint_path = run_dir / "checkpoints/epoch_100.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        all_records.extend(
            record_variant(
                variant=variant,
                checkpoint_path=checkpoint_path,
                output_dir=run_dir / "videos" / args.video_set,
                seeds=list(args.seeds),
                device=device,
            )
        )

    summary_path = args.experiment_root / f"{args.video_set}-summary.json"
    summary_path.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
