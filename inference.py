import json
from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import torch

from models import ActorCritic


def evaluate_policy(ac, env, episodes=3, render=False, device=None):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = ac(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
    return total_reward / episodes


def record_policy_videos(
    ac: ActorCritic,
    env_name: str,
    episodes: int,
    video_dir: str | Path,
    name_prefix: str = "final-policy",
    device: str | torch.device = "auto",
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Record deterministic policy episodes and write their statistics."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if episodes == 0:
        return []

    video_dir = Path(video_dir)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=name_prefix,
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    was_training = ac.training
    ac.eval()
    records: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            seed = base_seed + episode_index
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    logits, _ = ac(obs_tensor)
                    action = logits.argmax(dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

            records.append(
                {
                    "episode": episode_index,
                    "seed": seed,
                    "reward": total_reward,
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "video": str(
                        (video_dir / f"{name_prefix}-episode-{episode_index}.mp4")
                        .resolve()
                    ),
                }
            )
    finally:
        env.close()
        ac.train(was_training)

    summary_path = video_dir / "summary.json"
    summary_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def run_inference(model_path: str | Path, env_name: str = "CartPole-v1",
                  episodes: int = 3, video_dir: str | Path = "videos",
                  name_prefix: str = "ppo_eval", device: str | torch.device = "auto") -> Tuple[float, Path]:
    """
    Load a checkpointed model, run deterministic policy inference, and record video.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    probe_env = gym.make(env_name)
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.n
    probe_env.close()
    ac = ActorCritic(obs_dim, act_dim).to(device)

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    ac.load_state_dict(state_dict)

    records = record_policy_videos(
        ac=ac,
        env_name=env_name,
        episodes=episodes,
        video_dir=video_dir,
        name_prefix=name_prefix,
        device=device,
    )
    avg_reward = sum(record["reward"] for record in records) / len(records)
    return avg_reward, Path(video_dir)

if __name__ == "__main__":
    checkpoint_path = Path(
        "outputs/training/LunarLander-v3/"
        "LunarLander-no-value-clip-20260814-v1/"
        "value-clip-3/checkpoints/latest.pt"
    )
    video_dir = checkpoint_path.parent.parent / "videos" / "manual-inference"
    run_inference(
        model_path=checkpoint_path,
        env_name="LunarLander-v3",
        episodes=5,
        video_dir=video_dir,
        name_prefix="eval",
    )
