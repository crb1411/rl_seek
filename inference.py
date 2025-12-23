from pathlib import Path
from typing import Tuple

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


def run_inference(model_path: str | Path, env_name: str = "CartPole-v1",
                  episodes: int = 3, video_dir: str | Path = "videos",
                  name_prefix: str = "ppo_eval", device: str | torch.device = "auto") -> Tuple[float, Path]:
    """
    Load a checkpointed model, run deterministic policy inference, and record video.
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_dir=str(video_dir), name_prefix=name_prefix)

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    ac = ActorCritic(obs_dim, act_dim).to(device)

    state = torch.load(model_path, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    ac.load_state_dict(state_dict)
    ac.eval()

    total_reward = 0.0
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

    env.close()
    avg_reward = total_reward / episodes
    return avg_reward, video_dir
