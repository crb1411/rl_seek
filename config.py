from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum
import torch


class Advantage_Policy(Enum):
    RETURN = 0
    ADVANTAGE = 1
    TD_ERROR = 2
    PPO_GAE = 3
    ADVANTAGE_DISCOUNTED = 4
    # Standard PPO: GAE actor advantage + lambda-return critic target.
    STANDARD_PPO = 5
    
@dataclass
class TrainingConfig:
    env_name: str = "CartPole-v1"
    # Total transitions across all rollout environments per PPO epoch.
    steps_per_epoch: int = 8192
    # One uses an in-process vector env; values > 1 use subprocess workers.
    num_envs: int = 8
    epochs: int = 100
    gamma: float = 0.99
    lam: float = 0.95
    pi_lr: float = 3e-4
    train_iters: int = 10
    batch_size: int = 256
    render_test: bool = False
    policy_target: Advantage_Policy = Advantage_Policy.PPO_GAE
    use_adv_normalizer: bool = True
    use_clip: bool = False
    clip_ratio: float = 0.2
    # Policy clipping and critic value clipping are independent controls.
    use_value_clip: bool = True
    value_clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    # If save_root is None, the run directory is built under output_root as:
    # outputs/training/<env>/<experiment>/<run>.
    output_root: str | Path = "outputs"
    experiment_name: str = "manual"
    save_root: Optional[str | Path] = None
    checkpoint_dir: str | Path = "checkpoints"
    checkpoint_freq: int = 20
    resume_path: Optional[str] = None
    # Record deterministic evaluation episodes after training. Set to 0 to disable.
    video_episodes: int = 5
    # Relative subdirectory inside save_root (the current run directory).
    video_dir: str | Path = "videos"
    use_wandb: bool = False
    run_name: Optional[str] = None
    wandb_entity: str = "crb_1411"
    wandb_project: str = "seek_rl"
    wandb_group: Optional[str] = None
    seed: int = 20260814
    device: str = "auto"
