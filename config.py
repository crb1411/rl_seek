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
    
@dataclass
class TrainingConfig:
    env_name: str = "CartPole-v1"
    steps_per_epoch: int = 2000
    epochs: int = 100
    gamma: float = 0.99
    lam: float = 0.95
    pi_lr: float = 3e-4
    train_iters: int = 10
    batch_size: int = 32
    render_test: bool = False
    policy_target: Advantage_Policy = Advantage_Policy.PPO_GAE
    use_adv_normalizer: bool = False
    use_clip: bool = False
    clip_ratio: float = 0.2
    save_root: str | Path = "/data/seek/rl_rundata/logs_ac"
    checkpoint_dir: str | Path = None
    checkpoint_freq: int = 20
    resume_path: Optional[str] = None
    use_wandb: bool = False
    run_name: Optional[str] = None
    wandb_entity: str = "crb_1411"
    wandb_project: str = "seek_rl"
    device: str = "auto"
