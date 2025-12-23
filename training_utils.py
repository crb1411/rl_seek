from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from advantage_normalizer import AdvantageNormalizer


def init_wandb(enable: bool, save_root: Path, config: dict, run_name: Optional[str] = None,
               resume_id: Optional[str] = None, entity: str = "crb_1411",
               project: str = "seek_rl"):
    if not enable:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; proceeding without wandb logging.")
        return None

    return wandb.init(
        entity=entity,
        project=project,
        dir=str(save_root),
        name=run_name,
        id=resume_id,
        resume="allow" if resume_id else None,
        config=config,
    )


def _normalizer_state(normalizer: AdvantageNormalizer | None):
    if normalizer is None:
        return None
    return {
        "mean": normalizer.mean,
        "var": normalizer.var,
        "momentum": normalizer.momentum,
        "eps": normalizer.eps,
    }


def _load_normalizer_state(normalizer: AdvantageNormalizer | None, state):
    if normalizer is None or state is None:
        return
    normalizer.mean = state.get("mean")
    normalizer.var = state.get("var")
    normalizer.momentum = state.get("momentum", normalizer.momentum)
    normalizer.eps = state.get("eps", normalizer.eps)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: optim.Optimizer,
                    normalizer: AdvantageNormalizer | None, epoch: int, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": _normalizer_state(normalizer),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: optim.Optimizer,
                    normalizer: AdvantageNormalizer | None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    _load_normalizer_state(normalizer, state.get("normalizer"))
    return state.get("epoch", -1), state.get("config")
