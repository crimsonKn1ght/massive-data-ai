"""Save/load the trainable parts of the alignment model (projections + spectrum encoder + temperature).

The frozen image tower is never written, so checkpoints stay small; each checkpoint dir holds
``model.safetensors`` + ``training_state.pt`` (optimizer/scheduler) + ``meta.json`` (step and metrics),
mirroring the reference checkpoint format in terraq-vl.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    metrics: Dict[str, Any],
    output_dir: str,
    tag: Optional[str] = None,
) -> str:
    """Write ``<output_dir>/<tag or checkpoint-step>/`` and return the directory path."""
    name = tag or f"checkpoint-{step}"
    checkpoint_dir = os.path.join(output_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_file(model.trainable_state_dict(), os.path.join(checkpoint_dir, "model.safetensors"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    meta = {"step": step, **metrics}
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return checkpoint_dir


def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load trainable weights (and optionally optimizer/scheduler); return the saved ``meta.json``."""
    state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_trainable_state_dict(state_dict)

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if (optimizer is not None or scheduler is not None) and os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=map_location, weights_only=True)
        if optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])

    meta_path = os.path.join(checkpoint_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}
