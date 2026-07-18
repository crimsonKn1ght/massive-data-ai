"""Shared builders used by the Phase 1/2 entry points (run_baseline, train, evaluate).

Keeping model, transform, and dataset construction in one place guarantees the baseline, training,
and evaluation runs all see identical preprocessing and identical model configuration.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import yaml

from data.dataset import AlignedPairDataset
from data.spectrum_processing import make_spectrum_transform
from models.alignment_model import CrossModalAlignment


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("model", {}).get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_model(config: dict, device: torch.device) -> CrossModalAlignment:
    return CrossModalAlignment.from_config(config).to(device)


def build_transforms(model: CrossModalAlignment, config: dict) -> Tuple[Callable, Callable]:
    data_cfg = config["data"]
    image_transform = model.image_encoder.build_transform(data_cfg["image"])
    spectrum_transform = make_spectrum_transform(data_cfg["spectrum"])
    return image_transform, spectrum_transform


def build_dataset(
    config: dict, split: Optional[str], image_transform: Callable, spectrum_transform: Callable
) -> AlignedPairDataset:
    return AlignedPairDataset(
        aligned_dir=config["data"]["aligned_dir"],
        split=split,
        image_transform=image_transform,
        spectrum_transform=spectrum_transform,
        shard_cache_size=int(config["data"].get("shard_cache_size", 16)),
    )
