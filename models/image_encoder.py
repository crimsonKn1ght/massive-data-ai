"""Frozen image towers. The CLIP tower is the real encoder; the flatten tower is a weightless,
deterministic stand-in for the CPU smoke path so no CLIP weights are downloaded.

Both are frozen: ``embed`` runs under ``no_grad`` and returns detached features, so only the
projection head that follows receives gradient (the connector-alignment recipe). Each tower also
knows the image transform it needs, so the dataset has a single source of truth for preprocessing.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from data.image_processing import clip_image_transform, raw_image_transform


class ClipImageEncoder(nn.Module):
    """Frozen CLIP vision tower. Mirrors terraq-vl's vision_encoder select_layer/select_feature."""

    def __init__(self, model_name: str, select_layer: int = -2, select_feature: str = "pooled"):
        super().__init__()
        from transformers import CLIPVisionModel  # lazy: only needed for the real image tower

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.output_dim = self.model.config.hidden_size

    def build_transform(self, image_cfg: Dict) -> Callable[[np.ndarray], torch.Tensor]:
        return clip_image_transform(image_cfg)

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values, output_hidden_states=True)
        hidden = outputs.hidden_states[self.select_layer]  # (B, tokens, D)
        if self.select_feature == "pooled":
            feature = hidden[:, 0]  # CLS token
        elif self.select_feature == "patch_mean":
            feature = hidden[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unknown select_feature {self.select_feature!r} (pooled | patch_mean)")
        return feature.detach()


class FlattenImageEncoder(nn.Module):
    """Frozen, weightless tower: flatten the image and apply a fixed seeded random projection.

    Deterministic (seeded), so baseline and trained runs see identical image features and the
    comparison is fair. Used only on the synthetic smoke path.
    """

    def __init__(self, output_dim: int = 64, seed: int = 0):
        super().__init__()
        self.output_dim = output_dim
        self.seed = seed
        self.register_buffer("projection", torch.empty(0), persistent=False)
        self._input_dim = None

    def build_transform(self, image_cfg: Dict) -> Callable[[np.ndarray], torch.Tensor]:
        return raw_image_transform(image_cfg)

    def _ensure_projection(self, input_dim: int, device: torch.device) -> None:
        if self._input_dim == input_dim and self.projection.numel() > 0:
            return
        generator = torch.Generator().manual_seed(self.seed)
        weight = torch.randn(input_dim, self.output_dim, generator=generator) / (input_dim ** 0.5)
        self.projection = weight.to(device)
        self._input_dim = input_dim

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        self._ensure_projection(flat.shape[1], flat.device)
        return (flat @ self.projection).detach()


def build_image_encoder(image_cfg: Dict) -> nn.Module:
    """Construct the image tower named by ``image_cfg.type`` (clip | flatten)."""
    encoder_type = image_cfg.get("type", "clip")
    if encoder_type == "clip":
        return ClipImageEncoder(
            model_name=image_cfg["model_name"],
            select_layer=int(image_cfg.get("select_layer", -2)),
            select_feature=image_cfg.get("select_feature", "pooled"),
        )
    if encoder_type == "flatten":
        return FlattenImageEncoder(output_dim=int(image_cfg.get("output_dim", 64)))
    if encoder_type == "identity":
        return IdentityImageEncoder(output_dim=int(image_cfg["output_dim"]))
    raise ValueError(f"Unknown image_encoder type {encoder_type!r} (clip | flatten | identity)")


class IdentityImageEncoder(nn.Module):
    """Passthrough tower for precomputed image features (see precompute_features.py).

    When the frozen image features have been cached, training/eval load them directly instead of
    running the CLIP forward each step, which is the dominant cost on a small GPU. The dataset yields
    the feature vector as the "image", so ``embed`` just returns it (detached).
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def build_transform(self, image_cfg: Dict) -> Callable[[np.ndarray], torch.Tensor]:
        def transform(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(array, dtype=np.float32)).reshape(-1)

        return transform

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.detach()
