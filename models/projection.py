"""Projection head: a 2-layer MLP (Linear -> GELU -> Linear) into the shared embedding space.

Same shape as the reference connector in terraq-vl (vlm_model/connector.py); here it maps each
modality's encoder output into the common space where the two are aligned.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or shared_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
