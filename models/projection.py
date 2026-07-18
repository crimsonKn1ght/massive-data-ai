"""Projection head: a 2-layer MLP (Linear -> GELU -> Linear) into the shared embedding space.

Same shape as the reference connector in terraq-vl (vlm_model/connector.py); here it maps each
modality's encoder output into the common space where the two are aligned.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or shared_dim
        # Normalize the encoder features before projection. Frozen image-tower features (e.g. cached
        # CLIP penultimate hidden states) carry a handful of massive, near-constant "outlier"
        # activations whose scale swamps the informative directions; feeding them in raw makes the
        # projection ill-conditioned and starves alignment. A LayerNorm puts every input dimension on
        # a comparable scale (and matches the per-spectrum normalization the flux side already gets),
        # which is standard for CLIP-style connectors over frozen features.
        self.input_norm = nn.LayerNorm(input_dim)
        # Optional dropout to fight overfitting; ``dropout=0.0`` (default) is a no-op and adds no
        # parameters. The two Linears are given explicit names ("0" and "2") so that inserting the
        # (parameterless) Dropout does NOT shift the output Linear's state_dict key - keeping
        # checkpoints saved without dropout loadable. Do not renumber these.
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("0", nn.Linear(input_dim, hidden_dim)),
                    ("1", nn.GELU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("2", nn.Linear(hidden_dim, shared_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.input_norm(x))
