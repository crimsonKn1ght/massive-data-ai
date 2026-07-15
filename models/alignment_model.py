"""CrossModalAlignment: frozen image tower + trainable spectrum tower, aligned in a shared space.

Each modality is encoded, projected into a shared dimension, and L2-normalized; a learnable
temperature (stored as a log logit-scale, CLIP-style, clamped) scales the cosine similarities used
by the InfoNCE loss. Only the projections, the spectrum encoder, and the temperature are trainable;
the image tower is frozen, so ``trainable_state_dict`` excludes it and checkpoints stay small.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import build_image_encoder
from .projection import ProjectionHead
from .spectrum_encoder import SpectrumEncoder1D


class CrossModalAlignment(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        spectrum_encoder: nn.Module,
        image_projection: nn.Module,
        spectrum_projection: nn.Module,
        temperature_init: float = 0.07,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder
        self.image_projection = image_projection
        self.spectrum_projection = spectrum_projection
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32))
        self.max_log_logit_scale = math.log(max_logit_scale)

    @classmethod
    def from_config(cls, config: Dict) -> "CrossModalAlignment":
        image_encoder = build_image_encoder(config["image_encoder"])
        spec_cfg = config["spectrum_encoder"]
        spectrum_encoder = SpectrumEncoder1D(
            hidden_channels=tuple(spec_cfg.get("hidden_channels", [64, 128, 256])),
            kernel_size=int(spec_cfg.get("kernel_size", 5)),
            embedding_dim=int(spec_cfg.get("embedding_dim", 512)),
        )
        proj_cfg = config["projection"]
        shared_dim = int(proj_cfg.get("shared_dim", 512))
        hidden_dim = int(proj_cfg.get("hidden_dim", shared_dim))
        image_projection = ProjectionHead(image_encoder.output_dim, shared_dim, hidden_dim)
        spectrum_projection = ProjectionHead(spectrum_encoder.output_dim, shared_dim, hidden_dim)
        temperature_init = float(config.get("model", {}).get("temperature_init", 0.07))
        return cls(image_encoder, spectrum_encoder, image_projection, spectrum_projection, temperature_init)

    def logit_scale(self) -> torch.Tensor:
        return self.log_logit_scale.clamp(max=self.max_log_logit_scale).exp()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder.embed(images)
        return F.normalize(self.image_projection(features), dim=-1)

    def encode_spectrum(self, spectra: torch.Tensor) -> torch.Tensor:
        features = self.spectrum_encoder(spectra)
        return F.normalize(self.spectrum_projection(features), dim=-1)

    def forward(self, images: torch.Tensor, spectra: torch.Tensor):
        return self.encode_image(images), self.encode_spectrum(spectra), self.logit_scale()

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the trainable parts only (excludes the frozen image tower)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("image_encoder.")}

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)
