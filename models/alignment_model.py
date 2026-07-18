"""CrossModalAlignment: frozen image tower + trainable spectrum tower, aligned in a shared space.

Each modality is encoded, projected into a shared dimension, and L2-normalized; a learnable
temperature (stored as a log logit-scale, CLIP-style, clamped) scales the cosine similarities used
by the InfoNCE loss. Only the projections, the spectrum encoder, and the temperature are trainable;
the image tower is frozen, so ``trainable_state_dict`` excludes it and checkpoints stay small.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import build_image_encoder
from .projection import ProjectionHead
from .spectrum_encoder import SpectrumEncoder1D

logger = logging.getLogger(__name__)


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
            dropout=float(spec_cfg.get("dropout", 0.0)),
            augment_noise_std=float(spec_cfg.get("augment_noise_std", 0.0)),
            augment_mask_frac=float(spec_cfg.get("augment_mask_frac", 0.0)),
        )
        proj_cfg = config["projection"]
        shared_dim = int(proj_cfg.get("shared_dim", 512))
        hidden_dim = int(proj_cfg.get("hidden_dim", shared_dim))
        proj_dropout = float(proj_cfg.get("dropout", 0.0))
        image_projection = ProjectionHead(image_encoder.output_dim, shared_dim, hidden_dim, dropout=proj_dropout)
        spectrum_projection = ProjectionHead(spectrum_encoder.output_dim, shared_dim, hidden_dim, dropout=proj_dropout)
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
        # Migrate checkpoints written by the brief version where a Dropout inserted into the
        # projection MLP shifted the output Linear from index 2 to 3 (``mlp.3.*`` -> ``mlp.2.*``).
        # Only broken projection checkpoints carry ``.mlp.3.``; canonical ones never do, so this is a
        # no-op for them.
        state_dict = {k.replace(".mlp.3.", ".mlp.2."): v for k, v in state_dict.items()}
        result = self.load_state_dict(state_dict, strict=False)
        # The frozen image tower is intentionally excluded from checkpoints, so its keys are expected
        # to be missing. Any OTHER missing/unexpected key means an architecture mismatch that would
        # silently leave weights at random init (as an index shift once did), so surface it loudly.
        missing = [k for k in result.missing_keys if not k.startswith("image_encoder.")]
        if missing or result.unexpected_keys:
            logger.warning(
                "load_trainable_state_dict: checkpoint does not match the model - "
                "missing=%s unexpected=%s (loaded layers left at initialization)",
                missing,
                list(result.unexpected_keys),
            )
