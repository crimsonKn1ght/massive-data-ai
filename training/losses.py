"""Symmetric InfoNCE (CLIP-style) contrastive loss over in-batch image/spectrum positives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(image_emb: torch.Tensor, spectrum_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Each object's image and spectrum are the positive pair; all other pairs in the batch are
    negatives. Averages the image-to-spectrum and spectrum-to-image cross-entropies."""
    logits = logit_scale * image_emb @ spectrum_emb.t()  # (B, B)
    targets = torch.arange(image_emb.shape[0], device=image_emb.device)
    loss_image = F.cross_entropy(logits, targets)
    loss_spectrum = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_image + loss_spectrum)
