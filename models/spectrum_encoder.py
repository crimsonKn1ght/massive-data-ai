"""Trainable 1-D CNN spectrum encoder.

No strong frozen pretrained spectrum encoder exists, so this small tower is trained jointly with
the projection heads - the single, deliberate deviation from the "freeze everything" recipe. It maps
a fixed-length flux vector to an embedding via a stack of Conv1d blocks and global average pooling.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SpectrumEncoder1D(nn.Module):
    def __init__(
        self,
        hidden_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        augment_noise_std: float = 0.0,
        augment_mask_frac: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in hidden_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Dropout on the pooled features fights overfitting; 0.0 (default) is a no-op and adds no
        # parameters, so checkpoints stay compatible.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_channels, embedding_dim)
        self.output_dim = embedding_dim
        # Train-only input augmentation (both default 0.0 = off): Gaussian noise and random bin
        # masking on the normalized flux, which enlarges the effective sample and discourages the
        # encoder from memorizing per-spectrum quirks.
        self.augment_noise_std = float(augment_noise_std)
        self.augment_mask_frac = float(augment_mask_frac)

    def _augment(self, spectra: torch.Tensor) -> torch.Tensor:
        if self.augment_noise_std > 0.0:
            spectra = spectra + self.augment_noise_std * torch.randn_like(spectra)
        if self.augment_mask_frac > 0.0:
            keep = (torch.rand_like(spectra) >= self.augment_mask_frac).to(spectra.dtype)
            spectra = spectra * keep
        return spectra

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        if self.training:  # augmentation is applied only in training mode (off during eval/retrieval)
            spectra = self._augment(spectra)
        x = spectra.unsqueeze(1)  # (B, 1, L)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.dropout(x)
        return self.head(x)  # (B, embedding_dim)
