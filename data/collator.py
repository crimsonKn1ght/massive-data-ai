"""Batch collation for AlignedPairDataset items."""

from __future__ import annotations

from typing import Dict, List

import torch


class AlignedCollator:
    """Stack per-object tensors into batch tensors, keeping object ids as a list."""

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        images = torch.stack([b["image"] for b in batch])
        spectra = torch.stack([b["spectrum"] for b in batch])
        redshift = torch.stack([b["redshift"] for b in batch])
        object_ids = [b["object_id"] for b in batch]
        return {
            "images": images,
            "spectra": spectra,
            "redshift": redshift,
            "object_ids": object_ids,
        }
