"""AlignedPairDataset: read the manifest + shards and yield (image, spectrum, redshift) per object.

Shards are loaded lazily through a small LRU cache so a large aligned dataset does not have to fit
in memory. Each item applies the configured image and spectrum transforms, so the dataset is the
single place the two modalities are turned into model-ready tensors.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from crossmatch.schema import ManifestEntry, catalog_value

_SHARD_KEYS = ("image", "spectrum_flux", "spectrum_wavelength")


class AlignedPairDataset(Dataset):
    def __init__(
        self,
        aligned_dir: str,
        split: Optional[str],
        image_transform: Callable[[np.ndarray], "torch.Tensor"],
        spectrum_transform: Callable[[np.ndarray, np.ndarray], "torch.Tensor"],
        redshift_key: str = "redshift",
        shard_cache_size: int = 16,
    ):
        self.shard_dir = os.path.join(aligned_dir, "shards")
        self.image_transform = image_transform
        self.spectrum_transform = spectrum_transform
        self.redshift_key = redshift_key
        self.entries: List[ManifestEntry] = self._load_entries(aligned_dir, split)
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        # A bounded LRU keeps the streaming footprint small for the huge raw dataset. Precomputed
        # feature shards are tiny, though, and the shuffled loader touches every shard each step, so a
        # 16-shard cache thrashes (re-decompressing shards every batch) and starves the GPU. Setting
        # shard_cache_size <= 0 keeps every shard resident (load-once), which is the right choice for
        # the cached-feature path where the whole dataset comfortably fits in RAM.
        self._cache_size = shard_cache_size

    @staticmethod
    def _load_entries(aligned_dir: str, split: Optional[str]) -> List[ManifestEntry]:
        manifest_path = os.path.join(aligned_dir, "manifest.jsonl")
        entries: List[ManifestEntry] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = ManifestEntry.from_json(json.loads(line))
                if split is None or entry.split == split:
                    entries.append(entry)
        if not entries:
            raise RuntimeError(f"No manifest entries for split={split!r} in {aligned_dir}")
        return entries

    def _shard(self, name: str) -> Dict[str, np.ndarray]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(os.path.join(self.shard_dir, name)) as data:
            arrays = {key: data[key] for key in _SHARD_KEYS}
        self._cache[name] = arrays
        if self._cache_size > 0 and len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        shard = self._shard(entry.shard)
        image = self.image_transform(shard["image"][entry.index])
        spectrum = self.spectrum_transform(
            shard["spectrum_flux"][entry.index], shard["spectrum_wavelength"][entry.index]
        )
        redshift = catalog_value(entry, self.redshift_key)
        return {
            "image": image,
            "spectrum": spectrum,
            "redshift": torch.tensor(float("nan") if redshift is None else redshift, dtype=torch.float32),
            "object_id": entry.object_id,
        }
