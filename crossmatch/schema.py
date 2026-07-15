"""Shared shape for one cross-matched object and the on-disk aligned-dataset layout.

An aligned dataset is a directory with:

    <output_dir>/
        manifest.jsonl        one JSON object per matched source (see ManifestEntry)
        shards/shard_XXXXX.npz compressed arrays for a contiguous block of sources

Each shard stores three parallel arrays keyed ``image``, ``spectrum_flux`` and
``spectrum_wavelength``; the manifest row records which shard and index hold a given object,
its split, sky coordinates, and the scalar catalog fields used as probe targets. This mirrors
the on-disk-assets plus JSON-index layout used by the reference builders in terraq-vl, and lets
the dataset stream shards without holding the whole sample in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AlignedRecord:
    """One cross-matched object: an image, its spectrum, coordinates, and scalar catalog fields."""

    object_id: str
    ra: float
    dec: float
    image: np.ndarray                 # (C, H, W) float32
    spectrum_flux: np.ndarray         # (L,) float32
    spectrum_wavelength: np.ndarray   # (L,) float32
    catalog: Dict[str, float] = field(default_factory=dict)  # redshift, magnitudes, ...


@dataclass
class ManifestEntry:
    """A single line of ``manifest.jsonl``: where an object's arrays live plus its scalar fields."""

    object_id: str
    ra: float
    dec: float
    shard: str          # shard file name, e.g. "shard_00000.npz"
    index: int          # position of this object inside the shard arrays
    split: str          # "train" | "val" | "test"
    catalog: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ManifestEntry":
        return ManifestEntry(
            object_id=d["object_id"],
            ra=d["ra"],
            dec=d["dec"],
            shard=d["shard"],
            index=int(d["index"]),
            split=d["split"],
            catalog=d.get("catalog", {}),
        )


def catalog_value(entry: ManifestEntry, key: str) -> Optional[float]:
    """Return a scalar catalog field (e.g. ``redshift``) or ``None`` if the object lacks it."""
    value = entry.catalog.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
