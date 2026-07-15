"""Write an aligned dataset (shards + manifest + deterministic split) from a stream of records.

Both the real cross-match (``crossmatch/lsdb_match.py``) and the synthetic generator
(``crossmatch/synthetic.py``) yield ``AlignedRecord`` objects; this module is the single sink that
buffers them into ``.npz`` shards, then assigns a seeded, per-object train/val/test split and writes
``manifest.jsonl``. Keeping the split per object (each matched object goes wholly into one split)
means there is no leakage between train, val and test.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, Iterable, List

import numpy as np

from .schema import AlignedRecord, ManifestEntry

logger = logging.getLogger(__name__)


def _shard_name(shard_index: int) -> str:
    return f"shard_{shard_index:05d}.npz"


class _ShardWriter:
    """Buffers records and flushes fixed-size ``.npz`` shards, collecting manifest rows as it goes."""

    def __init__(self, output_dir: str, shard_size: int):
        self.shard_dir = os.path.join(output_dir, "shards")
        os.makedirs(self.shard_dir, exist_ok=True)
        self.shard_size = shard_size
        self.shard_index = 0
        self.buffer: List[AlignedRecord] = []
        self.entries: List[ManifestEntry] = []

    def add(self, record: AlignedRecord) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard = _shard_name(self.shard_index)
        images = np.stack([r.image.astype(np.float32) for r in self.buffer])
        flux = np.stack([r.spectrum_flux.astype(np.float32) for r in self.buffer])
        wavelength = np.stack([r.spectrum_wavelength.astype(np.float32) for r in self.buffer])
        np.savez_compressed(
            os.path.join(self.shard_dir, shard),
            image=images,
            spectrum_flux=flux,
            spectrum_wavelength=wavelength,
        )
        for i, r in enumerate(self.buffer):
            self.entries.append(
                ManifestEntry(
                    object_id=r.object_id,
                    ra=float(r.ra),
                    dec=float(r.dec),
                    shard=shard,
                    index=i,
                    split="train",  # replaced by assign_splits() once all objects are known
                    catalog={k: float(v) for k, v in r.catalog.items()},
                )
            )
        logger.info("Wrote %s (%d objects)", shard, len(self.buffer))
        self.shard_index += 1
        self.buffer = []


def assign_splits(
    entries: List[ManifestEntry], seed: int, val_fraction: float, test_fraction: float
) -> Dict[str, int]:
    """Deterministically label each entry train/val/test in place; return per-split counts."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be in [0, 1)"
        )
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    n = len(entries)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test : n_test + n_val])
    counts = {"train": 0, "val": 0, "test": 0}
    for i, entry in enumerate(entries):
        if i in test_ids:
            entry.split = "test"
        elif i in val_ids:
            entry.split = "val"
        else:
            entry.split = "train"
        counts[entry.split] += 1
    return counts


def write_aligned_dataset(
    records: Iterable[AlignedRecord],
    output_dir: str,
    shard_size: int,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_objects: int | None = None,
) -> Dict[str, int]:
    """Consume a stream of records, write shards + a split manifest, and return split counts."""
    os.makedirs(output_dir, exist_ok=True)
    writer = _ShardWriter(output_dir, shard_size)

    n_seen = 0
    for record in records:
        writer.add(record)
        n_seen += 1
        if max_objects is not None and n_seen >= max_objects:
            break
    writer.flush()

    if not writer.entries:
        raise RuntimeError("No records were written - the record stream was empty.")

    counts = assign_splits(writer.entries, seed, val_fraction, test_fraction)

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in writer.entries:
            f.write(json.dumps(entry.to_json()) + "\n")

    logger.info(
        "Aligned dataset written to %s: %d objects (train=%d val=%d test=%d)",
        output_dir,
        len(writer.entries),
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return counts
