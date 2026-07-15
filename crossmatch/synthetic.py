"""Synthetic aligned records for the CPU smoke path (no downloads, no GPU).

Each object draws a hidden latent vector ``z``; its image and its spectrum are both fixed linear
functions of ``z`` plus noise, and its "redshift" is a fixed function of ``z``. Because the two
modalities share ``z``, a contrastive model can genuinely learn to align them (retrieval rises well
above chance) and a linear probe can recover the redshift - so the smoke run exercises the real
success criteria, not just the plumbing. The projection matrices are seeded, so the whole dataset
is deterministic.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .schema import AlignedRecord


def synthetic_records(
    n: int,
    seed: int = 42,
    image_size: int = 16,
    n_bins: int = 128,
    bands: int = 3,
    latent_dim: int = 16,
    noise: float = 0.1,
) -> Iterator[AlignedRecord]:
    """Yield ``n`` deterministic aligned records whose image and spectrum share a latent."""
    rng = np.random.default_rng(seed)

    image_pixels = bands * image_size * image_size
    # Fixed (seeded) maps from the shared latent to each modality and to redshift.
    image_map = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(image_pixels, latent_dim)).astype(np.float32)
    spec_map = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(n_bins, latent_dim)).astype(np.float32)
    redshift_map = rng.normal(size=(latent_dim,)).astype(np.float32)

    wavelength = np.linspace(3600.0, 9800.0, n_bins).astype(np.float32)

    for i in range(n):
        z = rng.normal(size=(latent_dim,)).astype(np.float32)

        image = image_map @ z + noise * rng.normal(size=(image_pixels,)).astype(np.float32)
        image = image.reshape(bands, image_size, image_size)

        flux = spec_map @ z + noise * rng.normal(size=(n_bins,)).astype(np.float32)

        redshift = float(1.0 / (1.0 + np.exp(-(redshift_map @ z))))  # sigmoid -> (0, 1)

        yield AlignedRecord(
            object_id=f"synthetic_{i:07d}",
            ra=float(rng.uniform(0.0, 360.0)),
            dec=float(rng.uniform(-90.0, 90.0)),
            image=image.astype(np.float32),
            spectrum_flux=flux.astype(np.float32),
            spectrum_wavelength=wavelength.copy(),
            catalog={"redshift": redshift},
        )
