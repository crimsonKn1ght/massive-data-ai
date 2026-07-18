"""Spectrum preprocessing: resample flux onto a fixed wavelength grid and normalize.

The result is a fixed-length 1-D tensor used both as the frozen baseline feature (Phase 1) and as
the input to the trainable spectrum encoder (Phase 2), so the two phases see identical inputs.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch


def _fit_length(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Linearly resample a 1-D array to ``n_bins`` samples over its own index range."""
    if values.shape[0] == n_bins:
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
    return np.interp(dst, src, values).astype(np.float32)


def build_spectrum_grid(spectrum_cfg: Dict) -> np.ndarray:
    """Return the fixed wavelength grid (``n_bins`` samples over [wl_min, wl_max]) the model expects."""
    wl_min = float(spectrum_cfg.get("wavelength_min", 3600.0))
    wl_max = float(spectrum_cfg.get("wavelength_max", 9800.0))
    n_bins = int(spectrum_cfg.get("n_bins", 1024))
    return np.linspace(wl_min, wl_max, n_bins).astype(np.float32)


def resample_spectrum(flux: np.ndarray, wavelength: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample a raw ``(flux, wavelength)`` spectrum onto ``grid``; no normalization.

    Single source of truth for the resampling step so the precompute cache and the live transform
    cannot drift. When the flux is already on ``grid`` (cached-feature path), ``np.interp`` samples the
    same points and returns it unchanged, so re-running this in the dataset is a no-op.
    """
    flux = np.asarray(flux, dtype=np.float32).reshape(-1)
    wavelength = np.asarray(wavelength, dtype=np.float32).reshape(-1)
    if wavelength.shape[0] == flux.shape[0] and wavelength.shape[0] > 1:
        return np.interp(grid, wavelength, flux, left=flux[0], right=flux[-1]).astype(np.float32)
    return _fit_length(flux, grid.shape[0])


def make_spectrum_transform(spectrum_cfg: Dict) -> Callable[[np.ndarray, np.ndarray], torch.Tensor]:
    """Build a ``(flux, wavelength) -> 1-D tensor`` transform from the ``data.spectrum`` config."""
    grid = build_spectrum_grid(spectrum_cfg)

    def transform(flux: np.ndarray, wavelength: np.ndarray) -> torch.Tensor:
        resampled = resample_spectrum(flux, wavelength, grid)
        median = float(np.median(resampled))
        std = float(np.std(resampled))
        normalized = (resampled - median) / (std + 1e-6)
        return torch.from_numpy(normalized.astype(np.float32))

    return transform
