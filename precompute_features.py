"""Precompute frozen image-tower features so alignment training skips the CLIP forward each step.

Reads an aligned dataset (produced by build_crossmatch.py), runs the configured image encoder over
every image once, and writes a parallel aligned dataset whose shards hold the image *features* in
place of the raw cutouts. Spectra are resampled onto the fixed model grid here too (the manifest is
copied through unchanged), so the cached shards are small and no per-step spectrum interpolation is
needed. Training and evaluation then use a config with ``image_encoder.type: identity`` pointed at the
feature dataset, which loads features directly - the per-step image load, resize, and CLIP forward
disappear.

    python precompute_features.py --config configs/align.yaml --out-dir aligned/legacy_desi_clipfeat

The feature dimension equals the source image encoder's output_dim (1024 for CLIP ViT-L/14); set that
as ``image_encoder.output_dim`` in the cached (identity) config.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil

import numpy as np
import torch

from common import load_config, resolve_device
from data.spectrum_processing import build_spectrum_grid, resample_spectrum
from models.image_encoder import build_image_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen image-tower features for fast training")
    parser.add_argument("--config", type=str, required=True, help="config with the source image encoder")
    parser.add_argument("--out-dir", type=str, required=True, help="output aligned dataset (features)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    logger.info("Device: %s", device)

    encoder = build_image_encoder(config["image_encoder"]).to(device).eval()
    image_cfg = config["data"]["image"]
    logger.info("Image encoder output_dim: %d (set this as image_encoder.output_dim in the cached config)", encoder.output_dim)

    # Resample spectra onto the fixed model grid once, here, instead of storing raw full-resolution
    # spectra and re-resampling every training step. This shrinks the cached shards (a raw DESI
    # spectrum is ~7.8k samples; the grid is n_bins, e.g. 1024) and removes the per-item interp at
    # train time. Normalization stays in the live transform, so re-running resample on the cached grid
    # is a no-op and training output is unchanged.
    spectrum_grid = build_spectrum_grid(config["data"]["spectrum"])

    src = config["data"]["aligned_dir"]
    src_shards = os.path.join(src, "shards")
    out_shards = os.path.join(args.out_dir, "shards")
    os.makedirs(out_shards, exist_ok=True)
    shutil.copy(os.path.join(src, "manifest.jsonl"), os.path.join(args.out_dir, "manifest.jsonl"))

    shard_names = sorted(n for n in os.listdir(src_shards) if n.endswith(".npz"))
    for shard_name in shard_names:
        with np.load(os.path.join(src_shards, shard_name)) as data:
            images = data["image"]
            spectrum_flux = data["spectrum_flux"]
            spectrum_wavelength = data["spectrum_wavelength"]

        features = []
        for start in range(0, len(images), args.batch_size):
            batch = images[start : start + args.batch_size]
            with torch.no_grad():
                features.append(encoder.batch_embed(batch, image_cfg, device).float().cpu().numpy())
        features_array = (
            np.concatenate(features) if features else np.zeros((0, encoder.output_dim), dtype=np.float32)
        )

        if len(spectrum_flux):
            resampled_flux = np.stack(
                [resample_spectrum(spectrum_flux[i], spectrum_wavelength[i], spectrum_grid) for i in range(len(spectrum_flux))]
            ).astype(np.float32)
        else:
            resampled_flux = np.zeros((0, spectrum_grid.shape[0]), dtype=np.float32)
        # Store the grid per object so the (shard, index) contract and the dataset transform are
        # unchanged; the repeated grid compresses to almost nothing.
        grid_wavelength = np.broadcast_to(spectrum_grid, resampled_flux.shape).astype(np.float32)

        np.savez_compressed(
            os.path.join(out_shards, shard_name),
            image=features_array.astype(np.float32),
            spectrum_flux=resampled_flux,
            spectrum_wavelength=grid_wavelength,
        )
        logger.info("Wrote %s: features %s | spectra %s", shard_name, features_array.shape, resampled_flux.shape)

    logger.info("Feature dataset written to %s", args.out_dir)


if __name__ == "__main__":
    main()
