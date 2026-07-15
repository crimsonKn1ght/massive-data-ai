"""Shared evaluation: embed a dataset with the alignment model, then run retrieval + probes.

Both the Phase 1 baseline (an untrained model) and Phase 2 evaluation (a loaded checkpoint) call
``evaluate_alignment`` so the numbers are computed exactly the same way and are directly comparable.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.collator import AlignedCollator
from eval.probes import probe_regression
from eval.retrieval import retrieval_metrics


@torch.no_grad()
def compute_embeddings(
    model, dataset, device: torch.device, batch_size: int = 128, num_workers: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_emb, spectrum_emb, redshift) as numpy arrays for every object in ``dataset``."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=AlignedCollator(),
    )
    image_chunks, spectrum_chunks, redshift_chunks = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        spectra = batch["spectra"].to(device)
        image_emb, spectrum_emb, _ = model(images, spectra)
        image_chunks.append(image_emb.float().cpu().numpy())
        spectrum_chunks.append(spectrum_emb.float().cpu().numpy())
        redshift_chunks.append(batch["redshift"].numpy())
    return (
        np.concatenate(image_chunks),
        np.concatenate(spectrum_chunks),
        np.concatenate(redshift_chunks),
    )


def evaluate_alignment(
    model,
    train_dataset,
    test_dataset,
    device: torch.device,
    recall_ks: Sequence[int] = (1, 5, 10),
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, object]:
    """Retrieval on the test split; redshift probe trained on train, scored on test."""
    train_image, train_spectrum, train_z = compute_embeddings(
        model, train_dataset, device, batch_size, num_workers
    )
    test_image, test_spectrum, test_z = compute_embeddings(
        model, test_dataset, device, batch_size, num_workers
    )
    return {
        "retrieval": retrieval_metrics(test_image, test_spectrum, recall_ks),
        "probe_redshift": {
            "image": probe_regression(train_image, train_z, test_image, test_z),
            "spectrum": probe_regression(train_spectrum, train_z, test_spectrum, test_z),
        },
    }
