"""Cross-modal retrieval metrics: given aligned image and spectrum embeddings, how well does an
image retrieve its own spectrum? recall@k and median rank over the held-out set."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def retrieval_metrics(
    image_emb: np.ndarray, spectrum_emb: np.ndarray, ks: Sequence[int] = (1, 5, 10)
) -> Dict[str, float]:
    """Image-to-spectrum retrieval. Row i's correct match is column i (same object)."""
    n = image_emb.shape[0]
    if n == 0:
        return {"n": 0}
    sims = image_emb @ spectrum_emb.T                      # (N, N) cosine sims (embeddings are L2-normalized)
    order = np.argsort(-sims, axis=1)                      # indices sorted by descending similarity
    ranks = np.argmax(order == np.arange(n)[:, None], axis=1)  # 0-based rank of the correct match
    metrics: Dict[str, float] = {"n": int(n)}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks < k))
    metrics["median_rank"] = float(np.median(ranks) + 1)  # report 1-based
    return metrics
