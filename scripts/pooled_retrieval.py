"""Fixed-pool retrieval evaluation, for comparing models across dataset sizes.

Raw recall@k over the whole test split is not comparable between runs whose test sets differ in size
(chance recall@1 is 1/N). This script fixes the candidate-pool size: it embeds the test split once,
then averages image-to-spectrum recall@k over many random subsets of a fixed size, so a 20k run and a
138k run can be compared on equal footing (see docs/phase2_findings.md).

    python scripts/pooled_retrieval.py --config configs/align_cached.yaml \
        --checkpoint checkpoints/align_cached/best
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# Allow running as ``python scripts/pooled_retrieval.py`` from the repo root: put the repo root (this
# file's parent directory) on the path so the top-level modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import build_dataset, build_model, build_transforms, load_config, resolve_device
from eval.pipeline import compute_embeddings
from eval.retrieval import retrieval_metrics
from training.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pooled_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-pool image-to-spectrum retrieval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pool", type=int, default=2000, help="candidate-pool size per draw")
    parser.add_argument("--repeats", type=int, default=20, help="number of random pools to average")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, map_location=str(device))
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    image_transform, spectrum_transform = build_transforms(model, config)
    dataset = build_dataset(config, args.split, image_transform, spectrum_transform)
    batch_size = int(config.get("training", {}).get("per_device_batch_size", 128))
    num_workers = int(config.get("training", {}).get("dataloader_num_workers", 0))
    image_emb, spectrum_emb, _ = compute_embeddings(model, dataset, device, batch_size, num_workers)

    n = image_emb.shape[0]
    ks = tuple(config.get("eval", {}).get("recall_ks", [1, 5, 10]))
    pool = min(args.pool, n)
    if pool < n:
        rng = np.random.default_rng(args.seed)
        rows = []
        for _ in range(args.repeats):
            idx = rng.choice(n, size=pool, replace=False)
            m = retrieval_metrics(image_emb[idx], spectrum_emb[idx], ks)
            rows.append([m[f"recall@{k}"] for k in ks])
        rows = np.asarray(rows)
        logger.info("Pooled over %d random subsets of %d (from %d %s objects):", args.repeats, pool, n, args.split)
        for j, k in enumerate(ks):
            logger.info("  recall@%-2d = %.4f +/- %.4f", k, rows[:, j].mean(), rows[:, j].std())
    else:
        # Pool covers the whole split: a single deterministic pass (no subsampling possible).
        m = retrieval_metrics(image_emb, spectrum_emb, ks)
        logger.info("Full-split retrieval over %d %s objects (pool >= split size):", n, args.split)
        for k in ks:
            logger.info("  recall@%-2d = %.4f", k, m[f"recall@{k}"])
        logger.info("  median_rank = %.1f", m["median_rank"])


if __name__ == "__main__":
    main()
