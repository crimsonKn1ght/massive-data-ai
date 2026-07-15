"""Phase 2 evaluation: score a trained checkpoint on the held-out test split and compare to baseline.

    python evaluate.py --config configs/align.yaml --checkpoint checkpoints/align/best

Writes ``<eval.metrics_dir>/aligned.json`` and prints a baseline-vs-aligned table (retrieval recall@k
and the redshift-probe R2). Success is aligned retrieval well above the baseline and a higher probe R2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

from common import build_dataset, build_model, build_transforms, load_config, resolve_device
from eval.pipeline import evaluate_alignment
from training.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")


def _load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _print_comparison(baseline: Optional[dict], aligned: dict, recall_ks) -> None:
    def retrieval(metrics, key):
        return metrics["retrieval"].get(key, float("nan"))

    def probe_r2(metrics):
        return metrics["probe_redshift"]["image"].get("r2", float("nan"))

    header = f"{'metric':<22}{'baseline':>14}{'aligned':>14}"
    print("\n" + header)
    print("-" * len(header))
    for k in recall_ks:
        key = f"recall@{k}"
        base = f"{retrieval(baseline, key):.4f}" if baseline else "n/a"
        print(f"{'image->spectrum ' + key:<22}{base:>14}{retrieval(aligned, key):>14.4f}")
    base_r2 = f"{probe_r2(baseline):.4f}" if baseline else "n/a"
    print(f"{'redshift probe R2':<22}{base_r2:>14}{probe_r2(aligned):>14.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained alignment checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    logger.info("Device: %s", device)

    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, map_location=str(device))
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    image_transform, spectrum_transform = build_transforms(model, config)
    train_dataset = build_dataset(config, "train", image_transform, spectrum_transform)
    test_dataset = build_dataset(config, "test", image_transform, spectrum_transform)

    eval_cfg = config.get("eval", {})
    recall_ks = tuple(eval_cfg.get("recall_ks", [1, 5, 10]))
    aligned = evaluate_alignment(
        model,
        train_dataset,
        test_dataset,
        device=device,
        recall_ks=recall_ks,
        batch_size=int(config.get("training", {}).get("per_device_batch_size", 128)),
        num_workers=int(config.get("training", {}).get("dataloader_num_workers", 0)),
    )

    metrics_dir = eval_cfg.get("metrics_dir", "./metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "aligned.json"), "w") as f:
        json.dump(aligned, f, indent=2)

    baseline = _load_json(os.path.join(metrics_dir, "baseline.json"))
    _print_comparison(baseline, aligned, recall_ks)
    logger.info("Wrote %s", os.path.join(metrics_dir, "aligned.json"))


if __name__ == "__main__":
    main()
