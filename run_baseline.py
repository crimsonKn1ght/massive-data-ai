"""Phase 1 entry point: record the "before" numbers with an untrained alignment model.

The image tower is a fixed frozen encoder and the projections/spectrum encoder are randomly
initialized, so this is the honest pre-alignment control: image-to-spectrum retrieval should be near
chance, and the redshift probe measures the signal already present in the frozen representations.
The same ``evaluate_alignment`` is used here and in ``evaluate.py`` so the comparison is apples to
apples. Results are written to ``<eval.metrics_dir>/baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from common import build_dataset, build_model, build_transforms, load_config, resolve_device
from eval.pipeline import evaluate_alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 frozen baseline (retrieval + probes)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    logger.info("Device: %s", device)

    model = build_model(config, device)
    image_transform, spectrum_transform = build_transforms(model, config)
    train_dataset = build_dataset(config, "train", image_transform, spectrum_transform)
    test_dataset = build_dataset(config, "test", image_transform, spectrum_transform)
    logger.info("Train objects: %d | Test objects: %d", len(train_dataset), len(test_dataset))

    eval_cfg = config.get("eval", {})
    metrics = evaluate_alignment(
        model,
        train_dataset,
        test_dataset,
        device=device,
        recall_ks=tuple(eval_cfg.get("recall_ks", [1, 5, 10])),
        batch_size=int(config.get("training", {}).get("per_device_batch_size", 128)),
        num_workers=int(config.get("training", {}).get("dataloader_num_workers", 0)),
    )

    metrics_dir = eval_cfg.get("metrics_dir", "./metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, "baseline.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Baseline retrieval: %s", metrics["retrieval"])
    logger.info("Baseline redshift probe (image): %s", metrics["probe_redshift"]["image"])
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
