"""Phase 2 entry point: train the CrossModalAlignment model with InfoNCE.

    python train.py --config configs/align.yaml
    python train.py --config configs/align_smoke.yaml   # synthetic CPU smoke
"""

from __future__ import annotations

import argparse
import logging

from accelerate import Accelerator

from common import build_dataset, build_model, build_transforms, load_config, resolve_device
from training.trainer import AlignmentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cross-modal alignment model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    train_cfg = config.get("training", {})
    use_bf16 = bool(train_cfg.get("bf16", False)) and device.type == "cuda"

    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else "no",
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
    )

    logger.info("Device: %s | mixed_precision: %s", device, "bf16" if use_bf16 else "no")
    model = build_model(config, device)
    image_transform, spectrum_transform = build_transforms(model, config)
    train_dataset = build_dataset(config, "train", image_transform, spectrum_transform)
    val_dataset = build_dataset(config, "val", image_transform, spectrum_transform)
    logger.info("Train objects: %d | Val objects: %d", len(train_dataset), len(val_dataset))

    trainer = AlignmentTrainer(
        model=model,
        train_dataset=train_dataset,
        config=config,
        accelerator=accelerator,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
