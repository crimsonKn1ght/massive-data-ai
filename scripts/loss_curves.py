"""Write a run's training/validation loss (and val recall@k) curve across its checkpoints to a CSV.

A checkpoint is saved at every eval step and stores the spectrum encoder's BatchNorm buffers, so each
one reproduces the exact model state at that step. This loads every ``checkpoint-*`` in a run's output
dir, evaluates it on the val split for val InfoNCE loss and recall@k, and pairs it with the training
loss recorded in the checkpoint's meta.json - one CSV row per step.

    python scripts/loss_curves.py --config configs/align_cached.yaml \
        --checkpoints-dir checkpoints/align_cached --out full_run_110k_loss_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import build_dataset, build_model, build_transforms, load_config, resolve_device
from data.collator import AlignedCollator
from eval.retrieval import retrieval_metrics
from torch.utils.data import DataLoader
from training.checkpoint import load_checkpoint
from training.losses import info_nce


@torch.no_grad()
def eval_split(model, loader, device, ks):
    model.eval()
    total_loss, n_batches = 0.0, 0
    image_chunks, spectrum_chunks = [], []
    for batch in loader:
        image_emb, spectrum_emb, logit_scale = model(batch["images"].to(device), batch["spectra"].to(device))
        total_loss += info_nce(image_emb, spectrum_emb, logit_scale).item()
        n_batches += 1
        image_chunks.append(image_emb.float().cpu().numpy())
        spectrum_chunks.append(spectrum_emb.float().cpu().numpy())
    metrics = retrieval_metrics(np.concatenate(image_chunks), np.concatenate(spectrum_chunks), ks)
    return total_loss / max(1, n_batches), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a run's train/val loss curve from its checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints-dir", required=True, help="dir holding checkpoint-*/ (e.g. checkpoints/align_cached)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    model = build_model(config, device)
    image_transform, spectrum_transform = build_transforms(model, config)
    dataset = build_dataset(config, args.split, image_transform, spectrum_transform)
    batch_size = int(config.get("training", {}).get("per_device_batch_size", 128))
    num_workers = int(config.get("training", {}).get("dataloader_num_workers", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=AlignedCollator())
    ks = tuple(config.get("eval", {}).get("recall_ks", [1, 5, 10]))

    checkpoints = []
    for d in glob.glob(os.path.join(args.checkpoints_dir, "checkpoint-*")):
        if not os.path.isfile(os.path.join(d, "model.safetensors")):
            continue
        meta_path = os.path.join(d, "meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        try:
            step = int(meta.get("step", os.path.basename(d).rsplit("-", 1)[-1]))
        except ValueError:
            continue
        checkpoints.append((step, d, meta))
    checkpoints.sort()
    if not checkpoints:
        sys.exit(f"No checkpoint-*/ with model.safetensors found under {args.checkpoints_dir}")

    print(f"Evaluating {len(checkpoints)} checkpoints on the {args.split} split ({len(dataset)} objects)...")
    rows = []
    for step, d, meta in checkpoints:
        load_checkpoint(model, d, map_location=str(device))
        val_loss, metrics = eval_split(model, loader, device, ks)
        row = {"step": step, "train_loss": meta.get("train_loss"), "val_loss": round(val_loss, 4)}
        for k in ks:
            row[f"val_recall@{k}"] = round(float(metrics.get(f"recall@{k}", float("nan"))), 4)
        row["val_median_rank"] = metrics.get("median_rank")
        rows.append(row)
        print(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
