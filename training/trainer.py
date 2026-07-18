"""Training loop for CrossModalAlignment.

Mirrors the reference VLMTrainer in terraq-vl: AdamW over the trainable parameters only, a
cosine-warmup schedule, gradient accumulation and clipping, periodic logging, a held-out validation
pass every ``eval_steps`` (InfoNCE loss + image-to-spectrum recall@1), and checkpointing every
``save_steps`` with a ``best/`` copy kept by best validation recall@1. The frozen image tower is held
in eval mode throughout. Written for single-device / single-process runs (the stated target); it uses
Accelerate for mixed precision and device placement.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data.collator import AlignedCollator
from eval.retrieval import retrieval_metrics
from training.checkpoint import save_checkpoint
from training.losses import info_nce
from training.lr_scheduler import build_cosine_warmup_scheduler

logger = logging.getLogger(__name__)


class AlignmentTrainer:
    def __init__(self, model, train_dataset, config: Dict[str, Any], accelerator: Accelerator, val_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.accelerator = accelerator

        train_cfg = config.get("training", {})
        self.output_dir = train_cfg.get("output_dir", "./checkpoints/align")
        self.num_epochs = int(train_cfg.get("num_epochs", 30))
        self.per_device_batch_size = int(train_cfg.get("per_device_batch_size", 128))
        self.gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.learning_rate = float(train_cfg.get("learning_rate", 5e-4))
        self.warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.05))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.logging_steps = int(train_cfg.get("logging_steps", 10))
        self.save_steps = int(train_cfg.get("save_steps", 100))
        self.eval_steps = int(train_cfg.get("eval_steps", self.save_steps))
        self.dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))
        self.recall_ks = tuple(config.get("eval", {}).get("recall_ks", [1, 5, 10]))
        self.best_recall = -1.0

    def _set_train_mode(self) -> None:
        self.model.train()
        self.accelerator.unwrap_model(self.model).image_encoder.eval()

    def train(self) -> None:
        collator = AlignedCollator()
        # persistent_workers keeps the (fork-copied) worker shard caches warm across epochs instead of
        # rebuilding them from scratch each epoch; only valid when workers are actually spawned.
        persistent = self.dataloader_num_workers > 0
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            collate_fn=collator,
            drop_last=True,
            persistent_workers=persistent,
        )
        val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.per_device_batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                collate_fn=collator,
                drop_last=False,
                persistent_workers=persistent,
            )

        num_update_steps_per_epoch = max(1, len(train_loader) // self.gradient_accumulation_steps)
        num_training_steps = num_update_steps_per_epoch * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = build_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)

        self.model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Trainable parameters: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", 100.0 * trainable / total)
        logger.info("Total update steps: %d (warmup %d)", num_training_steps, num_warmup_steps)

        global_step = 0
        running_loss = 0.0
        running_count = 0
        start_time = time.time()
        grad_checked = False
        unwrapped = self.accelerator.unwrap_model(self.model)
        trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]

        self._set_train_mode()
        for epoch in range(self.num_epochs):
            if global_step >= num_training_steps:
                break
            logger.info("Starting epoch %d/%d", epoch + 1, self.num_epochs)
            for batch in train_loader:
                with self.accelerator.accumulate(self.model):
                    image_emb, spectrum_emb, logit_scale = self.model(batch["images"], batch["spectra"])
                    loss = info_nce(image_emb, spectrum_emb, logit_scale)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if not grad_checked:
                            assert any(p.grad is not None for p in trainable_params), (
                                "No trainable parameter received gradient - the projections/spectrum "
                                "encoder are detached from the loss."
                            )
                            grad_checked = True
                        self.accelerator.clip_grad_norm_(trainable_params, self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                running_loss += loss.detach().item()
                running_count += 1

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = running_loss / max(1, running_count)
                        elapsed = time.time() - start_time
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "Step %d/%d | Loss: %.4f | LR: %.2e | %.1fs",
                            global_step, num_training_steps, avg_loss, lr, elapsed,
                        )
                        running_loss = 0.0
                        running_count = 0

                    if val_loader is not None and global_step % self.eval_steps == 0:
                        val_loss, val_recall = self._evaluate(val_loader)
                        logger.info(
                            "Step %d/%d | Val loss: %.4f | Val recall@1: %.4f",
                            global_step, num_training_steps, val_loss, val_recall,
                        )
                        self._maybe_save_best(optimizer, scheduler, global_step, val_loss, val_recall)

                    if global_step % self.save_steps == 0 and self.accelerator.is_main_process:
                        save_checkpoint(
                            unwrapped, optimizer, scheduler, global_step, {"train_loss": loss.item()}, self.output_dir
                        )
                        logger.info("Saved checkpoint at step %d", global_step)

                    if global_step >= num_training_steps:
                        break

        if self.accelerator.is_main_process:
            final_metrics: Dict[str, Any] = {"train_loss": loss.item()}
            if val_loader is not None:
                val_loss, val_recall = self._evaluate(val_loader)
                final_metrics.update({"val_loss": val_loss, "val_recall@1": val_recall})
                logger.info("Final | Val loss: %.4f | Val recall@1: %.4f", val_loss, val_recall)
                self._maybe_save_best(optimizer, scheduler, global_step, val_loss, val_recall)
            save_checkpoint(unwrapped, optimizer, scheduler, global_step, final_metrics, self.output_dir)
            if self.best_recall < 0:
                # No validation set: the final checkpoint is the best we have.
                save_checkpoint(unwrapped, optimizer, scheduler, global_step, final_metrics, self.output_dir, tag="best")
            logger.info("Training complete at step %d", global_step)

    def _maybe_save_best(self, optimizer, scheduler, step: int, val_loss: float, val_recall: float) -> None:
        if not self.accelerator.is_main_process:
            return
        if val_recall > self.best_recall:
            self.best_recall = val_recall
            unwrapped = self.accelerator.unwrap_model(self.model)
            save_checkpoint(
                unwrapped, optimizer, scheduler, step,
                {"val_loss": val_loss, "val_recall@1": val_recall}, self.output_dir, tag="best",
            )
            logger.info("New best val recall@1 %.4f at step %d -> saved best/", val_recall, step)

    @torch.no_grad()
    def _evaluate(self, val_loader) -> Tuple[float, float]:
        """Return (mean InfoNCE loss, image-to-spectrum recall@1) over the validation loader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        image_chunks, spectrum_chunks = [], []
        for batch in val_loader:
            image_emb, spectrum_emb, logit_scale = self.model(batch["images"], batch["spectra"])
            total_loss += info_nce(image_emb, spectrum_emb, logit_scale).item()
            n_batches += 1
            image_chunks.append(image_emb.float().cpu().numpy())
            spectrum_chunks.append(spectrum_emb.float().cpu().numpy())
        self._set_train_mode()
        image_emb = np.concatenate(image_chunks)
        spectrum_emb = np.concatenate(spectrum_chunks)
        recall = retrieval_metrics(image_emb, spectrum_emb, (1,)).get("recall@1", float("nan"))
        return total_loss / max(1, n_batches), recall
