# Phase 2 findings: why cached-feature alignment recall is low, and what moves it

This documents the investigation into the `align_cached` training run where image-to-spectrum
retrieval recall@1 sat near chance while the training loss fell. It records what was ruled out, the
two fixes that were applied, the measured results on the real 16k/2k/2k dataset, the conclusion, and
the ordered next steps. Numbers below are from the CLIP ViT-L/14 cached-feature path in bf16 on the
RTX 2000 Ada single-GPU run.

## Symptom

- Train loss fell (5.43 -> ~4.2 in the original run) but val recall@1 stayed at 0.002-0.01.
- `nvidia-smi` showed the GPU idle (0% util, ~0.8 GB) at ~3.3 s per step.

## What was ruled out

1. **The training/eval pipeline is not broken.** On a synthetic dataset whose image and spectrum are
   built from a genuinely shared latent, the same code drives val recall@1 from 0.0025 (chance) to
   0.60-0.76. The InfoNCE loss, the `image_emb @ spectrum_emb.T` recall metric, the L2 normalization,
   and the manifest->shard indexing are all correct.

2. **BatchNorm train/eval mismatch is not the cause.** A known contrastive-learning failure is the
   spectrum encoder's `BatchNorm1d` leaking cross-sample batch statistics in training that vanish at
   eval. Measured on the best checkpoint:

   | val InfoNCE | value |
   |---|---|
   | eval mode (running stats) | 4.428 |
   | train mode (batch stats)  | 4.373 |

   Near-identical, so the large train-vs-val gap is genuine generalization failure, not a BN artifact.

## Fixes applied (committed)

Both are grounded in the observations above and were A/B-tested. Neither is a "make recall good"
switch; they remove two real defects.

### 1. Pre-projection LayerNorm (`models/projection.py`)

The projection heads consumed encoder features raw. Frozen CLIP penultimate hidden states carry a
few massive, near-constant "outlier" activations whose scale swamps the informative directions,
leaving the projection ill-conditioned (worse in bf16). The spectrum flux was already per-object
normalized, so the two towers were also inconsistent. A `LayerNorm` at each projection head's input
puts every dimension on a comparable scale.

- Effect on the real run: final **train loss 4.2 -> 2.57** (much stronger in-batch optimization).
- Does not regress the synthetic path (smoke recall@1 unchanged/slightly higher).

### 2. Cached-path dataloader cache (`data/dataset.py`, `common.py`, `configs/align_cached.yaml`, `training/trainer.py`)

Splits are interleaved across every shard, so the shuffled loader touched more shards per batch than
the 16-shard LRU held, re-decompressing `.npz` shards every step and starving the GPU. The cache size
is now configurable (`data.shard_cache_size`, `<= 0` keeps every shard resident); `align_cached.yaml`
sets `0` because the cached-feature dataset is tiny and fits in RAM. The raw (100 TB) path keeps the
bounded default. `persistent_workers` keeps warm worker caches across epochs.

- Effect: **~3.3 s/step -> ~0.035 s/step** (1860 steps in ~65 s; ~90x). GPU is now fed.
- Isolated measurement on a 30-shard set: **119 ms/batch -> 10 ms/batch**.

## Results on the real dataset (16k train / 2k val / 2k test)

Retrieval on the held-out test split (N = 2000):

| metric | baseline (untrained) | aligned (trained) |
|---|---|---|
| recall@1 | 0.0000 | 0.0050 |
| recall@5 | 0.0010 | 0.0380 |
| recall@10 | 0.0035 | 0.0715 |
| median rank | 949.5 | **142.0** |

Redshift linear-probe R2 on the held-out test split (generalizes by construction: probe trained on
train embeddings, scored on test):

| tower | baseline | aligned |
|---|---|---|
| image | 0.558 | 0.663 |
| spectrum | 0.405 | **0.841** |

## Conclusion

The alignment works, but only at a coarse level, and the model overfits the fine residual.

1. **Real, generalizing alignment.** Median rank fell from ~950 (random over 2000) to 142 - the true
   spectrum is pulled about 7x closer - and both towers encode redshift well on held-out data
   (spectrum R2 0.41 -> 0.84, image 0.56 -> 0.66). This is not noise.

2. **Coarse, redshift-dominated shared signal.** The generalizable structure shared between a frozen
   CLIP image feature and a DESI spectrum is essentially redshift plus broad galaxy type - a
   low-dimensional property that thousands of objects share. That is enough to rank the match into the
   top ~7% but not to isolate it at rank 1, which is exactly what recall@10 = 0.07 with recall@1 =
   0.005 shows.

3. **Overfitting of the fine residual.** With BN ruled out, the train-batch loss 2.57 vs val-batch
   loss ~4.4 is a genuine gap: beyond the coarse redshift alignment, the model memorizes
   training-pair-specific detail instead of finding generalizable fine structure.

4. **The frozen image tower is the likely ceiling.** The image side is frozen CLIP (pretrained on
   natural images and text, never on galaxy cutouts) and only its projection head is trainable. Its
   redshift R2 (0.66) trails the trained spectrum encoder's (0.84), and any fine image<->spectrum
   structure absent from the CLIP features cannot be recovered downstream. This caps top-1 retrieval
   regardless of the spectrum side or the projection.

## Recommended next steps (ordered by leverage vs cost)

Iteration is now ~1 minute per full run, so the cheap levers are worth trying first.

1. **Regularize to close the overfit gap (cheap, now).** Add dropout to the spectrum encoder and the
   projection heads, raise `weight_decay`, and/or shrink the spectrum encoder; add spectrum
   augmentation (Gaussian noise, wavelength masking). Success = val loss and the train/val gap drop
   and recall@5/@10 rise. If recall does not move while the gap closes, the ceiling (step 4 above) is
   binding, not overfitting.

2. **More matched pairs (moderate).** 16k pairs is small for contrastive learning; more data directly
   attacks overfitting. See "Counting available matches" below before committing to a bigger build.

3. **Replace or unfreeze the image tower (highest ceiling, biggest cost).** Swap frozen CLIP for a
   galaxy-appropriate image encoder (AstroCLIP / DINO trained on galaxy cutouts) or fine-tune part of
   the tower. This is the real unlock if step 1 confirms the ceiling is the image features, but it
   needs the raw images (not just cached features) and more compute.

## Evaluation practice

recall@1 over 2000 candidates is a stringent metric that hides coarse progress. Track recall@k and
median rank together (both already in `evaluate.py`); median rank moving is the earliest signal that
alignment is improving before recall@1 does.

## Counting available matches

The cross-match caps at `output.n_objects` (20000) in `configs/crossmatch_legacy_desi.yaml`. To learn
how many pairs the Legacy-north x DESI-EDR crossmatch can actually produce, count matches without
materializing images/spectra (pull only ra/dec/id/ZWARN, which is cheap):

```python
import lsdb
img = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
                        columns=["ra", "dec", "object_id"])
spec = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_desi_edr_sv3",
                         columns=["ra", "dec", "object_id", "ZWARN"])
matched = img.crossmatch(spec, n_neighbors=1, radius_arcsec=1.0,
                         suffix_method="overlapping_columns")
df = matched.compute()
print("total matches:", len(df))
zwarn = [c for c in df.columns if c.startswith("ZWARN")][0]
print("good quality (ZWARN == 0):", int((df[zwarn] == 0).sum()))
```

Set a Hugging Face token first (`export HF_TOKEN=...`) to avoid rate-limited 504s. The good-quality
count is the usable ceiling (`crossmatched_records` skips `ZWARN != 0`). To then build more, raise
`output.n_objects` and rerun `build_crossmatch.py`, `precompute_features.py`, `run_baseline.py`,
`train.py`, `evaluate.py`. If the count is only marginally above 20000, "more data" is not a lever and
attention should go to regularization (step 1) and the image tower (step 3).
