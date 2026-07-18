# Phase 2: contrastive alignment — results and analysis

Phase 2 trains the `CrossModalAlignment` model that maps a galaxy's Legacy Surveys image and its DESI
spectrum into a shared embedding space, then measures how well an image retrieves its own spectrum and
how much of that space is organized by redshift. This note covers the training setup, the main model,
a data-scaling study, a regularization study, and where the current approach tops out.

Trained weights, precomputed features, per-checkpoint loss curves, and the metrics JSONs live in the
Hugging Face model repo:
[grKnight/galaxy-image-spectrum-clip](https://huggingface.co/grKnight/galaxy-image-spectrum-clip).

## Setup

The image tower is a frozen CLIP ViT-L/14; its penultimate CLS features are cached once and a trainable
two-layer MLP projects them into the shared space. The spectrum tower is a small 1-D CNN trained from
scratch with its own projection. Both projections write into a 512-d space, and a symmetric InfoNCE
loss with a learnable temperature pulls each object's two views together while pushing different
objects apart. Only the projections, the spectrum encoder, and the temperature are trained; the image
tower stays frozen, which keeps checkpoints small and training cheap.

The data is the Legacy Surveys (north) × DESI EDR SV3 cross-match on sky position within 1 arcsec —
about 137.6k good-quality pairs (`ZWARN == 0`), split 80/10/10 by object so no object leaks between
splits. Everything ran on a single RTX 2000 Ada in bf16.

## Main model — 110k training pairs

Retrieval is image → spectrum. Because chance recall scales with the number of candidates, all recall
numbers here are averaged over 20 random 2000-object pools drawn from the held-out test split, which
keeps them comparable across runs (`scripts/pooled_retrieval.py`):

| recall@1 | recall@5 | recall@10 |
|---|---|---|
| 0.0176 ± 0.0024 | 0.0677 ± 0.0046 | 0.1186 ± 0.0057 |

A held-out linear probe onto spectroscopic redshift reaches R² 0.84 from the spectrum embedding and
0.66 from the image embedding. The training and validation loss per checkpoint
(`full_run_110k/loss_curve.csv`) settles at train 2.96 / val 4.17.

The alignment is real but coarse. The correct spectrum lands in the top ~12% of a 2000-object pool far
more often than chance (recall@10 ≈ 0.12 against ≈0.005 random), yet rarely at rank 1. What the two
modalities reliably share — and what generalizes to held-out objects — is close to a one-dimensional
quantity: redshift, plus broad galaxy type, which thousands of objects have in common. That is enough
to pull the right spectrum much closer than random without singling it out.

## How much the data matters

Training-set size was the largest lever we found. Going from 16k to 110k training pairs (the same
pipeline, same config, just a bigger cross-match) roughly doubled retrieval and nearly halved the
generalization gap:

| | recall@1 (2000-pool) | recall@10 | train loss | val loss | train/val gap |
|---|---|---|---|---|---|
| 16k pairs | 0.0090 | 0.0745 | 2.46 | 4.81 | 2.35 |
| 110k pairs | 0.0176 | 0.1186 | 2.96 | 4.17 | 1.21 |

Two things to read here. First, the retrieval gain is genuine: on identically-sized candidate pools,
recall@1 went 0.0090 → 0.0176 (≈2×) with tight error bars. Raw recall@k over the *whole* test split is
not comparable between these runs — the larger build has a ~7× bigger test set, so a strictly better
model can post a lower raw recall@1 simply because it retrieves against more distractors. Fixing the
pool size removes that artifact. Second, the val loss dropped while the train loss rose, which is the
signature of less overfitting rather than more capacity — exactly what more data should do.

## Regularization

If the remaining train/val gap (1.21) were the thing limiting retrieval, regularization should buy
retrieval back. It did not. Adding dropout (0.2 in the projections and spectrum encoder), train-only
spectrum augmentation (Gaussian noise 0.1, bin masking 0.15), and heavier weight decay (0.1) — the
config is `configs/align_cached_reg.yaml` — nearly erased the gap but left retrieval flat, slightly
worse:

| | train loss | val loss | train/val gap | recall@1 (2000-pool) | image probe R² |
|---|---|---|---|---|---|
| 110k baseline | 2.96 | 4.17 | 1.21 | 0.0176 | 0.66 |
| 110k regularized | 3.80 | 4.005 | 0.20 | 0.0142 | 0.68 |

The gap all but closed and the redshift probe nudged up, but recall@1 fell to 0.0142. So at this scale
overfitting is no longer what holds retrieval back. The bottleneck has moved to the image features
themselves.

## Where this tops out

The image tower is frozen CLIP, pretrained on natural images and text and never on galaxy cutouts, and
only its projection head is trainable. Its redshift probe R² (0.66) trails the from-scratch spectrum
encoder's (0.84), and any fine image↔spectrum structure that is simply absent from the CLIP features
cannot be recovered by a projection on top of them. That caps top-1 retrieval regardless of how much
the spectrum side or the projections improve.

The natural next step, and the one with the most headroom, is a galaxy-appropriate image encoder — an
AstroCLIP- or DINO-style tower trained on galaxy imagery — in place of stock CLIP, or unfreezing part
of the tower. That needs the raw cutouts at feature-computation time and more compute, so it is a
larger piece of work than anything above, but it targets the actual limit.

## Notes on conditioning and throughput

A couple of design choices matter more than their size suggests.

Frozen CLIP penultimate features carry a few very large, near-constant "outlier" activations that
otherwise dominate the projection's input and leave it poorly conditioned (worse in bf16). A LayerNorm
at each projection's input puts every dimension on a comparable scale — and matches the per-object
normalization the flux side already gets — which noticeably improved optimization.

Because the image features are cached, training never re-runs CLIP. `precompute_features.py` also
resamples each spectrum onto the fixed model grid at cache time, so a shard holds 1024-bin spectra
rather than the full ~7.8k-sample DESI arrays, and the whole ~138k cached set is only a couple of GB.
The training dataset then keeps every shard resident in RAM (`shard_cache_size: 0`), which is why a full
30-epoch run finishes in about a minute on this GPU — the bounded LRU cache (the default for the
raw-image path) is kept only for the much larger uncached dataset.

## Reproduction

```
git clone https://github.com/crimsonKn1ght/galaxy-image-spectrum-clip
cd galaxy-image-spectrum-clip
pip install -r requirements.txt
export HF_TOKEN=...   # avoids rate-limited HATS reads

python build_crossmatch.py    --config configs/crossmatch_legacy_desi.yaml           # full ~138k build (~40 GB)
python precompute_features.py --config configs/align.yaml --out-dir aligned/legacy_desi_clipfeat
python run_baseline.py --config configs/align_cached.yaml
python train.py        --config configs/align_cached.yaml
python evaluate.py     --config configs/align_cached.yaml --checkpoint checkpoints/align_cached/best
python scripts/pooled_retrieval.py --config configs/align_cached.yaml --checkpoint checkpoints/align_cached/best
```

Lower `output.n_objects` in `configs/crossmatch_legacy_desi.yaml` for a quick trial. The precomputed
features in the model repo (`precomputed_features/`) let you skip the ~40 GB build and the CLIP
precompute and go straight to `run_baseline.py` / `train.py`.

## Practical notes

Report recall@k alongside median rank, and evaluate over a fixed candidate-pool size when comparing
runs whose test sets differ — raw recall@k scales with the candidate count and will otherwise mislead.

To size a cross-match before committing to a build, count matches from catalog metadata without pulling
any image or spectrum arrays:

```python
import lsdb
img = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north", columns=["ra", "dec", "object_id"])
spec = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_desi_edr_sv3", columns=["ra", "dec", "object_id", "ZWARN"])
matched = img.crossmatch(spec, n_neighbors=1, radius_arcsec=1.0, suffix_method="overlapping_columns")
df = matched.compute()
print("good matches:", int((df[[c for c in df.columns if c.startswith("ZWARN")][0]] == 0).sum()))
```

For Legacy-north × DESI-EDR-SV3 at 1 arcsec that returns ~137.6k good pairs. One caveat for full-scale
builds: `crossmatch/lsdb_match.py` computes each sky partition with no retry, so a transient read
failure over the multi-hour stream can abort the whole build — worth hardening with a bounded
retry-and-skip before scaling much further.
