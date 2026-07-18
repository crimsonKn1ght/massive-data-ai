# galaxy-image-spectrum-clip

Cross-modal representation learning on the Multimodal Universe (MMU).

This project builds aligned image and spectrum data for the same astronomical objects by
cross-matching MMU surveys on sky position, then trains a small contrastive model that maps a
galaxy's image and its spectrum into one shared embedding space (an AstroCLIP-style alignment).
Once the two modalities share a space, an image can retrieve its spectrum and the learned
embedding predicts physical properties such as redshift.

The design keeps the encoders mostly frozen and trains only small projection heads plus a compact
spectrum encoder, following the connector-alignment recipe: cheap to train, single-GPU friendly,
with a synthetic path that runs end to end on CPU with no downloads.

Trained weights, precomputed features, per-checkpoint loss curves, and metrics are published on
Hugging Face: [grKnight/galaxy-image-spectrum-clip](https://huggingface.co/grKnight/galaxy-image-spectrum-clip).

## Why this is possible now

The Multimodal Universe (about 100 TB across 20+ surveys) has been republished in HATS format
(sky-tiled Parquet). LSDB reads those tiles lazily and cross-matches two catalogs by opening only
the overlapping sky tiles, so building an aligned image-plus-spectrum sample is a laptop-scale
operation instead of an 80 TB download. See the announcement:
https://huggingface.co/blog/hugging-science/multimodal-universe-hats

## Pipeline

The project is organized in three phases.

### Phase 0 - streaming cross-match (`crossmatch/`, `build_crossmatch.py`)

Open two HATS catalogs (Legacy Surveys images and DESI spectra), cross-match them on sky position,
and write an aligned dataset: per object an image array, a spectrum (flux and wavelength), catalog
columns (spectroscopic redshift, magnitudes), and sky coordinates. Output is written as compressed
`.npz` shards plus a `manifest.jsonl`, split deterministically by object into train/val/test so
there is no leakage.

### Phase 1 - frozen baselines (`run_baseline.py`, `eval/`)

Extract frozen image embeddings (CLIP) and a frozen spectrum feature, then measure the "before":
image-to-spectrum retrieval recall@k (expected near random) and a linear probe of each embedding
onto redshift. These numbers are the bar the trained model must beat.

### Phase 2 - contrastive alignment (`models/`, `training/`, `train.py`, `evaluate.py`)

Train a `CrossModalAlignment` model: a frozen CLIP image tower with a trainable projection, a small
trainable 1D spectrum encoder with its own projection, and a learnable temperature. A symmetric
InfoNCE loss pulls each object's image and spectrum together in the shared space and pushes
different objects apart. `evaluate.py` reruns retrieval and the redshift probe on the held-out test
split and reports the trained model against the Phase 1 baseline.

## Results

The full run trains on ~110k cross-matched pairs. Image-to-spectrum retrieval on the held-out test
split (scored over fixed 2000-object candidate pools) and a held-out linear probe onto redshift:

| | recall@1 | recall@5 | recall@10 | redshift probe R² (image / spectrum) |
|---|---|---|---|---|
| untrained baseline | ~0.001 | ~0.003 | ~0.005 | 0.58 / 0.41 |
| trained (110k) | 0.018 | 0.068 | 0.119 | 0.66 / 0.84 |

The alignment is real but coarse: the correct spectrum lands in the top ~12% of a 2000-object pool far
more often than chance, but seldom at rank 1, because the signal the two modalities reliably share is
close to redshift plus broad galaxy type. Training-set size mattered most — going from 16k to 110k
pairs roughly doubled recall@1 — and past that the frozen CLIP image tower is the limit, since its
features were never trained on galaxy imagery. The scaling and regularization studies, the per-checkpoint
loss curves, and the full analysis are in [`docs/phase2_findings.md`](docs/phase2_findings.md) and the
[model repo](https://huggingface.co/grKnight/galaxy-image-spectrum-clip).

## Quick start (synthetic, CPU, no downloads)

The synthetic mode generates paired image and spectrum arrays that share a hidden latent, so
alignment is genuinely learnable and every module can be exercised without a GPU or network access.

```
pip install -r requirements.txt

python build_crossmatch.py --synthetic --n 300 --output-dir aligned_smoke
python run_baseline.py --config configs/align_smoke.yaml
python train.py --config configs/align_smoke.yaml
python evaluate.py --config configs/align_smoke.yaml --checkpoint checkpoints/smoke/best
```

## Real run (single GPU, cached-feature path)

This reproduces the results in `docs/phase2_findings.md`. `configs/crossmatch_legacy_desi.yaml`
defaults to the full ~137.6k-pair build (`output.n_objects: 140000`), which streams ~40 GB from
HuggingFace and runs the CLIP feature precompute once on the GPU. For a quick end-to-end trial, lower
`output.n_objects` (e.g. to 20000) before building; everything downstream is identical.

```
# Resolve and verify the HATS catalog paths in configs/crossmatch_legacy_desi.yaml first.
# Set a HuggingFace token (export HF_TOKEN=...) to avoid rate-limited 504s.
python build_crossmatch.py    --config configs/crossmatch_legacy_desi.yaml               # ~40 GB, hours
python precompute_features.py --config configs/align.yaml --out-dir aligned/legacy_desi_clipfeat  # CLIP once, GPU
python run_baseline.py --config configs/align_cached.yaml
python train.py        --config configs/align_cached.yaml
python evaluate.py     --config configs/align_cached.yaml --checkpoint checkpoints/align_cached/best
```

The raw `aligned/legacy_desi` build is only needed by `precompute_features`; once the cached
`aligned/legacy_desi_clipfeat` exists you can delete the raw build to reclaim the ~40 GB. To train the
image tower on the fly instead of caching, swap `configs/align_cached.yaml` for `configs/align.yaml`
and skip the precompute step.

## Data sources and licensing

Base datasets: `MultimodalUniverse/legacysurvey` (images) and `MultimodalUniverse/desi` (spectra);
HATS versions are in the `UniverseTBD/multimodal-universe-hats` collection. Each underlying survey
carries its own terms; check the dataset cards before redistributing data or trained weights. The
code in this repository is released under the MIT License (see `LICENSE`).
