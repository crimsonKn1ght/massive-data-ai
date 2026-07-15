# massive-data-ai

Cross-modal representation learning on the Multimodal Universe (MMU).

This project builds aligned image and spectrum data for the same astronomical objects by
cross-matching MMU surveys on sky position, then trains a small contrastive model that maps a
galaxy's image and its spectrum into one shared embedding space (an AstroCLIP-style alignment).
Once the two modalities share a space, an image can retrieve its spectrum and the learned
embedding predicts physical properties such as redshift.

The design keeps the encoders mostly frozen and trains only small projection heads plus a compact
spectrum encoder, following the connector-alignment recipe: cheap to train, single-GPU friendly,
with a synthetic path that runs end to end on CPU with no downloads.

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

## Real run (single GPU; data prototyped on a laptop first)

```
# Resolve and verify the HATS catalog paths in configs/crossmatch_legacy_desi.yaml first.
python build_crossmatch.py --config configs/crossmatch_legacy_desi.yaml
python run_baseline.py --config configs/align.yaml
python train.py --config configs/align.yaml
python evaluate.py --config configs/align.yaml --checkpoint checkpoints/align/best
```

## Data sources and licensing

Base datasets: `MultimodalUniverse/legacysurvey` (images) and `MultimodalUniverse/desi` (spectra);
HATS versions are in the `UniverseTBD/multimodal-universe-hats` collection. Each underlying survey
carries its own terms; check the dataset cards before redistributing data or trained weights. The
code in this repository is released under the MIT License (see `LICENSE`).
