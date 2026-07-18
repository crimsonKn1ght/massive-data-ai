---
license: mit
tags:
  - astronomy
  - astrophysics
  - contrastive-learning
  - cross-modal-retrieval
  - clip
  - galaxy-spectra
  - multimodal-universe
library_name: pytorch
pipeline_tag: feature-extraction
---

# Cross-modal alignment: Legacy Surveys images <-> DESI spectra

AstroCLIP-style contrastive alignment of galaxy **images** (Legacy Surveys grz) and **spectra**
(DESI EDR SV3) into a shared embedding space, cross-matched on sky position from the
[Multimodal Universe](https://huggingface.co/blog/hugging-science/multimodal-universe-hats) HATS
catalogs. A frozen CLIP ViT-L/14 image tower (features cached once) with a trainable projection, a
small trainable 1-D CNN spectrum encoder, and a symmetric InfoNCE loss.

Code: https://github.com/crimsonKn1ght/galaxy-image-spectrum-clip — this repo stores the trained
artifacts, metrics, precomputed features, and the full write-up (`docs/phase2_findings.md`).

## Repository layout

```
full_run_110k/            main model: 110k train pairs, no regularization
  checkpoints/            model.safetensors (trainable weights only) + optimizer/scheduler + meta.json
  metrics/                baseline.json (Phase-1 untrained control) + aligned.json (evaluate output)
  loss_curve.csv          per-checkpoint training and validation loss + val recall@k
ab_regularized_110k/      regularized variant: same data with dropout + spectrum augmentation + weight decay
  checkpoints/
  metrics/
  loss_curve.csv
precomputed_features/     cached CLIP ViT-L/14 image features + resampled 1024-bin spectra + manifest
configs/                  the exact YAML configs used
docs/phase2_findings.md   training setup, results, and analysis
README.md                 this card
```

`raw_crossmatch/` (the ~40 GB raw image+spectrum build) is included only if exported with
`--include-raw`; it is regenerable from the code and configs.

## Results

Retrieval is image -> spectrum. `pooled recall@k` averages over 20 random 2000-object candidate pools
(the size-independent comparison; see below). `gap` is final train minus val InfoNCE loss.

| run | train pairs | pooled recall@1 | pooled recall@10 | val loss | train/val gap | image probe R2 |
|---|---|---|---|---|---|---|
| Phase-1 baseline (untrained) | - | ~0.0005 (chance) | ~0.005 | ~5.5 | - | 0.576 |
| trained, 20k | 16k | 0.0090 | 0.0745 | 4.81 | 2.35 | 0.659 |
| **trained, 110k (main)** | **110k** | **0.0176** | **0.1186** | **4.17** | **1.21** | **0.663** |
| 110k + regularized | 110k | 0.0142 | 0.1056 | 4.005 | 0.20 | 0.684 |

The redshift linear-probe R2 (held-out) reaches **0.84 on the spectrum tower** and 0.66-0.68 on the
(frozen) image tower.

### What the numbers say

- The pipeline learns real alignment: on genuinely-alignable synthetic data it reaches recall@1 ~0.6;
  the Phase-1 baseline here is chance.
- **More data is the dominant lever.** Going 20k -> 110k pairs halved the overfitting gap
  (2.35 -> 1.21) and roughly doubled pooled recall@1 (0.0090 -> 0.0176). (Raw recall@k over the full
  test set is *not* comparable across runs because the test pool grows with the dataset - hence the
  fixed-pool metric.)
- **Regularization is not the bottleneck at 110k.** It nearly closed the gap (-> 0.20) but retrieval
  did not improve, which locates the remaining ceiling at the **frozen CLIP image tower** (image probe
  R2 0.67 vs spectrum 0.84). CLIP was never trained on galaxy cutouts; a galaxy-appropriate image
  encoder is the next lever.

Retrieval is coarse: it ranks the true spectrum into roughly the top ~12% of a 2000-object pool
(recall@10 ~0.12) but rarely at rank 1 - the shared, generalizable signal is largely redshift plus
broad galaxy type.

## Using a checkpoint

Checkpoints hold only the trainable parts (projections + spectrum encoder + temperature); the frozen
image tower is rebuilt at load. With the code repo:

```python
from common import load_config, build_model, resolve_device
from training.checkpoint import load_checkpoint

config = load_config("configs/align_cached.yaml")   # from configs/ in this repo
device = resolve_device(config)
model = build_model(config, device)
load_checkpoint(model, "full_run_110k/checkpoints/best", map_location=str(device))
model.eval()
```

`configs/align_cached.yaml` uses `image_encoder.type: identity`, i.e. it expects **precomputed** CLIP
features as the image input (see `precomputed_features/`). To run on new raw images instead, use
`configs/align.yaml`, which runs CLIP ViT-L/14 on the fly.

## Reproduce

```
git clone https://github.com/crimsonKn1ght/galaxy-image-spectrum-clip && cd galaxy-image-spectrum-clip
pip install -r requirements.txt
export HF_TOKEN=...   # avoids rate-limited HATS reads
python build_crossmatch.py    --config configs/crossmatch_legacy_desi.yaml   # ~40 GB (n_objects: 140000)
python precompute_features.py --config configs/align.yaml --out-dir aligned/legacy_desi_clipfeat
python run_baseline.py --config configs/align_cached.yaml
python train.py        --config configs/align_cached.yaml
python evaluate.py     --config configs/align_cached.yaml --checkpoint checkpoints/align_cached/best
python scripts/pooled_retrieval.py --config configs/align_cached.yaml --checkpoint checkpoints/align_cached/best
```

Lower `output.n_objects` for a quick trial. Full method and rationale in `docs/phase2_findings.md`.

## Data sources and licensing

Base datasets: `MultimodalUniverse/legacysurvey` (images) and `MultimodalUniverse/desi` (spectra);
HATS versions in the `UniverseTBD/multimodal-universe-hats` collection. Each underlying survey carries
its own terms - check the dataset cards before redistributing data or trained weights. The code is
released under the MIT License; these artifacts are derived from the surveys above.
