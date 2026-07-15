"""Phase 0 entry point: build an aligned image+spectrum dataset.

Real cross-match (requires the astro stack and verified HATS paths in the config):

    python build_crossmatch.py --config configs/crossmatch_legacy_desi.yaml

Synthetic smoke set (no downloads, no GPU):

    python build_crossmatch.py --synthetic --n 300 --output-dir aligned_smoke
"""

from __future__ import annotations

import argparse
import logging

import yaml

from crossmatch.build import write_aligned_dataset
from crossmatch.synthetic import synthetic_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_crossmatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an aligned image+spectrum dataset")
    parser.add_argument("--config", type=str, default=None, help="crossmatch config YAML (real data)")
    parser.add_argument("--synthetic", action="store_true", help="generate a synthetic smoke dataset")
    parser.add_argument("--output-dir", type=str, default="aligned_smoke", help="synthetic output dir")
    parser.add_argument("--n", type=int, default=300, help="synthetic object count")
    parser.add_argument("--image-size", type=int, default=16, help="synthetic image side length")
    parser.add_argument("--n-bins", type=int, default=128, help="synthetic spectrum length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Building synthetic aligned dataset: n=%d -> %s", args.n, args.output_dir)
        write_aligned_dataset(
            records=synthetic_records(
                n=args.n, seed=args.seed, image_size=args.image_size, n_bins=args.n_bins
            ),
            output_dir=args.output_dir,
            shard_size=128,
            seed=args.seed,
            val_fraction=0.2,
            test_fraction=0.2,
            max_objects=args.n,
        )
        return

    if not args.config:
        parser.error("provide --config for a real build, or --synthetic for the smoke set")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Imported here so the synthetic path does not require lsdb/hats/astropy.
    from crossmatch.lsdb_match import crossmatched_records

    out = config["output"]
    split = config["split"]
    write_aligned_dataset(
        records=crossmatched_records(config, max_objects=out.get("n_objects")),
        output_dir=out["output_dir"],
        shard_size=int(out.get("shard_size", 512)),
        seed=int(split.get("seed", 42)),
        val_fraction=float(split.get("val_fraction", 0.1)),
        test_fraction=float(split.get("test_fraction", 0.1)),
        max_objects=out.get("n_objects"),
    )


if __name__ == "__main__":
    main()
