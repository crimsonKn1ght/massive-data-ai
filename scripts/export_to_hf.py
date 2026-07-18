"""Package this run's artifacts and push them to a Hugging Face model repo.

Runs on the training machine (where checkpoints/metrics/precomputed features live) and uploads them
into a structured HF repo, then uploads docs/model_card.md as the repo README. Only paths that exist
on disk are uploaded, so it works with whatever the current pod happens to have.

    export HF_TOKEN=<a WRITE token>          # or: huggingface-cli login
    python scripts/export_to_hf.py --repo-id grKnight/legacy-desi-alignment

Options:
    --public         create the repo public (default: private)
    --include-raw    also upload aligned/legacy_desi (the ~40 GB raw build; off by default)
    --checkpoints {all,best}   upload every checkpoint dir (default) or only best/
    --dry-run        print the upload plan and sizes without creating the repo or uploading
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def _human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def build_plan(args) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Return (folder_uploads, file_uploads) as (local_path, repo_path, label), skipping what's absent."""
    ckpt = "best" if args.checkpoints == "best" else "."  # subdir within a checkpoints dir

    def ckpt_src(name: str) -> str:
        base = os.path.join(REPO_ROOT, "checkpoints", name)
        return os.path.join(base, "best") if args.checkpoints == "best" else base

    def ckpt_dst(prefix: str) -> str:
        return f"{prefix}/checkpoints/best" if args.checkpoints == "best" else f"{prefix}/checkpoints"

    candidate_folders = [
        (ckpt_src("align_cached"), ckpt_dst("full_run_110k"), "full run checkpoints"),
        (os.path.join(REPO_ROOT, "metrics", "align_cached"), "full_run_110k/metrics", "full run metrics"),
        (ckpt_src("align_cached_reg"), ckpt_dst("ab_regularized_110k"), "regularized checkpoints"),
        (os.path.join(REPO_ROOT, "metrics", "align_cached_reg"), "ab_regularized_110k/metrics", "regularized metrics"),
        (os.path.join(REPO_ROOT, "aligned", "legacy_desi_clipfeat"), "precomputed_features", "precomputed features"),
        (os.path.join(REPO_ROOT, "configs"), "configs", "configs"),
    ]
    if args.include_raw:
        candidate_folders.append(
            (os.path.join(REPO_ROOT, "aligned", "legacy_desi"), "raw_crossmatch", "raw crossmatch build (~40 GB)")
        )

    folder_uploads = [(src, dst, label) for (src, dst, label) in candidate_folders if os.path.isdir(src)]

    candidate_files = [
        (os.path.join(REPO_ROOT, "docs", "model_card.md"), "README.md", "model card -> README"),
        (os.path.join(REPO_ROOT, "docs", "phase2_findings.md"), "docs/phase2_findings.md", "findings write-up"),
    ]
    file_uploads = [(src, dst, label) for (src, dst, label) in candidate_files if os.path.isfile(src)]
    return folder_uploads, file_uploads


def main() -> None:
    parser = argparse.ArgumentParser(description="Export run artifacts to a Hugging Face model repo")
    parser.add_argument("--repo-id", required=True, help="target repo, e.g. grKnight/legacy-desi-alignment")
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    parser.add_argument("--include-raw", action="store_true", help="also upload the ~40 GB raw build")
    parser.add_argument("--checkpoints", choices=["all", "best"], default="all")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and sizes; do not upload")
    args = parser.parse_args()

    folder_uploads, file_uploads = build_plan(args)

    print(f"Target repo: {args.repo_id} ({'public' if args.public else 'private'})")
    print("Upload plan:")
    total = 0
    for src, dst, label in folder_uploads:
        size = _dir_size(src)
        total += size
        print(f"  [dir ] {label:28s} {src}  ->  {dst}/   ({_human(size)})")
    for src, dst, label in file_uploads:
        size = os.path.getsize(src)
        total += size
        print(f"  [file] {label:28s} {src}  ->  {dst}   ({_human(size)})")
    print(f"Total to upload: {_human(total)}")

    missing_hint = []
    if not any(d.startswith("full_run_110k/checkpoints") for _s, d, _l in folder_uploads):
        missing_hint.append("checkpoints/align_cached")
    if not any(d == "precomputed_features" for _s, d, _l in folder_uploads):
        missing_hint.append("aligned/legacy_desi_clipfeat")
    if missing_hint:
        print(f"NOTE: not found on disk (skipped): {', '.join(missing_hint)}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    if not args.token:
        sys.exit("No HF token: set HF_TOKEN or pass --token (needs WRITE access).")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed (pip install huggingface_hub).")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    print(f"\nRepo ready: https://huggingface.co/{args.repo_id}")

    # Upload the README first so the repo is readable even if a large folder upload is interrupted.
    for src, dst, label in file_uploads:
        print(f"Uploading {label} ...")
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=args.repo_id, repo_type="model")
    for src, dst, label in folder_uploads:
        print(f"Uploading {label} ({_human(_dir_size(src))}) -> {dst}/ ...")
        api.upload_folder(
            folder_path=src,
            path_in_repo=dst,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {label}",
        )

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
