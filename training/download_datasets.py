#!/usr/bin/env python3
"""
Download chess recognition datasets for training.

Downloads:
1. chesscog rendered dataset from OSF (synthetic 3D boards)
2. samryan18/chess-dataset from GitHub (real photos)

Usage:
    python3 training/download_datasets.py [--chesscog] [--samryan18] [--all]
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent / "data"
DOWNLOADS_DIR = BASE_DIR / "downloads"


def download_chesscog():
    """Download chesscog rendered dataset from OSF project xf3ka."""
    dest = BASE_DIR / "chesscog_render"
    if dest.exists() and any(dest.rglob("*.png")):
        count = sum(1 for _ in dest.rglob("*.png"))
        print(f"  Already have {count} images in {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print("  Downloading from OSF project xf3ka...")
    print("  This is ~4GB, will take a while...\n")

    # Download each zip separately using osfclient fetch
    for fname in ["val.zip", "test.zip"]:
        out = dest / fname
        if out.exists():
            print(f"  Already have {fname}")
            continue
        print(f"  Downloading {fname}...")
        result = subprocess.run(
            [sys.executable, "-m", "osfclient", "-p", "xf3ka", "fetch",
             f"osfstorage/{fname}", str(out)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            continue
        print(f"  Downloaded {fname} ({out.stat().st_size / 1024 / 1024:.0f} MB)")

    # Extract zips
    for fname in ["val.zip", "test.zip"]:
        zpath = dest / fname
        if not zpath.exists():
            continue
        extract_dir = dest / fname.replace(".zip", "")
        if extract_dir.exists():
            print(f"  Already extracted {fname}")
            continue
        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)
        print(f"  Extracted {fname}")

    # Skip train.zip for now (it's split into train.zip + train.z01, ~3GB)
    # val + test give us enough data to fine-tune
    print("  Note: Skipping train.zip (3GB+). val+test should suffice for fine-tuning.")
    print("  To get train data too, run:")
    print(f"    python3 -m osfclient -p xf3ka fetch osfstorage/train.zip {dest}/train.zip")
    print(f"    python3 -m osfclient -p xf3ka fetch osfstorage/train.z01 {dest}/train.z01")


def download_samryan18():
    """Download samryan18/chess-dataset from GitHub."""
    dest = BASE_DIR / "samryan18"
    if dest.exists() and any(dest.rglob("*.jpeg")) or (dest.exists() and any(dest.rglob("*.jpg"))):
        print(f"  Already have samryan18 dataset")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print("  Cloning samryan18/chess-dataset...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/samryan18/chess-dataset.git", str(dest / "repo")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return

    # Check what we got
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        count = sum(1 for _ in (dest / "repo").rglob(ext))
        if count > 0:
            print(f"  Found {count} {ext} files")


def main():
    parser = argparse.ArgumentParser(description="Download chess recognition datasets")
    parser.add_argument("--chesscog", action="store_true", help="Download chesscog dataset")
    parser.add_argument("--samryan18", action="store_true", help="Download samryan18 dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if not args.chesscog and not args.samryan18 and not args.all:
        args.all = True

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Chess Recognition Dataset Downloader ===\n")

    if args.chesscog or args.all:
        print("[chesscog] Synthetic rendered chess boards from OSF")
        download_chesscog()
        print()

    if args.samryan18 or args.all:
        print("[samryan18] Real chess board photos from GitHub")
        download_samryan18()
        print()

    # Summary
    print("=== Summary ===")
    for name in ["chesscog_render", "samryan18"]:
        d = BASE_DIR / name
        if d.exists():
            imgs = sum(1 for _ in d.rglob("*.png")) + sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.jpeg"))
            jsons = sum(1 for _ in d.rglob("*.json"))
            print(f"  {name}: {imgs} images, {jsons} JSON files")
        else:
            print(f"  {name}: not downloaded")

    print("\nNext step: python3 training/prepare_squares.py")


if __name__ == "__main__":
    main()
