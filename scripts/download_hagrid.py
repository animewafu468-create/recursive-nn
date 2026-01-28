#!/usr/bin/env python3
"""
Download HaGRID dataset (sample version - 30k images, 384p).

This downloads the smaller sample version (~700MB) which is great for getting started.
For the full dataset (500k+ images, 15GB+), download from Kaggle manually.

Usage:
    python scripts/download_hagrid.py
    python scripts/download_hagrid.py --output ./data/hagrid
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm")
    import requests
    from tqdm import tqdm


# HaGRID sample dataset (30k images, 384p) - smaller and faster to download
HAGRID_SAMPLE_URL = "https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_sample.zip"

# Alternative: Kaggle dataset (requires Kaggle API)
KAGGLE_DATASET = "innominate817/hagrid-sample-30k-384p"


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract zip file with progress."""
    print(f"Extracting to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, dest_dir)


def organize_dataset(raw_dir: Path, output_dir: Path):
    """Organize dataset into train/test structure."""
    print("Organizing dataset structure...")

    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # HaGRID sample comes with train_val and test folders
    # We'll use train_val as train
    for split_name, out_dir in [("train_val", train_dir), ("test", test_dir)]:
        src_dir = raw_dir / split_name
        if not src_dir.exists():
            # Try alternate structure
            src_dir = raw_dir / "hagrid_dataset_sample" / split_name

        if not src_dir.exists():
            print(f"Warning: {src_dir} not found, checking for alternate structure...")
            continue

        # Move gesture folders
        for gesture_dir in src_dir.iterdir():
            if gesture_dir.is_dir():
                dest = out_dir / gesture_dir.name
                dest.mkdir(parents=True, exist_ok=True)

                # Move images
                for img in gesture_dir.glob("*.jpg"):
                    img.rename(dest / img.name)
                for img in gesture_dir.glob("*.png"):
                    img.rename(dest / img.name)

    print("Dataset organized!")


def download_from_kaggle(output_dir: Path):
    """Download using Kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle")
        import kaggle

    print("Downloading from Kaggle...")
    print("Note: Make sure you have ~/.kaggle/kaggle.json set up")

    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(output_dir),
        unzip=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Download HaGRID gesture dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/hagrid",
        help="Output directory",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle API instead of direct download",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("HaGRID Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    if args.kaggle:
        download_from_kaggle(output_dir)
    else:
        # Direct download
        zip_path = output_dir / "hagrid_sample.zip"

        if not zip_path.exists():
            print("Downloading HaGRID sample dataset (~700MB)...")
            print("This contains 30k images at 384p resolution.")
            print()
            download_file(HAGRID_SAMPLE_URL, zip_path, "HaGRID")
        else:
            print(f"Using existing download: {zip_path}")

        # Extract
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        extract_zip(zip_path, raw_dir)

        # Organize
        organize_dataset(raw_dir, output_dir)

        # Cleanup
        if not args.keep_zip:
            print("Cleaning up...")
            zip_path.unlink()
            import shutil
            shutil.rmtree(raw_dir, ignore_errors=True)

    # Verify
    print()
    print("=" * 50)
    print("Download complete!")
    print("=" * 50)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    if train_dir.exists():
        train_count = sum(1 for _ in train_dir.rglob("*.jpg"))
        print(f"Training images: {train_count}")

    if test_dir.exists():
        test_count = sum(1 for _ in test_dir.rglob("*.jpg"))
        print(f"Test images: {test_count}")

    print()
    print("Next steps:")
    print(f"  python scripts/train.py --config-name=hagrid")
    print()


if __name__ == "__main__":
    main()
