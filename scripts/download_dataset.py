"""Download the NASA C-MAPSS Turbofan Engine Degradation dataset.

Handles nested zip structures and Windows encoding issues.
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData"

EXPECTED_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]


def extract_recursively(zip_path, output_dir, temp_dir):
    """Extract zip, handling nested zips and directories."""
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        names = zf.namelist()
        print(f"  Zip contains {len(names)} entries:")
        for n in names:
            print(f"    {n}")
        
        # Extract everything to temp first
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(str(extract_dir))
    
    # Look for nested zips and extract them too
    for root, dirs, files in os.walk(str(extract_dir)):
        for f in files:
            fpath = Path(root) / f
            if f.lower().endswith('.zip'):
                print(f"  Found nested zip: {f}")
                try:
                    with zipfile.ZipFile(str(fpath), 'r') as inner_zf:
                        inner_names = inner_zf.namelist()
                        print(f"    Inner zip contains {len(inner_names)} entries")
                        inner_zf.extractall(str(extract_dir / "inner"))
                except Exception as e:
                    print(f"    Failed to extract inner zip: {e}")
    
    # Now find all expected files anywhere in the extracted tree
    found = 0
    for root, dirs, files in os.walk(str(extract_dir)):
        for f in files:
            if f in EXPECTED_FILES:
                src = Path(root) / f
                dst = output_dir / f
                shutil.copy2(str(src), str(dst))
                print(f"  Copied: {f} ({src.stat().st_size / 1024:.0f} KB)")
                found += 1
    
    print(f"  Found {found}/{len(EXPECTED_FILES)} expected files")
    return found > 0


def download_from_nasa(output_dir, temp_dir):
    """Try downloading the full zip from NASA S3."""
    zip_path = temp_dir / "cmapss.zip"
    
    print(f"Downloading from NASA S3...")
    print(f"URL: {DATASET_URL}")
    
    try:
        urllib.request.urlretrieve(DATASET_URL, str(zip_path))
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"Download complete: {size_mb:.1f} MB")
    except Exception as e:
        print(f"NASA S3 download failed: {e}")
        return False
    
    print("Extracting...")
    try:
        return extract_recursively(zip_path, output_dir, temp_dir)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def download_from_github(output_dir):
    """Fallback: download individual files from GitHub mirror."""
    print("Trying GitHub mirror (file-by-file)...")
    
    success_count = 0
    for filename in EXPECTED_FILES:
        url = f"{GITHUB_RAW_BASE}/{filename}"
        target_path = output_dir / filename
        
        try:
            sys.stdout.write(f"  Downloading {filename}... ")
            sys.stdout.flush()
            urllib.request.urlretrieve(url, str(target_path))
            size_kb = target_path.stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success_count += 1
        except Exception as e:
            print(f"FAILED ({e})")
    
    return success_count == len(EXPECTED_FILES)


def verify_download(output_dir):
    """Verify all expected files are present and non-empty."""
    print("\nVerification:")
    all_ok = True
    
    for filename in EXPECTED_FILES:
        filepath = output_dir / filename
        if filepath.exists() and filepath.stat().st_size > 0:
            size_kb = filepath.stat().st_size / 1024
            print(f"  [OK] {filename} ({size_kb:.0f} KB)")
        else:
            print(f"  [MISSING] {filename}")
            all_ok = False
    
    return all_ok


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw"
    temp_dir = project_root / "data" / "temp"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing = [f for f in EXPECTED_FILES if (output_dir / f).exists()]
    if len(existing) == len(EXPECTED_FILES):
        print("All C-MAPSS files already present in data/raw/")
        verify_download(output_dir)
        return
    
    print("=" * 60)
    print("NASA C-MAPSS Dataset Downloader")
    print("=" * 60)
    print(f"Output: {output_dir.resolve()}")
    print()
    
    # Try NASA S3 first
    success = download_from_nasa(output_dir, temp_dir)
    
    # Check what we got
    if not verify_download(output_dir):
        print("\nNASA zip did not contain all files. Trying GitHub mirror...")
        success = download_from_github(output_dir)
    
    # Clean up temp
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)
    
    # Final verification
    if verify_download(output_dir):
        print("\nDataset download complete!")
        print(f"Location: {output_dir.resolve()}")
    else:
        print("\nSome files are missing. Please download manually from:")
        print(f"  {DATASET_URL}")
        print(f"  Or Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        sys.exit(1)


if __name__ == "__main__":
    main()
