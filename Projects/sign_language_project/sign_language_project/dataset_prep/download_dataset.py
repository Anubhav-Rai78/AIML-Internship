# dataset_prep/download_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Guides the user to download the ASL Alphabet dataset and verifies the
#   folder structure is correct before training.
#
# DATASET USED
#   "ASL Alphabet" by Grassknoted on Kaggle
#   URL : https://www.kaggle.com/datasets/grassknoted/asl-alphabet
#   Size: ~1 GB  |  87,000 images  |  29 classes (A–Z + del, space, nothing)
#   License: Open
#
#   Alternatively, the "ASL Dataset" by Lexset:
#   URL: https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
#
# HOW TO DOWNLOAD (two ways)
#   Option 1 — Kaggle CLI (recommended):
#       pip install kaggle
#       # Put your kaggle.json API key in ~/.kaggle/kaggle.json
#       kaggle datasets download -d grassknoted/asl-alphabet
#       unzip asl-alphabet.zip -d data/
#
#   Option 2 — Manual:
#       Go to https://www.kaggle.com/datasets/grassknoted/asl-alphabet
#       Click Download → unzip into:  sign_language_project/data/asl_alphabet_train/
#
# EXPECTED FOLDER STRUCTURE AFTER EXTRACTION
#   sign_language_project/
#   └── data/
#       └── asl_alphabet_train/
#           ├── A/   (3000 images)
#           ├── B/   (3000 images)
#           ├── ...
#           └── Z/   (3000 images)
#
# Run this script AFTER extraction to verify the structure is correct.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "asl_alphabet_train")

# Classes we actually use: A–Z and digits 0–9
# The Kaggle ASL Alphabet dataset has A–Z (26 classes).
# For digits 0–9 we use the "ASL Digits" dataset (separate download, see below).
# For this script we verify whatever is present.
REQUIRED_ALPHA = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS         = [str(d) for d in range(10)]
EXPECTED_MIN_IMAGES_PER_CLASS = 100  # bare minimum to proceed


def verify_dataset(data_root: str) -> bool:
    """
    Walk through data_root and print a summary of what is found.
    Returns True if the dataset looks usable.
    """
    if not os.path.isdir(data_root):
        print(f"\n[ERROR] Dataset folder not found:\n  {data_root}\n")
        print("Please download the dataset. Instructions:\n")
        print("  1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("  2. Download & unzip so the folder structure is:")
        print("       sign_language_project/data/asl_alphabet_train/A/  ...Z/")
        print("\n  OR run:")
        print("       pip install kaggle")
        print("       kaggle datasets download -d grassknoted/asl-alphabet")
        print("       unzip asl-alphabet.zip -d sign_language_project/data/")
        return False

    print(f"\n[INFO] Scanning dataset at: {data_root}\n")
    classes_found = sorted(os.listdir(data_root))
    total_images  = 0
    issues        = []

    for cls in classes_found:
        cls_path = os.path.join(data_root, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        count = len(imgs)
        total_images += count
        status = "OK" if count >= EXPECTED_MIN_IMAGES_PER_CLASS else "LOW"
        print(f"  {cls:>8s} : {count:>5d} images  [{status}]")
        if count < EXPECTED_MIN_IMAGES_PER_CLASS:
            issues.append(cls)

    print(f"\n  Total images : {total_images}")
    print(f"  Total classes: {len(classes_found)}")

    if issues:
        print(f"\n[WARNING] These classes have fewer than {EXPECTED_MIN_IMAGES_PER_CLASS} images: {issues}")
        print("  Consider downloading more data or merging with another dataset.\n")
    else:
        print("\n[OK] Dataset looks good. You can proceed to training.\n")

    return len(issues) == 0


def download_via_kaggle(dataset_slug: str, dest_dir: str):
    """
    Try to download using the kaggle CLI.
    Requires kaggle package and ~/.kaggle/kaggle.json.
    """
    import subprocess
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Attempting Kaggle download: {dataset_slug}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "--unzip", "-p", dest_dir],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed. Please download manually.")
        return False
    print("[INFO] Download complete.")
    return True


if __name__ == "__main__":
    auto_download = "--download" in sys.argv

    if auto_download:
        dest = os.path.join(os.path.dirname(__file__), "..", "data")
        ok = download_via_kaggle("grassknoted/asl-alphabet", dest)
        if not ok:
            sys.exit(1)

    verify_dataset(DATA_ROOT)
