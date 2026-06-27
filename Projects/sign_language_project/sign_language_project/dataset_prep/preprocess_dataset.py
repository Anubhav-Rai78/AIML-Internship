# dataset_prep/preprocess_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Read raw dataset images (ASL Alphabet from Kaggle).
#   For each image:
#     1. Detect hand using MediaPipe Hands.
#     2. Crop the bounding box of the hand + 20% padding.
#     3. Resize to 64×64 pixels (grayscale).
#     4. Save to data/processed/<LABEL>/<filename>.jpg
#
#   WHY THIS STEP?
#   ─────────────
#   Raw dataset images are 200×200 with a plain background.
#   Real webcam frames have cluttered backgrounds.
#   By using MediaPipe to crop the hand DURING TRAINING, the CNN learns to
#   classify hand shapes extracted the same way the live script will extract
#   them — this improves real-world accuracy.
#
# USAGE
#   python dataset_prep/preprocess_dataset.py
#   python dataset_prep/preprocess_dataset.py --limit 500   # 500 images/class
#   python dataset_prep/preprocess_dataset.py --classes A B C   # only A, B, C
#
# OUTPUT
#   data/processed/
#   ├── A/  (cropped 64×64 grayscale .jpg)
#   ├── B/
#   └── ...
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.join(BASE_DIR, "..")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "asl_alphabet_train")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = 64   # pixels (CNN input: 64×64)
PADDING_PCT = 0.20  # expand bounding box by 20% on each side

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands


def crop_hand_roi(image_bgr: np.ndarray, padding: float = PADDING_PCT):
    """
    Use MediaPipe to detect the hand in image_bgr.
    Returns a cropped BGR image of just the hand region (with padding),
    or None if no hand is detected.
    """
    # MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,  # lower threshold for varied dataset images
    ) as hands:
        results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Get all (x, y) landmark coordinates
    lms = results.multi_hand_landmarks[0].landmark
    xs  = [lm.x for lm in lms]
    ys  = [lm.y for lm in lms]

    # Bounding box in pixel coords
    x_min = int(min(xs) * w)
    x_max = int(max(xs) * w)
    y_min = int(min(ys) * h)
    y_max = int(max(ys) * h)

    # Add padding
    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # Avoid degenerate crops
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return image_bgr[y1:y2, x1:x2]


def process_class(label: str, limit: int | None = None) -> dict:
    """
    Process all images for a single class label.
    Returns stats dict.
    """
    src_dir = os.path.join(RAW_DIR, label)
    dst_dir = os.path.join(PROCESSED_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if limit:
        image_files = image_files[:limit]

    saved   = 0
    skipped = 0

    for fname in tqdm(image_files, desc=f"  {label}", leave=False, ncols=70):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        # Skip if already processed
        if os.path.exists(dst_path):
            saved += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            skipped += 1
            continue

        # Crop hand ROI
        roi = crop_hand_roi(img)

        if roi is None:
            # Fallback: use centre 80% of image if MediaPipe fails
            # (dataset images have plain backgrounds, so this is usually ok)
            h, w = img.shape[:2]
            m = int(min(h, w) * 0.1)
            roi = img[m: h - m, m: w - m]

        # Resize to 64×64 grayscale
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Save
        cv2.imwrite(dst_path, resized)
        saved += 1

    return {"label": label, "saved": saved, "skipped": skipped}


def main(classes: list[str] | None = None, limit: int | None = None):
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] Raw dataset not found at: {RAW_DIR}")
        print("Run dataset_prep/download_dataset.py first.")
        sys.exit(1)

    available = sorted(os.listdir(RAW_DIR))
    available = [c for c in available if os.path.isdir(os.path.join(RAW_DIR, c))]

    if classes:
        # Filter to requested classes only
        available = [c for c in available if c in classes]
        not_found = [c for c in classes if c not in available]
        if not_found:
            print(f"[WARNING] Classes not found in dataset: {not_found}")

    print(f"\n[INFO] Processing {len(available)} classes -> {PROCESSED_DIR}")
    print(f"       Image size: {IMG_SIZE}x{IMG_SIZE} px  |  grayscale")
    if limit:
        print(f"       Limit: {limit} images per class\n")
    else:
        print()

    total_saved = 0
    for label in available:
        stats = process_class(label, limit=limit)
        print(f"  {stats['label']:>8s} -> saved: {stats['saved']:>4d}  "
              f"skipped: {stats['skipped']:>3d}")
        total_saved += stats["saved"]

    print(f"\n[DONE] Total images saved: {total_saved}")
    print(f"       Output directory  : {PROCESSED_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ASL dataset: crop hand ROI, resize to 64×64 grayscale."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per class (default: all). Use 500 for a quick test run.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Only process these classes. E.g.: --classes A B C",
    )
    args = parser.parse_args()
    main(classes=args.classes, limit=args.limit)
