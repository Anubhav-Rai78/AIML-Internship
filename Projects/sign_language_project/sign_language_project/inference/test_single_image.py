# inference/test_single_image.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Quick sanity-check: run the trained CNN on a single image file.
#   Useful to verify the model loads correctly and predicts as expected
#   BEFORE testing with a live webcam.
#
# USAGE
#   python inference/test_single_image.py --image path/to/hand.jpg
#   python inference/test_single_image.py --image path/to/hand.jpg --show
#
# The script will:
#   1. Load the image
#   2. Detect hand via MediaPipe
#   3. Crop and preprocess the hand ROI
#   4. Run CNN prediction
#   5. Print top-5 predictions with confidence scores
#   
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "sign_language_cnn.keras")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "models", "label_map.json")
IMG_SIZE     = 64
PADDING_PCT  = 0.20


def crop_hand_roi(image_bgr, padding=PADDING_PCT):
    """Detect and crop hand from image using MediaPipe."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    lms  = results.multi_hand_landmarks[0].landmark
    xs   = [lm.x for lm in lms]
    ys   = [lm.y for lm in lms]
    pad_x = (max(xs) - min(xs)) * padding
    pad_y = (max(ys) - min(ys)) * padding
    x1 = max(0, int((min(xs) - pad_x) * w))
    y1 = max(0, int((min(ys) - pad_y) * h))
    x2 = min(w, int((max(xs) + pad_x) * w))
    y2 = min(h, int((max(ys) + pad_y) * h))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return image_bgr[y1:y2, x1:x2]


def predict(image_path: str, show: bool = False):
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    print(f"[INFO] Image loaded: {image_path}  shape={img.shape}")

    # Crop hand
    roi = crop_hand_roi(img)
    if roi is None:
        print("[WARNING] No hand detected — using full image centre crop")
        h, w = img.shape[:2]
        m    = int(min(h, w) * 0.1)
        roi  = img[m:h-m, m:w-m]

    # Preprocess
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    inp     = resized.astype(np.float32) / 255.0
    inp     = inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    probs     = model.predict(inp, verbose=0)[0]
    top5_idx  = np.argsort(probs)[::-1][:5]

    print("\n── Top-5 Predictions ─────────────────────────────")
    for rank, idx in enumerate(top5_idx, 1):
        label = label_map.get(idx, str(idx))
        conf  = probs[idx]
        bar   = "█" * int(conf * 30)
        print(f"  {rank}. {label:>8s}  {conf:6.2%}  {bar}")
    print()

    # Optionally display the ROI
    if show:
        display = cv2.resize(roi, (300, 300))
        top_label = label_map.get(int(top5_idx[0]), "?")
        top_conf  = probs[top5_idx[0]]
        cv2.putText(display, f"{top_label}  {top_conf:.0%}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 100), 2)
        cv2.imshow("Prediction", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CNN on a single hand gesture image."
    )
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--show",  action="store_true", help="Display the cropped ROI")
    args = parser.parse_args()
    predict(args.image, show=args.show)
