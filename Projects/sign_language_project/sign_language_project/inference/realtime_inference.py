# USAGE
#   python inference/realtime_inference.py
#   python inference/realtime_inference.py --no-tts
#   python inference/realtime_inference.py --camera 1
#   python inference/realtime_inference.py --confidence 0.70

import os
import sys
import json
import argparse
import time
import collections

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.join(BASE_DIR, "..")
MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "sign_language_cnn.keras")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "models", "label_map.json")

# ── Tunable constants ─────────────────────────────────────────────────────────
IMG_SIZE         = 64
STABILITY_FRAMES = 15    # frames with same prediction → confirm (was 20)
VOTE_WINDOW      = 8    # rolling window for majority vote
CONFIDENCE_DEF   = 0.75 # default minimum confidence (was 0.75)
COOLDOWN_FRAMES  = 10   # frames between confirmations (was 25)
PADDING_SIZES    = [0.20]  # single crop — keeps FPS high
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 210, 90)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)
DARK   = (30, 30, 30)
CYAN   = (255, 210, 0)
GRAY   = (160, 160, 160)

def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    norm    = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def get_hand_crop(frame_bgr, hand_landmarks, padding):
    """
    Crop hand ROI from frame using MediaPipe landmarks + given padding.
    Returns (roi_bgr, (x1,y1,x2,y2)) or (None, None).
    """
    h, w  = frame_bgr.shape[:2]
    lms   = hand_landmarks.landmark
    xs    = [lm.x for lm in lms]
    ys    = [lm.y for lm in lms]
    pad_x = (max(xs) - min(xs)) * padding + 0.02
    pad_y = (max(ys) - min(ys)) * padding + 0.02
    x1    = max(0, int((min(xs) - pad_x) * w))
    y1    = max(0, int((min(ys) - pad_y) * h))
    x2    = min(w, int((max(xs) + pad_x) * w))
    y2    = min(h, int((max(ys) + pad_y) * h))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_multiscale(model, frame_bgr, hand_landmarks, label_map):
    """
    Run prediction at multiple crop padding sizes.
    Return (best_label, best_confidence, bbox, top3) where best is the
    crop+padding that produced the highest confidence score.
    """
    best_label = None
    best_conf  = 0.0
    best_bbox  = None
    best_top3  = []

    for pad in PADDING_SIZES:
        roi, bbox = get_hand_crop(frame_bgr, hand_landmarks, pad)
        if roi is None:
            continue
        inp   = preprocess_roi(roi)
        probs = model.predict(inp, verbose=0)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        conf  = float(probs[top3_idx[0]])
        if conf > best_conf:
            best_conf  = conf
            best_label = label_map[int(top3_idx[0])]
            best_bbox  = bbox
            best_top3  = [(label_map[int(i)], float(probs[i])) for i in top3_idx]

    return best_label, best_conf, best_bbox, best_top3


def draw_ui(frame, current_word, sentence, prediction_buffer,
            current_label, current_conf, top3, fps, stability_frames,
            confidence_threshold):
    """Draw all UI elements on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (8, 24), FONT, 0.55, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, "Sign Language Recognition  |  hold steady until bar fills",
                (80, 24), FONT, 0.48, WHITE, 1, cv2.LINE_AA)

    # ── Top-3 live predictions (top-right) ───────────────────────────────────
    for i, (lbl, conf) in enumerate(top3):
        colour = GREEN if i == 0 and conf >= confidence_threshold else GRAY
        text   = f"{lbl}  {conf:.0%}"
        cv2.putText(frame, text,
                    (w - 110, 70 + i * 26),
                    FONT, 0.7, colour, 2, cv2.LINE_AA)

    # ── Stability progress bar ────────────────────────────────────────────────
    panel_y = h - 130
    if current_label:
        same   = sum(1 for p in prediction_buffer if p == current_label)
        bar_w  = int((same / stability_frames) * (w - 24))
        cv2.rectangle(frame, (12, panel_y - 12),
                      (12 + bar_w, panel_y - 5), GREEN, -1)
        cv2.rectangle(frame, (12, panel_y - 12),
                      (w - 12, panel_y - 5), (80, 80, 80), 1)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), DARK, -1)
    cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, f"Word : {current_word}|",
                (12, panel_y + 28), FONT, 0.75, CYAN, 2, cv2.LINE_AA)

    full = sentence + current_word
    if len(full) > 46:
        full = "..." + full[-43:]
    cv2.putText(frame, f"Sent : {full}",
                (12, panel_y + 58), FONT, 0.62, WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame,
                "SPACE:word  BACK:delete  ENTER:speak  C:clear  Q:quit",
                (12, panel_y + 90), FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    return frame


def init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 0.9)
        return engine
    except Exception as e:
        print(f"[WARNING] TTS unavailable: {e}")
        return None


def speak(engine, text):
    if engine and text.strip():
        engine.say(text.strip())
        engine.runAndWait()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(camera_id=0, confidence_threshold=CONFIDENCE_DEF, use_tts=True):

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[INFO] Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    print(f"[INFO] {len(label_map)} classes loaded.")

    tts = init_tts() if use_tts else None

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.45,
        model_complexity=0,
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    print("[INFO] Webcam open. Press Q to quit.\n")

    # ── State ─────────────────────────────────────────────────────────────────
    sentence          = ""
    current_word      = ""
    vote_buffer       = collections.deque(maxlen=VOTE_WINDOW)
    prediction_buffer = collections.deque(maxlen=STABILITY_FRAMES)
    last_confirmed    = None
    confirm_cooldown  = 0

    fps_timer   = time.time()
    frame_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_timer   = time.time()

        # MediaPipe
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_label = None
        current_conf  = 0.0
        top3          = []

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Multi-scale prediction
            label, conf, bbox, top3 = predict_multiscale(
                model, frame, hand_lms, label_map
            )

            if conf >= confidence_threshold:
                current_label = label
                current_conf  = conf

            # Draw bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                colour = GREEN if conf >= confidence_threshold else ORANGE
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame,
                            f"{label}  {conf:.0%}" if label else f"? {conf:.0%}",
                            (x1, max(y1 - 10, 20)),
                            FONT, 0.85, colour, 2, cv2.LINE_AA)

            # Show ROI preview (small window — see exactly what CNN sees)
            # Show ROI preview (small window — see exactly what CNN sees)
            best_roi, _ = get_hand_crop(frame, hand_lms, 0.20)
            if best_roi is not None:
                gray_eq = cv2.cvtColor(best_roi, cv2.COLOR_BGR2GRAY)
                preview = cv2.resize(gray_eq, (128, 128),
                                     interpolation=cv2.INTER_NEAREST)
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                cv2.putText(preview_bgr, "CNN sees",
                            (4, 16), FONT, 0.4, (0, 255, 120), 1)
                # Paste preview into top-left of frame
                frame[42:170, 4:132] = preview_bgr
                cv2.rectangle(frame, (4, 42), (132, 170), (80, 80, 80), 1)

        # ── Voting ────────────────────────────────────────────────────────────
        vote_buffer.append(current_label)
        # Majority vote over last VOTE_WINDOW frames
        if vote_buffer:
            counts  = collections.Counter(
                [v for v in vote_buffer if v is not None]
            )
            voted   = counts.most_common(1)[0][0] if counts else None
        else:
            voted = None

        # Stability check on voted label
        prediction_buffer.append(voted)

        if confirm_cooldown > 0:
            confirm_cooldown -= 1

        if (
            len(prediction_buffer) == STABILITY_FRAMES
            and len(set(prediction_buffer)) == 1
            and voted is not None
            and voted != "del"
            and confirm_cooldown == 0
        ):
            if voted != last_confirmed:
                current_word    += voted
                last_confirmed   = voted
                confirm_cooldown = COOLDOWN_FRAMES
                if use_tts and tts:
                    speak(tts, voted)

        # Handle "del" sign — backspace
        if (
            voted == "del"
            and len(set(list(prediction_buffer)[-3:])) == 1
            and confirm_cooldown == 0
        ):
            if current_word:
                current_word = current_word[:-1]
            confirm_cooldown = COOLDOWN_FRAMES

        # ── Draw UI ───────────────────────────────────────────────────────────
        frame = draw_ui(
            frame, current_word, sentence,
            prediction_buffer, voted, current_conf,
            top3, fps_display, STABILITY_FRAMES,
            confidence_threshold
        )

        cv2.imshow("Sign Language Recognition", frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == 32:                      # SPACE — finish word
            if current_word:
                sentence    += current_word + " "
                current_word = ""
                last_confirmed = None
                prediction_buffer.clear()
                vote_buffer.clear()
        elif key == 8:                       # BACKSPACE
            if current_word:
                current_word = current_word[:-1]
        elif key == 13:                      # ENTER — speak
            full = (sentence + current_word).strip()
            if full:
                print(f"[SPEAK] {full}")
                speak(tts, full)
        elif key == ord("c"):                # C — clear
            sentence = ""
            current_word = ""
            last_confirmed = None
            prediction_buffer.clear()
            vote_buffer.clear()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if sentence or current_word:
        print(f"\n[RESULT] {(sentence + current_word).strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_DEF)
    parser.add_argument("--no-tts",     action="store_true")
    args = parser.parse_args()
    run(
        camera_id=args.camera,
        confidence_threshold=args.confidence,
        use_tts=not args.no_tts,
    )