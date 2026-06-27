# training/colab_train.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Single-file training script designed to run on Google Colab (FREE GPU).
#
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════

INSTALL_CMD = """
pip install -q mediapipe kaggle opencv-python-headless tqdm
"""

# In Colab, run:  !pip install -q mediapipe kaggle opencv-python-headless tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# ═══════════════════════════════════════════════════════════════════════════════

MOUNT_CODE = """
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/sign_language_project'
import os; os.makedirs(DRIVE_DIR + '/models', exist_ok=True)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Upload kaggle.json and download dataset
# ═══════════════════════════════════════════════════════════════════════════════

KAGGLE_CODE = """
from google.colab import files
uploaded = files.upload()          # upload your kaggle.json here

import os, shutil
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Download ASL Alphabet dataset (~1 GB)
os.system('kaggle datasets download -d grassknoted/asl-alphabet --unzip -p /content/data/')

# Verify
classes = os.listdir('/content/data/asl_alphabet_train')
print(f'Classes found: {sorted(classes)}')
print(f'Total classes: {len(classes)}')
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Preprocess: crop hand ROI with MediaPipe, resize 64×64 grayscale
# ═══════════════════════════════════════════════════════════════════════════════

PREPROCESS_CODE = """
import cv2, mediapipe as mp, os, numpy as np
from tqdm import tqdm

RAW_DIR  = '/content/data/asl_alphabet_train'
PROC_DIR = '/content/data/processed'
IMG_SIZE = 64
LIMIT    = 1000   # images per class — increase for better accuracy

mp_hands = mp.solutions.hands

def crop_hand(img_bgr, padding=0.20):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands:
        res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lms = res.multi_hand_landmarks[0].landmark
    xs  = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
    x1  = max(0, int(min(xs)*w - (max(xs)-min(xs))*w*padding))
    y1  = max(0, int(min(ys)*h - (max(ys)-min(ys))*h*padding))
    x2  = min(w, int(max(xs)*w + (max(xs)-min(xs))*w*padding))
    y2  = min(h, int(max(ys)*h + (max(ys)-min(ys))*h*padding))
    if x2-x1 < 10 or y2-y1 < 10: return None
    return img_bgr[y1:y2, x1:x2]

classes = sorted(os.listdir(RAW_DIR))
total   = 0
for cls in classes:
    src = os.path.join(RAW_DIR, cls)
    dst = os.path.join(PROC_DIR, cls)
    if not os.path.isdir(src): continue
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src)
                   if f.lower().endswith(('.jpg','.jpeg','.png')))[:LIMIT]
    for fname in tqdm(files, desc=cls, ncols=60, leave=False):
        img = cv2.imread(os.path.join(src, fname))
        if img is None: continue
        roi = crop_hand(img)
        if roi is None:
            h,w = img.shape[:2]; m=int(min(h,w)*0.1)
            roi = img[m:h-m, m:w-m]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(dst, fname), resized)
        total += 1
    print(f'  {cls}: {len(files)} images')
print(f'\\nTotal processed: {total}')
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Train CNN
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_CODE = """
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR   = '/content/data/processed'
MODEL_PATH = '/content/drive/MyDrive/sign_language_project/models/sign_language_cnn.keras'
LMAP_PATH  = '/content/drive/MyDrive/sign_language_project/models/label_map.json'
IMG_SIZE   = 64
BATCH      = 64
EPOCHS     = 50

# Data generators
train_gen = ImageDataGenerator(
    rescale=1/255., validation_split=0.15,
    horizontal_flip=True, rotation_range=10,
    zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
).flow_from_directory(DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH,
    class_mode='categorical', subset='training', seed=42)

val_gen = ImageDataGenerator(
    rescale=1/255., validation_split=0.15
).flow_from_directory(DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH,
    class_mode='categorical', subset='validation', seed=42)

NUM_CLASSES = len(train_gen.class_indices)
print(f'Classes: {NUM_CLASSES}  |  Train: {train_gen.samples}  |  Val: {val_gen.samples}')

# Save label map
label_map = {str(v): k for k,v in train_gen.class_indices.items()}
with open(LMAP_PATH, 'w') as f: json.dump(label_map, f, indent=2)

# Build CNN
inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x   = layers.Conv2D(32,(3,3),padding='same')(inp)
x   = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
x   = layers.Conv2D(32,(3,3),padding='same')(x)
x   = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
x   = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

x   = layers.Conv2D(64,(3,3),padding='same')(x)
x   = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
x   = layers.Conv2D(64,(3,3),padding='same')(x)
x   = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
x   = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

x   = layers.Conv2D(128,(3,3),padding='same')(x)
x   = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
x   = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

x   = layers.Flatten()(x)
x   = layers.Dense(256)(x); x = layers.BatchNormalization()(x)
x   = layers.Activation('relu')(x); x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inp, out, name='SignLanguageCNN')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
cbs = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=7,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                              save_best_only=True, verbose=1),
]
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=cbs)
print(f'\\nBest val accuracy: {max(history.history[\"val_accuracy\"]):.4f}')
print(f'Model saved → {MODEL_PATH}')
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Plot training curves
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_CODE = """
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train', lw=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   lw=2)
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', lw=2)
axes[1].plot(history.history['val_loss'], label='Val',   lw=2)
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/sign_language_project/training_plots.png', dpi=150)
plt.show()
print('Plot saved to Drive.')
"""

# ─────────────────────────────────────────────────────────────────────────────
# When this file is run as a standalone script (not in Colab), it prints
# step-by-step instructions so the user knows what to copy into Colab cells.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""

        GOOGLE COLAB TRAINING GUIDE -
  1. Go to: https://colab.research.google.com                               
  2. Runtime → Change runtime type → GPU (T4 free tier)                     
  3. Create a new notebook, then paste each CELL below in order.            

""")

    cells = [
        ("CELL 1 — Install packages",     INSTALL_CMD),
        ("CELL 2 — Mount Google Drive",   MOUNT_CODE),
        ("CELL 3 — Download dataset",     KAGGLE_CODE),
        ("CELL 4 — Preprocess images",    PREPROCESS_CODE),
        ("CELL 5 — Train CNN",            TRAIN_CODE),
        ("CELL 6 — Plot curves",          PLOT_CODE),
    ]

    for title, code in cells:
        print(f"\n{'─'*78}")
        print(f"  {title}")
        print(f"{'─'*78}")
        print(code)
