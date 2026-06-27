# training/train_cnn.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Train a Convolutional Neural Network (CNN) on preprocessed 64×64 grayscale
#   hand gesture images using TensorFlow / Keras.
#
# MODEL ARCHITECTURE  (designed for CPU-only, 16 GB RAM)
#   Input  : (64, 64, 1)   — grayscale
#   Block 1: Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
#   Block 2: Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
#   Block 3: Conv2D(128)→ BN → ReLU → MaxPool → Dropout(0.25)
#   Flatten → Dense(256) → BN → ReLU → Dropout(0.5)
#   Output : Dense(num_classes, softmax)
#
# TRAINING TRICKS USED
#   • ImageDataGenerator with augmentation (flip, rotation, zoom, shift)
#     to improve generalisation on real webcam frames
#   • EarlyStopping — stops if val_accuracy doesn't improve for 5 epochs
#   • ReduceLROnPlateau — halves learning rate if plateau detected
#   • ModelCheckpoint — saves the best model automatically
#
# USAGE
#   python training/train_cnn.py
#   python training/train_cnn.py --epochs 30 --batch_size 64
#
# OUTPUT
#   models/sign_language_cnn.keras   — best saved model
#   models/label_map.json            — {index: class_label} mapping
#   training/training_plots.png      — accuracy & loss curves
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / Colab
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.join(BASE_DIR, "..")
DATA_DIR      = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "sign_language_cnn.keras")
LABEL_MAP_PATH= os.path.join(MODELS_DIR, "label_map.json")
PLOT_PATH     = os.path.join(BASE_DIR,   "training_plots.png")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = 64
BATCH_SIZE  = 32
EPOCHS      = 50
VAL_SPLIT   = 0.15  # 15% of data used for validation
SEED        = 42

# ── Limit TF thread count for CPU training (avoids thermal throttling) ────────
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ── Model definition ──────────────────────────────────────────────────────────

def build_cnn(num_classes: int) -> tf.keras.Model:
    """
    Build and return the CNN model.

    Architecture notes:
    - Three convolutional blocks, each doubling filters: 32 → 64 → 128
    - BatchNormalization after each Conv layer stabilises training
    - MaxPooling halves spatial dims each block: 64 → 32 → 16 → 8
    - Dropout prevents overfitting on the relatively small dataset
    - Final Dense(256) learns to combine spatial features
    """
    model = models.Sequential(name="SignLanguageCNN")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Classifier head ──────────────────────────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def get_data_generators(data_dir: str, batch_size: int):
    """
    Create train and validation ImageDataGenerators from data_dir.
    Uses flow_from_directory so class names are read from subfolder names.

    Augmentation applied to TRAINING only:
      - horizontal flip (mirror gestures vary slightly in real use)
      - ±10° rotation
      - ±10% zoom
      - ±10% width/height shift
    """
    # Train augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VAL_SPLIT,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        fill_mode="nearest",
    )

    # Validation: only rescale, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VAL_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


# ── Training callbacks ────────────────────────────────────────────────────────

def get_callbacks() -> list:
    """
    Return a list of Keras callbacks:
    - EarlyStopping      : Stop if val_accuracy stagnates for 7 epochs
    - ReduceLROnPlateau  : Halve learning rate if val_loss plateaus
    - ModelCheckpoint    : Save best model weights automatically
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [early_stop, reduce_lr, checkpoint]


# ── Evaluation & plots ────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str):
    """Save accuracy and loss curves as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history["accuracy"],     label="Train accuracy", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy",   lw=2)
    axes[0].set_title("Model Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss plot
    axes[1].plot(history.history["loss"],     label="Train loss", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val loss",   lw=2)
    axes[1].set_title("Model Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training plots saved → {save_path}")


def evaluate_model(model, val_gen, class_names: list[str]):
    """Print classification report and save confusion matrix."""
    print("\n[INFO] Running evaluation on validation set …")

    # Collect all predictions
    val_gen.reset()
    y_true = []
    y_pred = []

    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        preds  = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds,   axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.4,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()

    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(epochs: int, batch_size: int):
    # ── Verify dataset ────────────────────────────────────────────────────────
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Processed data not found at: {DATA_DIR}")
        print("Run dataset_prep/preprocess_dataset.py first.")
        sys.exit(1)

    # ── Data generators ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading data from: {DATA_DIR}")
    train_gen, val_gen = get_data_generators(DATA_DIR, batch_size)

    class_names = list(train_gen.class_indices.keys())  # e.g. ['A','B','C',...]
    num_classes  = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Train samples: {train_gen.samples}  |  Val samples: {val_gen.samples}")

    # ── Save label map ────────────────────────────────────────────────────────
    # label_map[index] = class_name   — used by inference script
    label_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] Label map saved → {LABEL_MAP_PATH}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_cnn(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Starting training: {epochs} epochs, batch size {batch_size}")
    print("[INFO] Model will be auto-saved to:", MODEL_PATH)
    print("[INFO] Training on CPU — this may take 1 hour or more for 26 classes.\n"
          "       Tip: use Google Colab with GPU to finish in ~10 min.\n")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # ── Save final model (in case checkpoint didn't fire last) ────────────────
    model.save(MODEL_PATH)
    print(f"\n[INFO] Final model saved → {MODEL_PATH}")

    # ── Plots & evaluation ────────────────────────────────────────────────────
    plot_training_history(history, PLOT_PATH)
    evaluate_model(model, val_gen, class_names)

    print("\n[DONE] Training complete.")
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Label map  : {LABEL_MAP_PATH}")
    print(f"  Plots      : {PLOT_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for real-time sign language recognition."
    )
    parser.add_argument("--epochs",     type=int, default=EPOCHS,     help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
