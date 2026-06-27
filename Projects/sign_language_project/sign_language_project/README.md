# Real-Time Sign Language Recognition

A computer vision project that recognizes American Sign Language (ASL) hand gestures from a live webcam and converts them into text in real-time. Built using Python, OpenCV, MediaPipe, and TensorFlow.

---

## What this project does

You show an ASL hand sign to your webcam, and the system identifies the letter, builds it into a word, and optionally speaks it out loud using text-to-speech. The pipeline uses MediaPipe to detect and crop the hand from the frame, then passes the cropped image through a CNN trained on 87,000 ASL gesture images to predict the letter.

It supports all 26 letters (A–Z) plus a `del` sign that works like backspace.

---

## How well does it work

The model got around 90.8% accuracy on the validation set during training. In real life though, accuracy is lower — becouse because of domain gap. The training dataset(from kaggle).It works best when you have good front lighting and a plain background behind your hand.

Training was done on Google Colab with a free T4 GPU.

---

## Project structure

```
sign_language_project/
├── data/
│   ├── asl_alphabet_train/    ← raw Kaggle dataset (you download this)
│   └── processed/             ← cropped 64×64 grayscale images (auto-generated)
├── dataset_prep/
│   ├── download_dataset.py    ← verifies dataset structure
│   └── preprocess_dataset.py  ← crops hand ROI and resizes to 64×64
├── training/
│   ├── train_cnn.py           ← trains the CNN locally
│   └── colab_train.py         ← Colab version (recommended, uses free GPU)
├── inference/
│   ├── realtime_inference.py  ← the main webcam script
│   └── test_single_image.py   ← test the model on a single image
├── models/
│   ├── sign_language_cnn.keras ← trained model (download link below)
│   └── label_map.json          ← maps class index to letter name
└── requirements.txt
```

---

## Getting started

First clone the repo and install dependencies:

```bash
git clone https://github.com/AIML-Internship/Projectssign_language_project.git
cd sign_language_project
pip install -r requirements.txt
```

If you're on Windows and pyttsx3 fails to install, also run: 
```bash
pip install pywin32
```

Then download the trained model from Google Drive and place it inside the `models/` folder:

[Download sign_language_cnn.keras](https://drive.google.com/file/d/1HbmG71wcMCbDH_-0VyO05WylL_KQ7106/view?usp=drive_link)

Once the model is in place, run the webcam demo:

```bash
python inference/realtime_inference.py
```

There are a few options you can pass:

```bash
python inference/realtime_inference.py --no-tts          # turn off speech
python inference/realtime_inference.py --camera 1        # use external webcam
python inference/realtime_inference.py --confidence 0.85 # stricter predictions
```

While the webcam window is open, these keys work:
- `SPACE` — finish the current word
- `BACKSPACE` — delete the last letter
- `ENTER` — speak the full sentence
- `C` — clear everything
- `Q` or `ESC` — quit

---

## Tips for better accuracy

- Sit in front of a plain wall or background — a cluttered background confuses the model
- Make sure light is hitting your hand from the front, not behind
- Keep your hand roughly 30–40 cm from the camera so it fills about half the frame
- Hold each sign steady for about half a second until the green progress bar fills up
- Follow the exact ASL shapes — small differences matter, especially for letters like E, M, N, S, T

---

## How it works under the hood

Every frame from the webcam goes through MediaPipe Hands, which detects the hand and gives us 21 landmark points. We use those landmarks to crop a tight bounding box around the hand with 20% padding, convert it to grayscale, and resize it to 64×64 pixels. That crop goes into the CNN, which outputs probabilities for all 28 classes.

To avoid flickering predictions, we use a voting buffer over the last 8 frames and only confirm a letter when the same prediction appears consistently across 15 frames. This means you hold a sign for about half a second and it locks in, rather than random letters firing every frame.

The CNN itself has three convolutional blocks (32, 64, 128 filters) with batch normalization and dropout, followed by a dense layer of 256 units and a softmax output. Total parameters are around 2.2 million. It was trained with Adam, categorical cross-entropy loss, and data augmentation (rotation, zoom, horizontal flip) to improve generalisation.

---

## Training from scratch

If you want to retrain the model yourself instead of using the pre-trained one, here is how:

**Step 1 — Download the dataset**

Get the ASL Alphabet dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

It is about 1 GB with 87,000 images across 29 classes. Extract it to `data/asl_alphabet_train/`.

**Step 2 — Preprocess the images**

```bash
python dataset_prep/preprocess_dataset.py
```

This uses MediaPipe to crop the hand from each image, resizes to 64×64 grayscale, and saves to `data/processed/`. Takes around 30–60 minutes on CPU.

**Step 3 — Train**

Google Colab with a free T4 GPU is recommended — training takes about 15–25 minutes there vs 90+ minutes on CPU:

```bash
python training/colab_train.py
# follow the printed instructions to paste cells into Colab
```

Or train locally:

```bash
python training/train_cnn.py
```

The model saves automatically to `models/sign_language_cnn.keras` whenever validation accuracy improves.

---

## Troubleshooting

If you get a `No module named mediapipe` error, run `pip install mediapipe==0.10.14`.

If you get a Keras deserialization error when loading the model, your TensorFlow version is too old. Upgrade with `pip install tensorflow==2.16.1 --upgrade`.

If the webcam doesn't open, try passing `--camera 1` for an external webcam.

If the model keeps predicting the same letter no matter what sign you show, the issue is usually lighting or background. Try sitting in front of a plain wall with a lamp directly lighting your hand.

---

## Requirements

- Python 3.10
- Webcam
- 4 GB RAM minimum for running the demo
- 8 GB RAM for training
- No GPU needed for inference — runs at ~20–25 fps on a regular CPU
- Google Colab recommended for training (free T4 GPU)