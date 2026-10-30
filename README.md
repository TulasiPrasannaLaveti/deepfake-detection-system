# Deepfake Detection System

A machine learning-based system that classifies a video as **real** or **manipulated (deepfake)** using frame-level facial features — enabling automated detection without manual review.

Built by **Tulasi Prasanna** — [GitHub](https://github.com/TulasiPrasannaLaveti)

---

## Problem

Deepfakes — videos where a person's face is replaced or manipulated using AI — are getting easier to create and harder to spot with the naked eye. Manually checking every video is not practical. This project builds an automated first line of defence: given a video, it analyses the faces in it and gives a real/fake verdict with supporting numbers (what percentage of frames looked fake).

## How It Works

```
Video
  │
  ▼
Frame extraction (OpenCV)        — sample 1 frame in every 10
  │
  ▼
Face detection (Haar cascade)    — find the largest face in the frame
  │
  ▼
Face crop → resize to 128×128 → normalise
  │
  ▼
Feature extraction               — colour histograms, edge density,
                                   blur (Laplacian), FFT frequency energy
  │
  ▼
ML classifier (scikit-learn)     — Logistic Regression, per face/frame
  │
  ▼
Video verdict                    — majority vote across frames
                                   + % of frames classified fake
```

### Features (and why each one)

Every face is converted into a fixed-length vector of **59 numbers**, all of which can be explained — no black box:

| Feature | What it measures | Why it helps |
|---|---|---|
| Colour histograms (48 values) | Distribution of B, G, R colour values | Face-swap generators often leave subtle colour mismatches between the swapped face and the original scene |
| Channel & grayscale mean/std (8 values) | Average brightness and contrast per channel | Manipulated regions often differ slightly in lighting/contrast |
| Edge density (1 value) | Fraction of pixels that are Canny edges | Blending a generated face into a frame smooths or distorts edges, especially near the face boundary |
| Blur — Laplacian variance (1 value) | Image sharpness | Generated/blended faces are frequently slightly blurrier than real camera footage |
| FFT high-frequency energy ratio (1 value) | Share of frequency energy outside the low-frequency centre | Real camera images have natural high-frequency detail (skin texture, sensor noise); generated faces tend to lose it or have unnatural amounts |

### Model

The classifier is a scikit-learn **Pipeline**:

1. **StandardScaler** — normalises the feature values so features with big ranges (like blur) don't dominate features with small ranges (like ratios).
2. **LogisticRegression** — a simple, fast, explainable classifier whose decision is a weighted vote of the features.

This is a feature-based ML approach (not deep learning) by design: it trains quickly on a small dataset, runs on any laptop, and every part of it can be explained in an interview or viva.

## Tech Stack

- **Python**
- **OpenCV** — video reading, frame extraction, face detection (Haar cascade), image processing
- **scikit-learn** — Logistic Regression classifier, train/test split, evaluation metrics
- **NumPy** — numerical work, FFT feature

## Project Structure

```
deepfake-detection-system/
├── README.md
├── requirements.txt
├── data/                    # dataset goes here (not committed to Git)
│   ├── real/                # real face images
│   └── fake/                # deepfake face images
├── models/                  # trained model is saved here (model.pkl, not committed)
├── src/
│   ├── extract_frames.py    # video -> sampled frames
│   ├── face_detect.py       # Haar cascade face detection + cropping
│   ├── features.py          # the 59 explainable features
│   ├── train.py             # train + evaluate + save the model
│   ├── predict_video.py     # CLI: classify one video
│   └── predict_realtime.py  # live webcam demo (REAL/FAKE label overlay)
└── tests/
    └── test_features.py     # sanity tests for feature extraction (no dataset needed)
```

## Dataset

The dataset is **not** included in this repository (the images are large and licence-restricted). To train the model:

1. Download a deepfake face dataset, for example:
   - **FaceForensics++** (also available via Kaggle mirrors)
   - **Celeb-DF** (also available via Kaggle mirrors)
2. Create the folders `data/real/` and `data/fake/` if they don't exist.
3. Put **real face images** in `data/real/` and **manipulated face images** in `data/fake/` (`.jpg`, `.jpeg`, `.png` or `.bmp`).
4. Aim for at least a few hundred images per class for a meaningful result (training refuses to run with fewer than 4 per class and prints instructions instead of crashing).

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins OpenCV to 4.x on purpose — OpenCV 5
> removed the built-in Haar cascades this project uses for face detection.

## Train the Model

```bash
python -m src.train
```

This will:

1. Load images from `data/real/` and `data/fake/`
2. Detect and crop the face in each image (if no face is found, the whole image is used — dataset images are already face images)
3. Extract the 59 features per image
4. Split 80/20 into train/test (stratified, so both classes keep the same proportion)
5. Train the model and print a **classification report** (accuracy, precision, recall, F1)
6. Save the trained model to `models/model.pkl`

If the dataset folders are missing or empty, training prints clear instructions on what to put where, instead of an error traceback.

## Predict on a Video

```bash
python -m src.predict_video path/to/video.mp4
```

Example output format:

```
Video: sample.mp4
Frames analysed (with a face): 42
  labelled Real: 35
  labelled Fake: 7
Fake frames: 16.7%
Verdict: REAL
```

Options: `--frame-interval N` (analyse 1 in every N frames, default 10), `--max-frames N`, `--model PATH`.

A video with **no detectable faces** produces no verdict — the tool tells you this rather than guessing.

## Real-time Webcam Demo

Once a model is trained, you can run the same pipeline live on your webcam — each frame is face-detected, classified, and labelled on screen:

```bash
python -m src.predict_realtime
```

Press `q` to quit. This is a demo of the pipeline running in real time; the underlying model is the same classical ML baseline used for video prediction.

## Evaluation Metrics — Explained

Training prints a classification report. The important numbers:

- **Accuracy** — out of all test images, what fraction were classified correctly? Easy to understand, but misleading if one class is much bigger than the other.
- **Precision** — of the images the model *called* fake, how many actually *were* fake? Low precision = real videos get wrongly flagged.
- **Recall** — of the images that *actually were* fake, how many did the model catch? Low recall = deepfakes slip through.
- **F1-score** — a single number balancing precision and recall (their harmonic mean).

For deepfake detection, **recall matters a lot for the Fake class** (missing a deepfake is costly), but **precision matters too** (wrongly accusing a real video is also serious) — which is why this project reports all of them instead of accuracy alone.

> Note: this README deliberately contains no accuracy claims. The real numbers for your run are whatever `python -m src.train` prints for your dataset — run it and quote those.

## Run the Tests

```bash
python tests/test_features.py
```

This checks (with synthetic random images, no dataset needed) that feature extraction always returns a finite vector of the same fixed length.

## Limitations (Honest Ones)

- This is an **MVP / baseline**, not a state-of-the-art deepfake detector. Hand-crafted features on individual frames cannot capture everything a modern generator produces.
- It analyses **faces frame-by-frame** — it does not use temporal information (blinking patterns, lip-sync, inconsistencies *between* frames), which real-world systems exploit.
- Performance depends heavily on the **dataset**: training on one dataset (e.g. FaceForensics++) and testing on videos from a different source usually gives worse results (the "generalisation gap"). This was a known issue throughout the project: validating on unseen videos is the only honest test.
- Haar cascade face detection works best for **clear, front-facing faces**; profile views, occlusions, and very small faces may be missed.
- Heavily compressed or very low-resolution videos remove exactly the high-frequency detail some features rely on.

## Future Improvements

- Replace (or combine) the hand-crafted features with a **CNN such as Xception or EfficientNet** trained on face crops — the approach used by top FaceForensics++ solutions.
- Add **temporal features**: eye-blink rate, head-pose consistency, and frame-to-frame flicker.
- Add a simple **GUI / web interface** to upload a video and see the verdict with per-frame visualisation.
- Train and cross-test on **multiple datasets** (FaceForensics++ + Celeb-DF) to measure and improve generalisation.
- Try **SVM or Random Forest** as alternative classifiers and compare them against the Logistic Regression baseline.

## Author

**Tulasi Prasanna** (Laveti Tulasi Prasanna)
B.Tech, Malla Reddy University
GitHub: https://github.com/TulasiPrasannaLaveti
