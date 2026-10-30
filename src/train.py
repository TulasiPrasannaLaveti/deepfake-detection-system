"""Train the deepfake classifier on face images.

Dataset layout (images of faces, not full video frames):

    data/real/   -> real face images      (label 0, "Real")
    data/fake/   -> manipulated face images (label 1, "Fake")

Suitable public sources (download them yourself and respect their
licences): Kaggle mirrors of FaceForensics++ and Celeb-DF. See the
README for details.

For each image the pipeline detects the face (Haar cascade); if no
face is detected, the whole image is used as the face crop -- this
fallback works because the dataset images are already face images.
Features are then extracted with ``src.features.extract_features``.

The model is a scikit-learn Pipeline:

    StandardScaler -> LogisticRegression

StandardScaler normalises ("standardises") the feature values, and
Logistic Regression is the classifier. It is simple, fast, and its
output can be explained as a weighted vote of the features -- which
makes it a good honest baseline for this project.

Run from the project root:

    python -m src.train
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make `src.` imports work whether this file is run as a module
# (python -m src.train) or as a script (python src/train.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_detect import extract_largest_face  # noqa: E402
from src.features import extract_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_NAMES = ["Real", "Fake"]  # index = label
MIN_IMAGES_PER_CLASS = 4  # minimum needed for a meaningful stratified split

DATASET_INSTRUCTIONS = """
No usable dataset found. Training needs face images in these folders:

    data/real/    real face images
    data/fake/    manipulated (deepfake) face images

How to get data:
  1. Download a deepfake face dataset, e.g. FaceForensics++ or
     Celeb-DF (both are available via Kaggle mirrors).
  2. Put real face images in data/real/ and fake face images in
     data/fake/ (.jpg / .jpeg / .png / .bmp files).
  3. You need at least {min_images} images in EACH folder; a few
     hundred per class will give a much more meaningful result.
  4. Run training again:  python -m src.train

Nothing was trained and no model file was written.
""".strip()


def _load_class_images(folder, label):
    """Extract feature vectors for every usable image in a folder."""
    vectors, labels, skipped = [], [], 0
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        face = extract_largest_face(image)
        if face is None:
            # Dataset images are already faces, so fall back to the
            # whole image (extract_features resizes it anyway).
            face = image
        vectors.append(extract_features(face))
        labels.append(label)
    return vectors, labels, skipped


def load_dataset(data_dir=DATA_DIR):
    """Load (X, y) from data/real and data/fake.

    Returns ``None`` if the folders are missing, empty, or too small,
    after printing instructions -- training must never crash with a
    traceback just because the dataset has not been downloaded yet.
    """
    real_dir = Path(data_dir) / "real"
    fake_dir = Path(data_dir) / "fake"

    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(DATASET_INSTRUCTIONS.format(min_images=MIN_IMAGES_PER_CLASS))
        return None

    all_vectors, all_labels = [], []
    for label, folder in ((0, real_dir), (1, fake_dir)):
        vectors, labels, skipped = _load_class_images(folder, label)
        print(f"{folder}: {len(vectors)} usable image(s), {skipped} unreadable")
        if len(vectors) < MIN_IMAGES_PER_CLASS:
            print(
                f"\nNot enough images in {folder}: found {len(vectors)}, "
                f"need at least {MIN_IMAGES_PER_CLASS} per class.\n"
            )
            print(DATASET_INSTRUCTIONS.format(min_images=MIN_IMAGES_PER_CLASS))
            return None
        all_vectors.extend(vectors)
        all_labels.extend(labels)

    return np.array(all_vectors), np.array(all_labels)


def train():
    """Train, evaluate and save the model. Returns True on success."""
    loaded = load_dataset()
    if loaded is None:
        return False
    X, y = loaded

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nEvaluation on the held-out test split:")
    print(
        classification_report(
            y_test, predictions, target_names=LABEL_NAMES, zero_division=0
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "label_names": LABEL_NAMES,
                # 0 = Real, 1 = Fake
                "labels": {"real": 0, "fake": 1},
            },
            file,
        )
    print(f"Model saved to {MODEL_PATH}")
    print(f"Trained on {len(X_train)} image(s), tested on {len(X_test)} image(s).")
    return True


if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)
