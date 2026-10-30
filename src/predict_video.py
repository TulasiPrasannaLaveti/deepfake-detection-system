"""Predict whether a video is real or a deepfake.

Usage (from the project root):

    python -m src.predict_video <video_path>

The video is sampled frame by frame. In each sampled frame the
largest face is detected, its features are extracted, and the trained
classifier labels that face Real or Fake. The per-frame labels are
then aggregated into one video verdict by majority vote: if at least
half of the analysed frames are classified Fake, the video verdict
is FAKE, otherwise REAL. The percentage of fake frames is always
printed, because a video near 50% is far less certain than one at
5% or 95%.
"""

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_frames import extract_frames  # noqa: E402
from src.face_detect import extract_largest_face  # noqa: E402
from src.features import extract_features  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
FAKE_LABEL = 1


def load_model(model_path):
    """Load the saved model bundle written by src.train."""
    with open(model_path, "rb") as file:
        bundle = pickle.load(file)
    # Accept both the current dict format and a bare saved pipeline.
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


def predict_video(video_path, model, frame_interval=10, max_frames=None):
    """Classify a video.

    Returns a dict with:
        verdict: "FAKE", "REAL", or "UNKNOWN" (no faces found)
        fake_frames / real_frames / total_frames_analysed: counts
        fake_percentage: percentage of analysed frames labelled Fake
        mean_fake_probability: average predicted probability of Fake,
            or None if the model does not provide probabilities
    """
    frames = extract_frames(
        video_path, frame_interval=frame_interval, max_frames=max_frames
    )

    fake_frames = 0
    real_frames = 0
    fake_probabilities = []

    for frame in frames:
        face = extract_largest_face(frame)
        if face is None:
            continue  # no face in this frame -> nothing to classify
        features = extract_features(face).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        if prediction == FAKE_LABEL:
            fake_frames += 1
        else:
            real_frames += 1
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            if FAKE_LABEL in classes:
                fake_probabilities.append(float(probabilities[classes.index(FAKE_LABEL)]))

    analysed = fake_frames + real_frames
    if analysed == 0:
        return {
            "verdict": "UNKNOWN",
            "fake_frames": 0,
            "real_frames": 0,
            "total_frames_analysed": 0,
            "fake_percentage": 0.0,
            "mean_fake_probability": None,
        }

    fake_percentage = 100.0 * fake_frames / analysed
    return {
        "verdict": "FAKE" if fake_percentage >= 50.0 else "REAL",
        "fake_frames": fake_frames,
        "real_frames": real_frames,
        "total_frames_analysed": analysed,
        "fake_percentage": fake_percentage,
        "mean_fake_probability": (
            sum(fake_probabilities) / len(fake_probabilities)
            if fake_probabilities
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a video as REAL or FAKE (deepfake)."
    )
    parser.add_argument("video_path", help="Path to the video file to check")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Analyse 1 frame out of every N frames (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of sampled frames to read",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(
            f"No trained model found at {model_path}.\n"
            "Train one first (see the README's dataset instructions):\n"
            "    python -m src.train"
        )
        return 1

    if not Path(args.video_path).is_file():
        print(f"Video file not found: {args.video_path}")
        return 1

    model = load_model(model_path)
    try:
        result = predict_video(
            args.video_path,
            model,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
        )
    except RuntimeError as error:
        print(str(error))
        return 1

    print(f"Video: {args.video_path}")
    if result["total_frames_analysed"] == 0:
        print("No faces were detected in any sampled frame, so no verdict is possible.")
        print("Try a lower --frame-interval, or a video with a clearer, front-facing face.")
        return 0

    print(f"Frames analysed (with a face): {result['total_frames_analysed']}")
    print(f"  labelled Real: {result['real_frames']}")
    print(f"  labelled Fake: {result['fake_frames']}")
    print(f"Fake frames: {result['fake_percentage']:.1f}%")
    if result["mean_fake_probability"] is not None:
        print(f"Mean predicted probability of Fake: {result['mean_fake_probability']:.2f}")
    print(f"Verdict: {result['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
