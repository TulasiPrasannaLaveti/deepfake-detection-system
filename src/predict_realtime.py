"""Real-time deepfake check using a webcam.

Usage (from the project root):

    python -m src.predict_realtime

Opens the default webcam, detects the largest face in each frame,
classifies it with the trained model, and draws the label
(REAL / FAKE + fake probability) on the live video window.
Press "q" to quit.

A trained model is required first (see the README's dataset
instructions):  python -m src.train
The model is a classical ML baseline, so treat the live label as a
demo of the pipeline, not as proof about any real person.
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_detect import detect_faces, crop_face  # noqa: E402
from src.features import extract_features  # noqa: E402
from src.predict_video import DEFAULT_MODEL_PATH, load_model  # noqa: E402

FAKE_LABEL = 1


def classify_face(model, face):
    """Return (label_text, fake_probability_or_None) for one face crop."""
    features = extract_features(face).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        if FAKE_LABEL in classes:
            probability = float(probabilities[classes.index(FAKE_LABEL)])
    return ("FAKE" if prediction == FAKE_LABEL else "REAL"), probability


def main(camera_index=0):
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.is_file():
        print(
            f"No trained model found at {model_path}.\n"
            "Train one first (see the README's dataset instructions):\n"
            "    python -m src.train"
        )
        return 1

    model = load_model(model_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(
            "Could not open a webcam. Check that a camera is connected "
            "and no other app is using it, then try again."
        )
        return 1

    print("Webcam started. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Lost the webcam feed.")
                break

            boxes = detect_faces(frame)
            if boxes:
                x, y, w, h = boxes[0]
                face = crop_face(frame, boxes[0])
                if face is not None:
                    label, probability = classify_face(model, face)
                    colour = (0, 0, 255) if label == "FAKE" else (0, 180, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                    text = label
                    if probability is not None:
                        text += f" (fake prob: {probability:.2f})"
                    cv2.putText(
                        frame, text, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
                    )
            else:
                cv2.putText(
                    frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2,
                )

            cv2.imshow("Deepfake Detection - Real-time (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
