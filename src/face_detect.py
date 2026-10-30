"""Face detection using OpenCV's Haar cascade classifier.

Haar cascades ship with OpenCV, need no extra model download, and are
fast enough to run on every sampled frame of a video.
"""

import cv2

from src.features import FACE_SIZE

_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_cascade = None  # loaded lazily, once


def load_face_cascade():
    """Load (and cache) OpenCV's built-in frontal-face Haar cascade."""
    global _cascade
    if _cascade is None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This OpenCV build has no Haar cascades (OpenCV 5 removed "
                "them). Install a 4.x build instead:  pip install "
                "'opencv-python>=4.8,<5'"
            )
        cascade_path = cv2.data.haarcascades + _CASCADE_FILENAME
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {cascade_path}. "
                "Check that opencv-python is installed correctly."
            )
        _cascade = cascade
    return _cascade


def detect_faces(frame):
    """Detect faces in one BGR frame.

    Returns:
        A list of ``(x, y, w, h)`` bounding boxes, largest first.
        The list is empty when no face is found.
    """
    if frame is None or frame.size == 0:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = load_face_cascade()
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(boxes) == 0:
        return []
    # Largest face first: in most videos the main subject is the target.
    return sorted(
        [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes],
        key=lambda box: box[2] * box[3],
        reverse=True,
    )


def crop_face(frame, box):
    """Crop one bounding box from a frame and resize it to FACE_SIZE.

    Returns ``None`` if the box falls outside the frame.
    """
    x, y, w, h = box
    height, width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, FACE_SIZE)


def extract_faces(frame):
    """Return all detected faces in a frame, cropped and resized.

    Faces are returned largest first. The list is empty when no face
    is detected.
    """
    faces = []
    for box in detect_faces(frame):
        face = crop_face(frame, box)
        if face is not None:
            faces.append(face)
    return faces


def extract_largest_face(frame):
    """Return the largest detected face (resized to FACE_SIZE), or None."""
    faces = extract_faces(frame)
    return faces[0] if faces else None
