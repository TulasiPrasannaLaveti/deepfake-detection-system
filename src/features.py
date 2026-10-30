"""Explainable, hand-crafted features for face images.

This project deliberately uses classical image features instead of a
deep neural network, so that every input to the classifier can be
explained in plain words:

1. Colour histograms (B, G and R channels)
   Deepfake generators often leave subtle colour inconsistencies
   between a swapped face and the original footage.

2. Edge density (Canny edge detector)
   Blending a generated face into a frame tends to smooth or distort
   edges around the face boundary.

3. Blur / sharpness (variance of the Laplacian)
   Generated or blended faces are frequently slightly blurrier than
   real camera footage.

4. FFT high-frequency energy ratio
   Real camera images contain natural high-frequency detail (skin
   texture, sensor noise). Generated faces often lose some of this
   energy, or contain unnatural amounts of it.

Every face image, whatever its original size, produces a feature
vector of exactly ``FEATURE_VECTOR_LENGTH`` numbers.
"""

import cv2
import numpy as np

# Faces are resized to this size before features are computed, so the
# features are comparable across images of different resolutions.
FACE_SIZE = (128, 128)

HIST_BINS_PER_CHANNEL = 16  # x 3 channels (B, G, R) = 48 histogram values

# Names of every value in the feature vector, in order. Keeping this
# list lets us check the vector length and explain the features.
FEATURE_NAMES = (
    [f"b_hist_{i}" for i in range(HIST_BINS_PER_CHANNEL)]
    + [f"g_hist_{i}" for i in range(HIST_BINS_PER_CHANNEL)]
    + [f"r_hist_{i}" for i in range(HIST_BINS_PER_CHANNEL)]
    + [
        "b_mean",
        "b_std",
        "g_mean",
        "g_std",
        "r_mean",
        "r_std",
        "gray_mean",
        "gray_std",
        "edge_density",
        "laplacian_blur_log_var",
        "fft_high_freq_energy_ratio",
    ]
)

FEATURE_VECTOR_LENGTH = len(FEATURE_NAMES)  # 48 + 8 + 3 = 59


def _colour_histograms(image_bgr):
    """Normalised histogram of each colour channel (48 values)."""
    features = []
    for channel in range(3):  # B, G, R
        hist = cv2.calcHist([image_bgr], [channel], None, [HIST_BINS_PER_CHANNEL], [0, 256])
        hist = hist.flatten().astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.extend(hist.tolist())
    return features


def _channel_statistics(image_bgr, gray):
    """Mean and standard deviation of each channel and of grayscale (8 values)."""
    features = []
    for channel in range(3):  # B, G, R
        plane = image_bgr[:, :, channel].astype(np.float64) / 255.0
        features.append(float(plane.mean()))
        features.append(float(plane.std()))
    gray_float = gray.astype(np.float64) / 255.0
    features.append(float(gray_float.mean()))
    features.append(float(gray_float.std()))
    return features


def _edge_density(gray):
    """Fraction of pixels that are Canny edges (1 value, 0.0-1.0)."""
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def _blur_feature(gray):
    """Sharpness measured by the variance of the Laplacian (1 value).

    The raw variance can be in the thousands, so log1p is used to keep
    the value on a scale the classifier can handle comfortably.
    A higher value means a sharper image; a lower value means blurrier.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.log1p(laplacian.var()))


def _fft_high_frequency_ratio(gray):
    """Share of FFT energy that lies outside the low-frequency centre (1 value, 0.0-1.0)."""
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    energy = np.abs(spectrum) ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0

    height, width = gray.shape
    centre_y, centre_x = height // 2, width // 2
    # The central block (1/4 of the image in each dimension around the
    # centre) is treated as the low-frequency region.
    radius_y, radius_x = height // 8, width // 8
    low_frequency_energy = energy[
        centre_y - radius_y : centre_y + radius_y,
        centre_x - radius_x : centre_x + radius_x,
    ].sum()
    return float((total - low_frequency_energy) / total)


def extract_features(face_image):
    """Return the fixed-length feature vector for one face image.

    Args:
        face_image: A BGR image as a NumPy array (as returned by
            ``cv2.imread`` or a crop from a video frame). Any size is
            accepted; the image is resized to ``FACE_SIZE`` first.

    Returns:
        A 1-D ``numpy.ndarray`` of length ``FEATURE_VECTOR_LENGTH``
        containing only finite floats.

    Raises:
        ValueError: If the image is ``None`` or empty.
    """
    if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
        raise ValueError("extract_features() needs a non-empty image array")

    if face_image.ndim == 2:  # grayscale input -> treat as 3-channel
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

    resized = cv2.resize(face_image, FACE_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features = []
    features.extend(_colour_histograms(resized))
    features.extend(_channel_statistics(resized, gray))
    features.append(_edge_density(gray))
    features.append(_blur_feature(gray))
    features.append(_fft_high_frequency_ratio(gray))

    vector = np.asarray(features, dtype=np.float64)
    # Safety net: the classifier must never receive NaN / infinity.
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    assert vector.shape == (FEATURE_VECTOR_LENGTH,), (
        f"Feature vector has length {vector.shape[0]}, expected {FEATURE_VECTOR_LENGTH}"
    )
    return vector
