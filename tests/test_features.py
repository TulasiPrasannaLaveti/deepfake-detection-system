"""Tests for src.features -- runnable without any dataset.

Run either way, from the project root:

    python tests/test_features.py     # plain asserts, no pytest needed
    pytest tests/                     # also works if pytest is installed
"""

import sys
from pathlib import Path

import numpy as np

# Make `src.` imports work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_VECTOR_LENGTH, extract_features  # noqa: E402


def _random_image(height=100, width=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_feature_vector_length_is_fixed():
    features = extract_features(_random_image())
    assert isinstance(features, np.ndarray)
    assert features.shape == (FEATURE_VECTOR_LENGTH,)


def test_feature_vector_length_does_not_depend_on_image_size():
    small = extract_features(_random_image(40, 60, seed=1))
    large = extract_features(_random_image(300, 200, seed=2))
    assert small.shape == large.shape == (FEATURE_VECTOR_LENGTH,)


def test_features_are_finite():
    features = extract_features(_random_image())
    assert np.all(np.isfinite(features))


def test_solid_colour_image_does_not_crash():
    solid = np.full((128, 128, 3), 128, dtype=np.uint8)
    features = extract_features(solid)
    assert features.shape == (FEATURE_VECTOR_LENGTH,)
    assert np.all(np.isfinite(features))


def test_grayscale_image_is_accepted():
    gray = _random_image(seed=7)[:, :, 0]
    features = extract_features(gray)
    assert features.shape == (FEATURE_VECTOR_LENGTH,)


def test_empty_image_raises_value_error():
    try:
        extract_features(None)
    except ValueError:
        return
    raise AssertionError("extract_features(None) should raise ValueError")


if __name__ == "__main__":
    test_feature_vector_length_is_fixed()
    test_feature_vector_length_does_not_depend_on_image_size()
    test_features_are_finite()
    test_solid_colour_image_does_not_crash()
    test_grayscale_image_is_accepted()
    test_empty_image_raises_value_error()
    print("All feature tests passed.")
