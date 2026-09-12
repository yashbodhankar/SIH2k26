import cv2
import numpy as np

from backend.evaluation.synthetic import create_synthetic_pair


def test_synthetic_generator_returns_known_transform_pair():
    reference = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(reference, (70, 55), 18, 220, 3)

    pair = create_synthetic_pair(reference, rotation=0, scale=1, translation=(8, -4), noise_sigma=2)

    assert pair.source.shape == reference.shape
    assert pair.reference is reference
    assert np.allclose(pair.matrix[:, 2], [8, -4], atol=1)