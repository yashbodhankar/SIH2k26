import cv2
import numpy as np

from backend.evaluation.metrics import structural_similarity


def test_structural_similarity_handles_brightness_change():
    image = np.zeros((80, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (75, 60), 180, 3)
    brighter = cv2.convertScaleAbs(image, alpha=1.4, beta=35)

    score = structural_similarity(image, brighter)

    assert score > 0.9