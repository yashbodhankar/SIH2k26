import cv2
import numpy as np

from backend.config import RegistrationConfig
from backend.registration.pipeline import register


def test_synthetic_translation_registration():
    image = np.zeros((320, 420), dtype=np.uint8)
    cv2.circle(image, (110, 130), 36, 180, -1)
    cv2.rectangle(image, (220, 80), (330, 230), 255, 3)
    cv2.line(image, (30, 280), (380, 260), 160, 4)
    matrix = np.float32([[1, 0, 18], [0, 1, 12]])
    moved = cv2.warpAffine(image, matrix, (420, 320))
    result = register(moved, image, RegistrationConfig(method="sift", refine=False, min_confidence=0.1))
    assert result["metrics"]["inliers"] >= 3
    assert result["model"] in {"similarity", "affine", "homography"}


def test_structural_registration_supports_cropped_reference():
    rng = np.random.default_rng(26166)
    source = rng.normal(70, 20, (420, 620)).clip(0, 255).astype(np.uint8)
    for center, radius in [((130, 150), 35), ((300, 180), 48), ((470, 290), 30), ((210, 340), 26)]:
        cv2.circle(source, center, radius, 190, 3)
        cv2.circle(source, center, max(3, radius // 5), 30, -1)
    reference = source[90:360, 180:540]
    result = register(reference, source, RegistrationConfig(method="structural", refine=False, min_confidence=0.1))
    assert result["metrics"]["inliers"] >= 3


def test_region_retrieval_fallback_finds_scaled_crop():
    source = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(source, (70, 55), (230, 180), 180, 3)
    cv2.circle(source, (155, 120), 28, 220, 3)
    reference = source[35:205, 45:275]
    result = register(reference, source, RegistrationConfig(method="structural", refine=False, min_confidence=0.1))
    assert result["metrics"]["inliers"] >= 3
