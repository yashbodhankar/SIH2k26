from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PreparedImage:
    gray: np.ndarray
    normalized: np.ndarray
    gradient: np.ndarray
    orientation: np.ndarray
    edges: np.ndarray
    structure: np.ndarray
    pyramid: list[np.ndarray]


def validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Image is empty or unreadable")
    if image.ndim not in (2, 3):
        raise ValueError("Expected a 2D grayscale or 3D channel image")
    if not np.isfinite(image.astype(np.float32)).all():
        raise ValueError("Image contains NaN or infinite values")
    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise ValueError("Image must be at least 32 x 32 pixels")


def to_gray(image: np.ndarray) -> np.ndarray:
    validate_image(image)
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8) if gray.dtype != np.uint8 else gray


def _normalize(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1, 99))
    if high <= low:
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clipped = np.clip(gray, low, high)
    return ((clipped - low) * 255.0 / (high - low)).astype(np.uint8)


def prepare_image(image: np.ndarray, clahe: bool = True, pyramid_levels: int = 4) -> PreparedImage:
    gray = to_gray(image)
    normalized = _normalize(gray)
    if clahe:
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orientation = cv2.phase(gx, gy, angleInDegrees=True)
    edges = cv2.Canny(normalized, 50, 140)
    structure = cv2.equalizeHist(normalized)
    structure = cv2.bilateralFilter(structure, 5, 35, 35)
    pyramid = [normalized]
    for _ in range(1, max(1, pyramid_levels)):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return PreparedImage(gray, normalized, gradient, orientation, edges, structure, pyramid)
