from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticPair:
    source: np.ndarray
    reference: np.ndarray
    matrix: np.ndarray


def create_synthetic_pair(
    reference: np.ndarray,
    rotation: float = 7.0,
    scale: float = 0.88,
    translation: tuple[float, float] = (52.0, -26.0),
    brightness: float = 12.0,
    contrast: float = 1.18,
    noise_sigma: float = 0.0,
    blur: float = 0.0,
) -> SyntheticPair:
    if reference.ndim != 2:
        raise ValueError("Synthetic generator expects a grayscale reference image")
    height, width = reference.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation, scale)
    matrix[:, 2] += np.asarray(translation, dtype=np.float64)
    source = cv2.warpAffine(reference, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
    source = cv2.convertScaleAbs(source, alpha=contrast, beta=brightness)
    if blur > 0:
        kernel = max(3, int(round(blur)) * 2 + 1)
        source = cv2.GaussianBlur(source, (kernel, kernel), 0)
    if noise_sigma > 0:
        noise = np.random.default_rng(26166).normal(0.0, noise_sigma, source.shape)
        source = np.clip(source.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return SyntheticPair(source=source, reference=reference, matrix=matrix.astype(np.float32))