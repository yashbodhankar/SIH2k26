import cv2
import numpy as np

from backend.matching.classical import Match


def structural_similarity(source: np.ndarray, reference: np.ndarray) -> float:
    """Return illumination-tolerant gradient agreement in the [0, 1] range."""
    if source.shape[:2] != reference.shape[:2]:
        raise ValueError("Similarity images must have matching dimensions")
    source_gray = source if source.ndim == 2 else np.mean(source, axis=2)
    reference_gray = reference if reference.ndim == 2 else np.mean(reference, axis=2)
    source_gray = source_gray.astype(np.float32)
    reference_gray = reference_gray.astype(np.float32)
    source_gradient = cv2.magnitude(cv2.Sobel(source_gray, cv2.CV_32F, 1, 0), cv2.Sobel(source_gray, cv2.CV_32F, 0, 1))
    reference_gradient = cv2.magnitude(cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0), cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1))
    informative = (source_gradient > 5.0) | (reference_gradient > 5.0)
    if int(informative.sum()) < 32:
        return 0.0
    source_values = source_gradient[informative]
    reference_values = reference_gradient[informative]
    if np.std(source_values) < 1e-6 or np.std(reference_values) < 1e-6:
        return 0.0
    correlation = float(np.corrcoef(source_values, reference_values)[0, 1])
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def calculate_metrics(matches: list[Match], width: int, height: int, grid_size: int) -> dict:
    inliers = [m for m in matches if m.is_inlier]
    errors = [m.reprojection_error for m in inliers if m.reprojection_error is not None]
    occupied = {(min(grid_size - 1, int(m.source_x / max(width, 1) * grid_size)), min(grid_size - 1, int(m.source_y / max(height, 1) * grid_size))) for m in inliers}
    coverage = len(occupied) / max(grid_size * grid_size, 1)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    mean_error = float(np.mean(errors)) if errors else None
    inlier_ratio = len(inliers) / len(matches) if matches else 0.0
    confidence = float(np.clip(0.4 * inlier_ratio + 0.35 * coverage + 0.25 * max(0.0, 1.0 - (mean_error or 100.0) / 20.0), 0.0, 1.0))
    return {"candidate_matches": len(matches), "inliers": len(inliers), "inlier_ratio": inlier_ratio, "rmse_px": rmse, "mean_reprojection_error_px": mean_error, "spatial_coverage": coverage, "prototype_registration_confidence": confidence}
