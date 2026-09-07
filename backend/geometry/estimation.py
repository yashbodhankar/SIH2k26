import cv2
import numpy as np

from backend.matching.classical import Match


def _points(matches: list[Match]):
    source = np.float32([[m.source_x, m.source_y] for m in matches])
    reference = np.float32([[m.reference_x, m.reference_y] for m in matches])
    return source, reference


def estimate_transform(matches: list[Match], model: str = "auto", threshold: float = 4.0) -> tuple[str, np.ndarray, np.ndarray]:
    if len(matches) < 3:
        raise ValueError("At least 3 correspondences are required")
    source, reference = _points(matches)
    candidates = ["similarity", "affine", "homography"] if model == "auto" else [model]
    best = None
    for candidate in candidates:
        if candidate == "similarity":
            if len(matches) < 2:
                continue
            matrix, mask = cv2.estimateAffinePartial2D(source, reference, method=cv2.RANSAC, ransacReprojThreshold=threshold)
        elif candidate == "affine":
            matrix, mask = cv2.estimateAffine2D(source, reference, method=cv2.RANSAC, ransacReprojThreshold=threshold)
        else:
            if len(matches) < 4:
                continue
            matrix, mask = cv2.findHomography(source, reference, cv2.RANSAC, threshold)
        if matrix is None or mask is None:
            continue
        inlier_count = int(mask.ravel().sum())
        projected = cv2.transform(source[None, :, :], matrix)[0] if candidate != "homography" else cv2.perspectiveTransform(source[None, :, :], matrix)[0]
        error = float(np.mean(np.linalg.norm(projected - reference, axis=1)[mask.ravel() > 0])) if inlier_count else float("inf")
        score = (inlier_count, -error, -{"similarity": 0, "affine": 1, "homography": 2}[candidate])
        if best is None or score > best[0]:
            best = (score, candidate, matrix, mask.ravel().astype(bool), projected)
    if best is None:
        raise ValueError("RANSAC could not estimate a stable transformation")
    _, selected_model, matrix, inliers, projected = best
    for match, point, valid in zip(matches, projected, inliers):
        match.is_inlier = bool(valid)
        match.reprojection_error = float(np.linalg.norm(point - [match.reference_x, match.reference_y]))
    return selected_model, matrix, inliers
