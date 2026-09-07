from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Match:
    source_x: float
    source_y: float
    reference_x: float
    reference_y: float
    confidence: float
    is_inlier: bool = False
    reprojection_error: float | None = None


def _detector(method: str):
    if method.lower() == "orb":
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT is unavailable; install opencv-contrib-python")
    return cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02), cv2.NORM_L2


def detect_and_match(source: np.ndarray, reference: np.ndarray, method: str = "sift", ratio: float = 0.78, mutual: bool = True) -> list[Match]:
    detector, norm = _detector(method)
    source_keypoints, source_descriptors = detector.detectAndCompute(source, None)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference, None)
    if source_descriptors is None or reference_descriptors is None:
        return []
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(source_descriptors, reference_descriptors, k=2)
    reverse_pairs = matcher.knnMatch(reference_descriptors, source_descriptors, k=2)
    reverse_best = {pair[0].trainIdx: pair[0].queryIdx for pair in reverse_pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    matches = []
    for pair in pairs:
        if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
            continue
        best = pair[0]
        if mutual and reverse_best.get(best.trainIdx) != best.queryIdx:
            continue
        confidence = float(np.clip(1.0 - best.distance / max(pair[1].distance, 1e-6), 0.0, 1.0))
        source_point = source_keypoints[best.queryIdx].pt
        reference_point = reference_keypoints[best.trainIdx].pt
        matches.append(Match(source_point[0], source_point[1], reference_point[0], reference_point[1], confidence))
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def merge_matches(*match_sets: list[Match], distance: float = 3.0) -> list[Match]:
    """Merge structural/intensity matches without keeping duplicate points."""
    merged: list[Match] = []
    for matches in match_sets:
        for candidate in matches:
            duplicate = next(
                (
                    match
                    for match in merged
                    if (match.source_x - candidate.source_x) ** 2 + (match.source_y - candidate.source_y) ** 2 <= distance**2
                    and (match.reference_x - candidate.reference_x) ** 2 + (match.reference_y - candidate.reference_y) ** 2 <= distance**2
                ),
                None,
            )
            if duplicate is None:
                merged.append(candidate)
            elif candidate.confidence > duplicate.confidence:
                merged[merged.index(duplicate)] = candidate
    return sorted(merged, key=lambda item: item.confidence, reverse=True)


def detect_multiscale_matches(source: np.ndarray, reference: np.ndarray, method: str = "sift", ratio: float = 0.86) -> list[Match]:
    """Search relative image scales and map keypoints back to original coordinates."""
    scales = (0.5, 0.75, 1.0, 1.5, 2.0)
    all_matches: list[Match] = []
    for source_scale in scales:
        source_scaled = cv2.resize(source, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA if source_scale < 1 else cv2.INTER_LINEAR)
        for reference_scale in scales:
            reference_scaled = cv2.resize(reference, None, fx=reference_scale, fy=reference_scale, interpolation=cv2.INTER_AREA if reference_scale < 1 else cv2.INTER_LINEAR)
            if min(source_scaled.shape[:2]) < 48 or min(reference_scaled.shape[:2]) < 48:
                continue
            scaled_matches = detect_and_match(source_scaled, reference_scaled, method, ratio, mutual=False)
            for match in scaled_matches:
                match.source_x /= source_scale
                match.source_y /= source_scale
                match.reference_x /= reference_scale
                match.reference_y /= reference_scale
            all_matches.extend(scaled_matches)
    return merge_matches(all_matches, distance=4.0)


def retrieve_region_matches(source: np.ndarray, reference: np.ndarray, minimum_score: float = 0.35) -> list[Match]:
    """Find a cropped reference region in a larger source using gradient correlation."""
    source_float = source.astype(np.float32)
    reference_float = reference.astype(np.float32)
    scales = (0.25, 0.35, 0.5, 0.7, 1.0, 1.25)
    best: tuple[float, float, tuple[int, int]] | None = None
    for scale in scales:
        height = max(16, int(reference.shape[0] * scale))
        width = max(16, int(reference.shape[1] * scale))
        if height >= source.shape[0] or width >= source.shape[1]:
            continue
        template = cv2.resize(reference_float, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(source_float, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if best is None or score > best[0]:
            best = (float(score), scale, location)
    if best is None or best[0] < minimum_score:
        return []
    score, scale, (left, top) = best
    rows, columns = 3, 3
    matches: list[Match] = []
    for row in range(rows):
        for column in range(columns):
            reference_x = reference.shape[1] * (column + 0.5) / columns
            reference_y = reference.shape[0] * (row + 0.5) / rows
            matches.append(Match(left + reference_x * scale, top + reference_y * scale, reference_x, reference_y, score))
    return matches


def spatially_distribute(matches: list[Match], width: int, height: int, grid_size: int, max_per_cell: int, minimum_confidence: float) -> list[Match]:
    cells: dict[tuple[int, int], list[Match]] = {}
    for match in matches:
        if match.confidence < minimum_confidence:
            continue
        column = min(grid_size - 1, int(match.source_x / max(width, 1) * grid_size))
        row = min(grid_size - 1, int(match.source_y / max(height, 1) * grid_size))
        cells.setdefault((row, column), []).append(match)
    selected = []
    for cell_matches in cells.values():
        selected.extend(sorted(cell_matches, key=lambda item: item.confidence, reverse=True)[:max_per_cell])
    return sorted(selected, key=lambda item: item.confidence, reverse=True)
