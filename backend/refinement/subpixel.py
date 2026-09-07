import cv2
import numpy as np

from backend.matching.classical import Match


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-6:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def refine_matches(source: np.ndarray, reference: np.ndarray, matches: list[Match], window: int = 5) -> int:
    refined = 0
    source_float = source.astype(np.float32)
    reference_float = reference.astype(np.float32)
    for match in matches:
        if not match.is_inlier:
            continue
        x, y = int(round(match.source_x)), int(round(match.source_y))
        rx, ry = int(round(match.reference_x)), int(round(match.reference_y))
        search_margin = 2
        if (
            x < window
            or y < window
            or rx < window + search_margin
            or ry < window + search_margin
            or x + window >= source.shape[1]
            or y + window >= source.shape[0]
            or rx + window + search_margin >= reference.shape[1]
            or ry + window + search_margin >= reference.shape[0]
        ):
            continue
        patch = source_float[y - window:y + window + 1, x - window:x + window + 1]
        search = reference_float[ry - window - search_margin:ry + window + search_margin + 1, rx - window - search_margin:rx + window + search_margin + 1]
        if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
            continue
        result = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, peak = cv2.minMaxLoc(result)
        if not np.isfinite(score) or score < 0.2:
            continue
        peak_x, peak_y = peak
        offset_x = 0.0
        offset_y = 0.0
        if 0 < peak_x < result.shape[1] - 1:
            offset_x = _parabolic_offset(result[peak_y, peak_x - 1], result[peak_y, peak_x], result[peak_y, peak_x + 1])
        if 0 < peak_y < result.shape[0] - 1:
            offset_y = _parabolic_offset(result[peak_y - 1, peak_x], result[peak_y, peak_x], result[peak_y + 1, peak_x])
        match.reference_x = rx - window - 2 + peak_x + offset_x + window
        match.reference_y = ry - window - 2 + peak_y + offset_y + window
        refined += 1
    return refined
