import cv2
import numpy as np

from backend.matching.classical import Match
from backend.refinement.subpixel import refine_matches


def test_subpixel_refinement_recovers_fractional_translation():
    rng = np.random.default_rng(26166)
    source = rng.normal(0.0, 0.08, (120, 160)).astype(np.float32)
    cv2.circle(source, (70, 55), 12, 1.0, -1)
    cv2.line(source, (58, 47), (84, 63), 0.4, 2)
    shift = np.float32([[1, 0, 2.4], [0, 1, -1.7]])
    reference = cv2.warpAffine(source, shift, (160, 120), flags=cv2.INTER_LINEAR)
    match = Match(70, 55, 72, 53, 0.95, is_inlier=True)

    refined = refine_matches(source, reference, [match], window=8)

    assert refined == 1
    assert abs(match.reference_x - 72.4) < 0.35
    assert abs(match.reference_y - 53.3) < 0.35


def test_subpixel_refinement_skips_edge_match_without_crashing():
    image = np.zeros((32, 32), dtype=np.float32)
    match = Match(5, 5, 5, 5, 0.9, is_inlier=True)

    refined = refine_matches(image, image, [match], window=5)

    assert refined == 0