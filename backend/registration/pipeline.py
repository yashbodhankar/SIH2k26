import cv2
import numpy as np

from backend.config import RegistrationConfig
from backend.evaluation.metrics import calculate_metrics, structural_similarity
from backend.geometry.estimation import estimate_transform
from backend.matching.classical import Match, detect_and_match, detect_multiscale_matches, merge_matches, retrieve_region_matches, spatially_distribute
from backend.preprocessing.pipeline import prepare_image
from backend.refinement.subpixel import refine_matches


def register(source: np.ndarray, reference: np.ndarray, config: RegistrationConfig) -> dict:
    source_prepared = prepare_image(source, config.clahe, config.pyramid_levels)
    reference_prepared = prepare_image(reference, config.clahe, config.pyramid_levels)
    requested_method = config.method.lower().replace(" ", "_")
    if requested_method.startswith("structural"):
        requested_method = "structural"
    method = "sift" if requested_method in {"advanced", "structural"} else requested_method
    matches = detect_and_match(source_prepared.normalized, reference_prepared.normalized, method, config.ratio_threshold)
    if config.structural_matching and method == "sift":
        structural_matches = detect_and_match(source_prepared.gradient, reference_prepared.gradient, "sift", min(config.ratio_threshold, 0.82))
        matches = merge_matches(matches, structural_matches)
    if len(matches) < 3 and method == "sift":
        # Full-disk versus cropped-terrain pairs often have a large scale/framing gap.
        # Relax descriptor filtering only as a fallback; RANSAC remains mandatory.
        relaxed = detect_and_match(source_prepared.normalized, reference_prepared.normalized, "sift", 0.90, mutual=False)
        relaxed_structural = detect_and_match(source_prepared.gradient, reference_prepared.gradient, "sift", 0.90, mutual=False)
        matches = merge_matches(relaxed, relaxed_structural)
    if len(matches) < 3 and method == "sift":
        matches = detect_multiscale_matches(source_prepared.normalized, reference_prepared.normalized, "sift", 0.90)
        if config.structural_matching:
            matches = merge_matches(
                matches,
                detect_multiscale_matches(source_prepared.gradient, reference_prepared.gradient, "sift", 0.90),
            )
    if len(matches) < 3 and method == "sift":
        matches = retrieve_region_matches(source_prepared.gradient, reference_prepared.gradient)
    if len(matches) < 3:
        raise ValueError(f"Too few geometrically usable matches ({len(matches)})")
    matches = spatially_distribute(matches, source.shape[1], source.shape[0], config.grid_size, config.max_matches_per_cell, config.min_confidence)
    if len(matches) < 3:
        raise ValueError("Confidence filtering left too few matches")
    model, matrix, inlier_mask = estimate_transform(matches, config.transform, config.ransac_threshold)
    refined_count = refine_matches(source_prepared.normalized, reference_prepared.normalized, matches) if config.refine else 0
    if refined_count:
        try:
            model, matrix, inlier_mask = estimate_transform(matches, model, config.ransac_threshold)
        except ValueError:
            # Keep the stable global estimate if local refinement made the set degenerate.
            pass
    metrics = calculate_metrics(matches, reference.shape[1], reference.shape[0], config.grid_size)
    if metrics["inliers"] < 3:
        raise ValueError("Insufficient geometrically consistent correspondences")
    if model == "homography":
        registered = cv2.warpPerspective(source, matrix, (reference.shape[1], reference.shape[0]))
    else:
        registered = cv2.warpAffine(source, matrix, (reference.shape[1], reference.shape[0]))
    metrics["structural_similarity"] = structural_similarity(registered, reference)
    return {"registered": registered, "matches": matches, "model": model, "matrix": matrix, "metrics": metrics, "refined_count": refined_count, "source_prepared": source_prepared, "reference_prepared": reference_prepared}
