import json

import cv2
import numpy as np

from backend.matching.classical import Match


def matches_csv(matches: list[Match]) -> str:
    header = "source_x,source_y,reference_x,reference_y,confidence,is_inlier,reprojection_error"
    rows = [
        ",".join(
            [
                f"{match.source_x:.6f}",
                f"{match.source_y:.6f}",
                f"{match.reference_x:.6f}",
                f"{match.reference_y:.6f}",
                f"{match.confidence:.6f}",
                str(match.is_inlier).lower(),
                "" if match.reprojection_error is None else f"{match.reprojection_error:.6f}",
            ]
        )
        for match in matches
    ]
    return "\n".join([header, *rows]) + "\n"


def result_json(result: dict) -> str:
    payload = {
        "transformation_model": result["model"],
        "matrix": np.asarray(result["matrix"], dtype=float).tolist(),
        "metrics": result["metrics"],
        "refined_points": result["refined_count"],
    }
    return json.dumps(payload, indent=2)


def metrics_json(metrics: dict) -> str:
    return json.dumps(metrics, indent=2)


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image as PNG")
    return encoded.tobytes()