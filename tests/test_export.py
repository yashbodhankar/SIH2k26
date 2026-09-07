import numpy as np

from backend.evaluation.export import matches_csv, result_json
from backend.matching.classical import Match


def test_exports_preserve_match_and_transform_values():
    matches = [Match(1.25, 2.5, 3.75, 4.0, 0.9, True, 0.12)]
    csv_text = matches_csv(matches)
    assert "source_x,source_y" in csv_text
    assert "1.250000,2.500000,3.750000,4.000000,0.900000,true,0.120000" in csv_text

    payload = result_json(
        {
            "model": "affine",
            "matrix": np.eye(2, 3),
            "metrics": {"inliers": 1},
            "refined_count": 1,
        }
    )
    assert '"transformation_model": "affine"' in payload
    assert '"refined_points": 1' in payload