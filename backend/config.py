from dataclasses import dataclass


@dataclass
class RegistrationConfig:
    method: str = "structural"
    transform: str = "auto"
    grid_size: int = 8
    max_matches_per_cell: int = 10
    min_confidence: float = 0.35
    ratio_threshold: float = 0.78
    ransac_threshold: float = 4.0
    clahe: bool = True
    pyramid_levels: int = 4
    refine: bool = True
    structural_matching: bool = True
