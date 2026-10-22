"""Severity scoring for detected packaging defects."""
from typing import Dict, Iterable, Tuple

# Engineering-judgement weights (see report for justification); varied in sensitivity analysis.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "unsealed": 1.0,
    "packaging-error": 0.9,
    "loose-meat": 0.5,
    "twisted-meat": 0.5,
    "wrinkle": 0.2,
}

Defect = Tuple[str, float, float]  # (class_name, w, h) normalised 0-1


def box_extent(w: float, h: float) -> float:
    """Fraction of the image area covered by a normalised box.

    Values above 1 are clipped to 1 (predicted boxes can overshoot).

    Raises:
        ValueError: If w or h is negative or NaN.
    """
    if not (w >= 0 and h >= 0):
        raise ValueError(f"Box size must be non-negative, got w={w}, h={h}")
    return min(w, 1.0) * min(h, 1.0)


def defect_severity(class_name: str, w: float, h: float,
                    weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    """Class weight multiplied by the box's image-area fraction.

    Raises:
        KeyError: If class_name has no weight.
    """
    if class_name not in weights:
        raise KeyError(f"No severity weight for class '{class_name}'")
    return weights[class_name] * box_extent(w, h)


def tray_severity(defects: Iterable[Defect],
                  weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    """Sum of defect severities for one tray image (0.0 if no defects)."""
    return sum(defect_severity(c, w, h, weights) for c, w, h in defects)


def tray_passes(defects: Iterable[Defect], threshold: float,
                weights: Dict[str, float] = DEFAULT_WEIGHTS) -> bool:
    """True if tray severity is strictly below the threshold."""
    return tray_severity(defects, weights) < threshold