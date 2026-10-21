import pytest
from src.severity import (box_extent, defect_severity, tray_severity,
                          tray_passes)


def test_quarter_image_box():
    assert box_extent(0.5, 0.5) == pytest.approx(0.25)


def test_full_image_box():
    assert box_extent(1.0, 1.0) == pytest.approx(1.0)


def test_oversized_box_is_clipped():
    assert box_extent(1.2, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("w,h", [(-0.1, 0.5), (0.5, -0.1), (float("nan"), 0.5)])
def test_invalid_box_raises(w, h):
    with pytest.raises(ValueError):
        box_extent(w, h)


def test_weighted_severity():
    assert defect_severity("unsealed", 0.5, 0.5) == pytest.approx(0.25)
    assert defect_severity("wrinkle", 0.5, 0.5) == pytest.approx(0.05)


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        defect_severity("scratch", 0.1, 0.1)


def test_tray_sum_and_empty():
    defects = [("unsealed", 0.5, 0.5), ("wrinkle", 0.5, 0.5)]
    assert tray_severity(defects) == pytest.approx(0.30)
    assert tray_severity([]) == 0.0


def test_pass_fail_threshold():
    defects = [("unsealed", 0.5, 0.5)]  # severity 0.25
    assert tray_passes(defects, threshold=0.3)
    assert not tray_passes(defects, threshold=0.2)