"""Reading YOLO-format label files."""
from pathlib import Path
from typing import List, Tuple

Box = Tuple[int, float, float, float, float]  # (class_id, x_c, y_c, w, h)


def read_labels(label_path) -> List[Box]:
    """Read one YOLO label file.

    Args:
        label_path: Path to a .txt file, one box per line:
            ``class x_center y_center width height`` (all normalised 0-1).

    Returns:
        List of (class_id, x_center, y_center, width, height). Empty if the
        file has no boxes. Blank or short lines are skipped.

    Raises:
        ValueError: If a line contains non-numeric values.
    """
    rows = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows