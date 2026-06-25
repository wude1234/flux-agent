from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_grounded_sam2_mask import _normalize_florence_polygons


def test_normalize_florence_polygons_accepts_flat_polygon() -> None:
    polygons = _normalize_florence_polygons([[0, 0, 10, 0, 10, 10, 0, 10]])

    assert len(polygons) == 1
    assert polygons[0] == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_normalize_florence_polygons_accepts_multi_object_nested_polygons() -> None:
    polygons = _normalize_florence_polygons(
        [
            [
                [[1, 2], [5, 2], [5, 8], [1, 8]],
                [[10, 4], [14, 4], [14, 9], [10, 9]],
            ],
            [[[20, 1], [24, 1], [24, 5], [20, 5]]],
        ]
    )

    assert polygons == [
        [[1, 2], [5, 2], [5, 8], [1, 8]],
        [[10, 4], [14, 4], [14, 9], [10, 9]],
        [[20, 1], [24, 1], [24, 5], [20, 5]],
    ]
