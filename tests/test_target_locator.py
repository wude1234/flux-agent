from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.target_locator import (
    build_target_region_localization_request,
    layout_with_target_bbox,
    parse_target_region_localization_response,
)


def test_parse_target_region_localization_response_accepts_json_bbox() -> None:
    parsed = parse_target_region_localization_response(
        '{"found": true, "bbox": [10, 20, 30, 40], "confidence": 0.9}',
        image_size=(100, 100),
    )

    assert parsed["found"] is True
    assert parsed["bbox"] == [10, 20, 30, 40]
    assert parsed["confidence"] == 0.9


def test_parse_target_region_localization_response_accepts_nested_region() -> None:
    parsed = parse_target_region_localization_response(
        '{"target_visible": true, "region": {"target_bbox": [90, 80, 30, 40]}, "score": 9}',
        image_size=(100, 100),
    )

    assert parsed["found"] is True
    assert parsed["bbox"] == [90, 80, 10, 20]
    assert parsed["confidence"] == 0.9


def test_parse_target_region_localization_response_rejects_missing_bbox() -> None:
    parsed = parse_target_region_localization_response(
        '{"found": false, "reason": "target is not visible"}',
        image_size=(100, 100),
    )

    assert parsed["found"] is False
    assert parsed["bbox"] is None
    assert "missing_or_invalid_bbox" in parsed["warnings"]


def test_layout_with_target_bbox_updates_matching_object() -> None:
    layout = {
        "layout": {
            "canvas_size": [1024, 1024],
            "objects": [
                {"name": "small red robot", "bbox": [10, 10, 20, 20]},
                {"name": "blue umbrella", "bbox": [100, 100, 200, 200]},
            ],
        }
    }

    updated = layout_with_target_bbox(
        layout,
        "umbrella",
        [12, 44, 32, 14],
        image_size=(80, 80),
    )

    assert updated["layout"]["canvas_size"] == [80, 80]
    assert updated["layout"]["objects"][1]["bbox"] == [12, 44, 32, 14]
    assert updated["layout"]["objects"][1]["bbox_source"] == "vlm_target_region_locator"
    assert layout["layout"]["objects"][1]["bbox"] == [100, 100, 200, 200]


def test_target_region_request_compacts_large_critique() -> None:
    request = build_target_region_localization_request(
        user_prompt="a red robot holding a blue umbrella",
        prompt="a red robot holding a blue umbrella, " + "extra style " * 1000,
        image_path="/tmp/fake.png",
        image_size=(768, 768),
        target_name="umbrella",
        target_region="canopy",
        repair_goal="repair the umbrella so it is blue",
        critique={
            "score": 0.2,
            "raw_response": "x" * 50000,
            "request": "y" * 50000,
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The umbrella is red instead of blue.",
                    "prompt_span": "umbrella",
                },
                {
                    "type": "wrong_relation",
                    "evidence": "The robot is not gripping the handle.",
                    "prompt_span": "robot",
                },
            ],
        },
        layout_context={
            "layout": {
                "canvas_size": [1024, 1024],
                "objects": [{"name": "blue umbrella", "bbox": [1, 2, 3, 4]}],
            }
        },
    )

    assert len(request) < 6000
    assert "raw_response" not in request
    assert "The umbrella is red instead of blue" in request
    assert "not gripping" not in request
