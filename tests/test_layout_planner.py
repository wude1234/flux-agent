import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockLLMClient
from src.layout_planner import (
    LayoutPlanner,
    build_mock_layout_response,
    layout_to_enriched_prompt,
    layout_to_prompt_package,
    parse_layout_response,
    should_plan_layout,
    validate_layout,
)


def test_layout_planner_uses_llm_and_normalizes_ordered_objects() -> None:
    response = json.dumps(
        {
            "canvas_size": [512, 512],
            "background": {
                "description": "rainy street with wet pavement and red robot signage",
                "viewpoint": "front view",
            },
            "objects": [
                {
                    "name": "blue umbrella",
                    "description": "blue umbrella canopy with visible handle",
                    "bbox": [0.30, 0.06, 0.46, 0.26],
                    "order": 2,
                    "relations": ["above red robot"],
                },
                {
                    "name": "red robot",
                    "description": "small red robot gripping umbrella handle",
                    "bbox": [180, 250, 120, 160],
                    "order": 1,
                    "relations": ["below blue umbrella"],
                },
            ],
        }
    )
    llm = MockLLMClient(responses=[f"```json\n{response}\n```"])
    planner = LayoutPlanner(llm)

    layout = planner.plan(
        "a small red robot clearly gripping the handle of a blue umbrella",
        canvas_size=(512, 512),
    )

    assert len(llm.calls) == 1
    assert "ChainArchitect" in llm.calls[0]
    assert layout["canvas_size"] == [512, 512]
    assert layout["objects"][0]["name"] == "red robot"
    assert layout["objects"][1]["bbox"] == [154, 31, 236, 133]
    assert "red robot" not in layout["background"]["description"].lower()
    assert layout["warnings"]
    assert layout["raw_response"]
    assert layout["request"]


def test_parse_layout_response_accepts_nested_layout_and_background_prompt() -> None:
    response = json.dumps(
        {
            "layout": {
                "canvas_size": [1024, 768],
                "background_prompt": "cinematic rainy street",
                "viewpoint": "low-angle medium shot",
                "objects": [
                    {
                        "object": "robot",
                        "prompt": "small red robot",
                        "box": {"x": 100, "y": 300, "width": 200, "height": 300},
                    }
                ],
            }
        }
    )

    parsed = parse_layout_response(response)
    layout = validate_layout(parsed)

    assert layout["background"]["description"] == "cinematic rainy street"
    assert layout["background"]["viewpoint"] == "low-angle medium shot"
    assert layout["objects"][0]["bbox"] == [100, 300, 200, 300]


def test_validate_layout_rejects_bbox_outside_canvas() -> None:
    layout = {
        "canvas_size": [512, 512],
        "background": {"description": "rainy street", "viewpoint": "front view"},
        "objects": [
            {
                "name": "umbrella",
                "description": "blue umbrella",
                "bbox": [400, 20, 200, 100],
            }
        ],
    }

    with pytest.raises(ValueError, match="outside the canvas"):
        validate_layout(layout)


def test_validate_layout_strict_background_rejects_foreground_terms() -> None:
    layout = {
        "canvas_size": [512, 512],
        "background": {
            "description": "rainy street behind the blue umbrella",
            "viewpoint": "front view",
        },
        "objects": [
            {
                "name": "blue umbrella",
                "description": "blue umbrella",
                "bbox": [100, 20, 240, 160],
            }
        ],
    }

    with pytest.raises(ValueError, match="duplicates foreground"):
        validate_layout(layout, strict_background=True)


def test_layout_to_prompt_package_and_enriched_prompt_are_json_compatible() -> None:
    layout = validate_layout(
        {
            "canvas_size": [512, 512],
            "background": {"description": "rainy street", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "red robot",
                    "description": "small red robot gripping handle",
                    "bbox": [210, 250, 110, 180],
                    "order": 2,
                    "relations": ["below umbrella"],
                },
                {
                    "name": "blue umbrella",
                    "description": "blue canopy with visible handle",
                    "bbox": [160, 40, 240, 150],
                    "order": 1,
                    "relations": ["above robot"],
                },
            ],
        }
    )

    package = layout_to_prompt_package(
        layout,
        user_prompt="a red robot gripping a blue umbrella",
    )
    enriched = layout_to_enriched_prompt(
        "a red robot gripping a blue umbrella",
        package,
    )

    assert json.loads(json.dumps(package))["generation_order"] == [
        "blue umbrella",
        "red robot",
    ]
    assert "background: rainy street, front view" in package["layout_prompt"]
    assert "bbox [160, 40, 240, 150]" in package["layout_prompt"]
    assert "foreground layout:" in enriched
    assert "keep each object inside its bbox" in enriched


def test_should_plan_layout_detects_multi_object_spatial_prompt() -> None:
    assert should_plan_layout(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )
    assert not should_plan_layout("a blue sky")


def test_build_mock_layout_response_is_valid_for_robot_umbrella() -> None:
    response = build_mock_layout_response(
        "a small red robot clearly gripping the handle of a blue umbrella",
        canvas_size=(512, 512),
    )

    layout = validate_layout(parse_layout_response(response, canvas_size=(512, 512)))

    assert [item["name"] for item in layout["objects"]] == ["robot", "umbrella"]
    assert all(item["bbox"][2] > 0 and item["bbox"][3] > 0 for item in layout["objects"])


def test_build_mock_layout_response_respects_basic_spatial_relations() -> None:
    response = build_mock_layout_response(
        (
            "A red cube is left of a blue sphere, and the blue sphere is under "
            "a green cone; all three objects are visible."
        ),
        canvas_size=(512, 512),
    )

    layout = validate_layout(parse_layout_response(response, canvas_size=(512, 512)))
    by_name = {item["name"]: item for item in layout["objects"]}

    cube_bbox = by_name["cube"]["bbox"]
    sphere_bbox = by_name["sphere"]["bbox"]
    cone_bbox = by_name["cone"]["bbox"]
    assert cube_bbox[0] + cube_bbox[2] / 2 < sphere_bbox[0] + sphere_bbox[2] / 2
    assert sphere_bbox[1] + sphere_bbox[3] / 2 > cone_bbox[1] + cone_bbox[3] / 2
    assert any("cube left of sphere" in rel for rel in by_name["cube"]["relations"])
    assert any("sphere under cone" in rel for rel in by_name["sphere"]["relations"])
