import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.prompt_constraints import extract_constraints
from src.visual_reflector import VisualReflector, build_round_record


def test_select_best_uses_vlm_and_parses_json_scores() -> None:
    response = json.dumps(
        {
            "selected_index": 1,
            "scores": [
                {"index": 0, "score": 4, "reason": "wrong color"},
                {"index": 1, "score": 9, "reason": "closest to idea"},
            ],
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    result = reflector.select_best(
        "a blue dog on grass",
        ["white dog on grass", "blue dog on grass"],
        ["mock://image/0", "mock://image/1"],
    )

    assert result["selected_index"] == 1
    assert result["selected_image"] == "mock://image/1"
    assert result["selected_prompt"] == "blue dog on grass"
    assert result["scores"][1]["score"] == pytest.approx(0.9)
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_paths"] == ["mock://image/0", "mock://image/1"]
    assert "VisualCriticAgent" in vlm.calls[0]["prompt"]


def test_select_best_accepts_idea2img_start_end_fallback() -> None:
    vlm = MockVLMClient(responses=["Image 1 has the best alignment. <START>1<END>"])
    reflector = VisualReflector(vlm)

    result = reflector.select_best(
        "a red car",
        ["red car", "red car in rain"],
        ["mock://image/0", "mock://image/1"],
    )

    assert result["selected_index"] == 1
    assert result["scores"][1]["score"] == pytest.approx(1.0)


def test_reflect_returns_structured_critique_from_json() -> None:
    response = json.dumps(
        {
            "score": 0.62,
            "errors": [
                {
                    "type": "wrong_count",
                    "evidence": "The image shows 12 apples instead of 8.",
                    "prompt_span": "8 apples",
                }
            ],
            "strengths": ["The table and background are visible."],
            "revision_hint": "Specify exactly eight apples in one row on the table.",
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    critique = reflector.reflect(
        "8 apples on the table",
        "8 pink apples on a black table",
        "mock://image/apples",
        history=[{"round": 0, "revision_hint": "make apples count explicit"}],
    )

    assert critique["score"] == pytest.approx(0.62)
    assert critique["image_path"] == "mock://image/apples"
    assert critique["errors"][0]["type"] == "wrong_count"
    assert critique["strengths"] == ["The table and background are visible."]
    assert "exactly eight apples" in critique["revision_hint"]
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_paths"] == ["mock://image/apples"]


def test_reflect_parses_description_fields_and_dict_strengths() -> None:
    response = json.dumps(
        {
            "score": 8,
            "errors": [
                {
                    "type": "wrong_attribute",
                    "description": "The umbrella is red, but the user asked for blue.",
                }
            ],
            "strengths": [
                {"description": "Rainy street mood is strong."},
                {"text": "Robot is visible."},
            ],
            "revision_hint": "Make the umbrella blue and visibly held.",
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    critique = reflector.reflect(
        "a red robot holding a blue umbrella",
        "a red robot holding a blue umbrella",
        "mock://image/robot",
    )

    assert critique["score"] == pytest.approx(0.8)
    assert critique["errors"][0]["evidence"] == (
        "The umbrella is red, but the user asked for blue."
    )
    assert critique["strengths"] == ["Rainy street mood is strong.", "Robot is visible."]
    assert "original user" in vlm.calls[0]["prompt"]


def test_reflect_respects_explicit_empty_errors() -> None:
    response = json.dumps(
        {
            "score": 0.91,
            "errors": [],
            "strengths": ["The image satisfies the prompt."],
            "revision_hint": "No change.",
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    critique = reflector.reflect(
        "a futuristic city",
        "a futuristic city",
        "mock://image/city",
    )

    assert critique["score"] == pytest.approx(0.91)
    assert critique["errors"] == []
    assert critique["revision_hint"] == "No change."


def test_select_best_clamps_out_of_range_index_and_records_warning() -> None:
    response = json.dumps(
        {
            "selected_index": 1,
            "scores": {
                "object_counts": 10,
                "attributes": 9,
                "overall_image_quality": 8,
            },
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    result = reflector.select_best(
        "a red robot holding a blue umbrella",
        ["a red robot holding a blue umbrella"],
        ["mock://image/0"],
    )

    assert result["selected_index"] == 0
    assert result["warnings"]
    assert result["selected_image"] == "mock://image/0"


def test_reflect_accepts_natural_language_reason_fallback() -> None:
    vlm = MockVLMClient(
        responses=[
            "<START>The person is not in boat pose; describe the V-shaped pose clearly.<END>"
        ]
    )
    reflector = VisualReflector(vlm)

    critique = reflector.reflect(
        "person practicing yoga boat pose at beach",
        "person doing boat pose on a beach",
        "mock://image/yoga",
    )

    assert critique["score"] == pytest.approx(0.5)
    assert critique["errors"][0]["type"] == "wrong_relation"
    assert "V-shaped pose" in critique["revision_hint"]


def test_check_constraints_parses_failed_color_and_relation_checks() -> None:
    response = json.dumps(
        {
            "passed": False,
            "constraint_score": 0.45,
            "checks": [
                {
                    "type": "wrong_attribute",
                    "target": "umbrella",
                    "expected": "blue umbrella",
                    "observed": "red umbrella",
                    "passed": False,
                    "description": "The canopy appears red instead of blue.",
                },
                {
                    "type": "wrong_relation",
                    "target": "robot hand and umbrella handle",
                    "expected": "clearly gripping",
                    "observed": "handle is hidden",
                    "passed": False,
                    "description": "The robot is not visibly gripping the handle.",
                },
            ],
            "errors": [],
            "revision_hint": "Make the umbrella visibly blue and show the hand on the handle.",
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    check = reflector.check_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella",
        "a small red robot clearly gripping the handle of a blue umbrella",
        "mock://image/robot",
        extract_constraints(
            "a small red robot clearly gripping the handle of a blue umbrella"
        ),
    )

    assert check["passed"] is False
    assert check["score"] == pytest.approx(0.45)
    assert len(check["checks"]) == 2
    assert check["errors"][0]["type"] == "wrong_attribute"
    assert "blue" in check["revision_hint"]
    assert "strict visual constraint checker" in vlm.calls[0]["prompt"]


def test_check_constraints_preserves_localized_errors_and_repair_plan() -> None:
    response = json.dumps(
        {
            "passed": False,
            "score": 0.38,
            "checks": [],
            "localized_errors": [
                {
                    "error_type": "text_symbol",
                    "target_object": "top sign",
                    "target_attribute": "exact text",
                    "expected": "yellow text 'NO'",
                    "observed": "text is missing",
                    "bbox": [105, 45, 305, 210],
                    "bbox_confidence": 0.82,
                    "repair_kind": "text_overlay",
                    "editability": "high",
                    "repair_instruction": "render exact yellow text NO",
                    "prompt_patch": "top sign must display exact yellow text NO",
                }
            ],
        }
    )
    vlm = MockVLMClient(responses=[response])
    reflector = VisualReflector(vlm)

    check = reflector.check_constraints(
        "A black sign displays the exact yellow text 'NO'.",
        "A black sign displays the exact yellow text 'NO'.",
        "mock://image/sign",
    )

    assert check["passed"] is False
    assert check["localized_errors"][0]["bbox"] == [105, 45, 305, 210]
    assert check["localized_errors"][0]["repair_kind"] == "text_overlay"
    assert check["repair_plan"]["typed_route"] == "text_overlay"
    assert check["repair_plan"]["bbox"] == [105, 45, 305, 210]
    assert "localized_errors" in vlm.calls[0]["prompt"]


def test_build_round_record_is_json_compatible() -> None:
    record = build_round_record(
        round_index=1,
        prompt="8 apples on a table",
        images=["mock://image/0", "mock://image/1"],
        selected_image="mock://image/1",
        feedback={"score": 0.7, "errors": [], "strengths": [], "revision_hint": "ok"},
        revised_prompt="exactly 8 apples in one row on a table",
    )

    dumped = json.dumps(record)

    assert '"selected_image": "mock://image/1"' in dumped
    assert record["round"] == 1
    assert record["feedback"]["score"] == 0.7
