import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.local_editor import MockInpaintEditor
from src.relation_repair import (
    RelationActionRepairer,
    build_relation_region_localization_request,
    build_relation_repair_prompt,
    build_relation_verification_request,
    locate_relation_repair_region,
    parse_relation_region_localization_response,
    parse_relation_verification_response,
    plan_relation_repair_region,
    should_trigger_relation_repair,
)


USER_PROMPT = (
    "a small red robot clearly gripping the handle of a blue umbrella, "
    "cinematic rainy street photo"
)


def test_should_trigger_relation_repair_for_grip_contact_error() -> None:
    critique = {
        "score": 0.52,
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": "The robot hand is not clearly gripping the umbrella handle.",
                "prompt_span": "gripping the handle",
            }
        ],
        "revision_hint": "Show visible contact between the claw and handle.",
    }

    assert should_trigger_relation_repair(USER_PROMPT, critique)


def test_should_not_trigger_relation_repair_for_color_only_error() -> None:
    critique = {
        "score": 0.4,
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The umbrella is red instead of blue.",
                "prompt_span": "blue umbrella",
            }
        ],
    }

    assert not should_trigger_relation_repair(USER_PROMPT, critique)


def test_plan_relation_repair_region_uses_layout_union(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), (20, 20, 20)).save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [100, 80],
            "objects": [
                {"name": "small red robot", "bbox": [35, 34, 28, 34]},
                {"name": "blue umbrella", "bbox": [20, 8, 60, 36]},
            ],
        }
    }
    critique = {
        "errors": [{"type": "wrong_relation", "evidence": "The grip is hidden."}],
        "revision_hint": "Make the hand grip the handle.",
    }

    region, detection = plan_relation_repair_region(
        image_path,
        layout_context,
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        critique=critique,
        constraints=None,
    )

    assert region.name == "relation_action_contact"
    assert region.canvas_size == [100, 80]
    assert detection["method"] == "layout_hand_handle_contact_band"
    assert 0 <= region.bbox[0] < 100
    assert 0 <= region.bbox[1] < 80
    assert region.bbox[2] < 48
    assert region.bbox[3] < 40
    assert "grips the umbrella handle" in region.prompt
    assert "detached handle" in region.negative_prompt
    assert "changed face" in region.negative_prompt
    assert "changed identity" in region.negative_prompt


def test_plan_relation_repair_region_targets_lower_hand_handle_band(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (768, 768), (20, 20, 20)).save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [1024, 1024],
            "objects": [
                {"name": "blue umbrella", "bbox": [420, 180, 320, 360]},
                {"name": "small red robot", "bbox": [460, 320, 240, 400]},
            ],
        }
    }

    region, detection = plan_relation_repair_region(
        image_path,
        layout_context,
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        critique={"revision_hint": "Repair visible grip contact."},
        constraints=None,
    )

    robot_bbox = detection["subject_bbox"]
    assert region.bbox[1] > robot_bbox[1] + int(robot_bbox[3] * 0.35)
    assert region.bbox[1] < robot_bbox[1] + int(robot_bbox[3] * 0.75)
    assert region.bbox[2] < robot_bbox[2]
    assert region.bbox[3] < robot_bbox[3] * 0.75
    assert detection["targeting"] == "subject_upper_body_to_handle_lower_stem"


def test_plan_relation_repair_region_prefers_visual_bbox_over_layout(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), (20, 20, 20)).save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [100, 80],
            "objects": [
                {"name": "small red robot", "bbox": [70, 30, 20, 30]},
                {"name": "blue umbrella", "bbox": [60, 6, 36, 40]},
            ],
        }
    }

    region, detection = plan_relation_repair_region(
        image_path,
        layout_context,
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        critique={"revision_hint": "Repair visible grip contact."},
        constraints=None,
        visual_bbox=[10, 44, 18, 16],
        visual_diagnostics={"method": "vlm_contact_bbox_locator", "found": True},
    )

    assert detection["method"] == "image_grounded_vlm_contact_bbox"
    assert detection["targeting"] == "actual_image_hand_handle_contact"
    assert region.bbox[0] < 20
    assert region.bbox[1] < 50
    assert detection["layout_fallback_bbox"][0] > 50


def test_parse_relation_region_localization_response_accepts_bbox_shapes() -> None:
    parsed = parse_relation_region_localization_response(
        json.dumps(
            {
                "found": True,
                "contact_bbox": {"x1": 12, "y1": 20, "x2": 42, "y2": 55},
                "confidence": 0.8,
            }
        ),
        image_size=(100, 80),
    )

    assert parsed["found"] is True
    assert parsed["bbox"] == [12, 20, 30, 35]
    assert parsed["confidence"] == 0.8


def test_locate_relation_repair_region_uses_generated_image_content(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), (20, 20, 20)).save(image_path)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "found": True,
                    "bbox": [8, 42, 22, 18],
                    "confidence": 0.92,
                    "reason": "The visible claw and handle are near the left side.",
                }
            )
        ]
    )

    bbox, diagnostics = locate_relation_repair_region(
        vlm,
        image_path,
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        critique={"revision_hint": "Repair visible grip contact."},
        layout_context={
            "layout": {
                "canvas_size": [100, 80],
                "objects": [
                    {"name": "small red robot", "bbox": [70, 30, 20, 30]},
                    {"name": "blue umbrella", "bbox": [60, 6, 36, 40]},
                ],
            }
        },
    )

    assert bbox == [8, 42, 22, 18]
    assert diagnostics["found"] is True
    assert diagnostics["method"] == "vlm_contact_bbox_locator"
    assert "Use the visible image content as the source of truth" in vlm.calls[0]["prompt"]


def test_relation_repairer_generates_candidates_and_selects_verified_best(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [64, 64],
            "objects": [
                {"name": "small red robot", "bbox": [22, 30, 20, 24]},
                {"name": "blue umbrella", "bbox": [12, 8, 44, 30]},
            ],
        }
    }
    critique = {
        "score": 0.45,
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": "The hand is detached from the umbrella handle.",
            }
        ],
        "revision_hint": "Repair visible grip contact.",
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps({"found": True, "bbox": [18, 34, 18, 18], "confidence": 0.9}),
            json.dumps(
                {
                    "score": 0.5,
                    "passed": False,
                    "checks": {"visible_grip": False},
                    "errors": ["The handle is still detached."],
                }
            ),
            json.dumps(
                {
                    "score": 0.91,
                    "passed": True,
                    "checks": {
                        "handle_visible": True,
                        "hand_or_claw_visible": True,
                        "visible_grip": True,
                        "physical_contact": True,
                        "handle_connected_to_umbrella": True,
                        "not_merely_near_or_supported": True,
                        "user_colors_preserved": True,
                    },
                    "strengths": ["The claw visibly wraps around the handle."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    repairer = RelationActionRepairer(
        vlm,
        MockInpaintEditor(prefix="relation_test"),
        candidates=2,
        pass_threshold=0.82,
    )

    result = repairer.repair(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        image_path=str(image_path),
        critique=critique,
        constraints=None,
        output_dir=tmp_path / "repair",
        layout_context=layout_context,
        round_index=1,
    )

    assert result["accepted"] is True
    assert result["selected_index"] == 1
    assert result["score"] == 0.91
    assert len(result["candidates"]) == 2
    assert Path(result["edited_image"]).exists()
    assert Path(result["candidates"][1]["mask_path"]).exists()
    assert result["detection"]["method"] == "image_grounded_vlm_contact_bbox"
    assert result["detection"]["detected_bbox"][0] < 25
    assert result["candidates"][1]["edit_result"]["region"]["bbox"][2] <= result[
        "candidates"
    ][0]["edit_result"]["region"]["bbox"][2]
    assert len(vlm.calls) == 3


def test_relation_verification_rejects_high_score_without_local_contact_evidence() -> None:
    parsed = parse_relation_verification_response(
        json.dumps(
            {
                "score": 0.95,
                "passed": True,
                "checks": {
                    "visible_grip": True,
                    "physical_contact": True,
                    "handle_connected_to_umbrella": True,
                    "user_colors_preserved": True,
                },
                "errors": [],
                "strengths": ["The subject appears close to the handle."],
            }
        ),
        pass_threshold=0.82,
    )

    assert parsed["passed"] is False
    assert parsed["evidence_quality"]["passed"] is False
    failure_types = {
        item["type"] for item in parsed["evidence_quality"]["failures"]
    }
    assert "missing_relation_evidence:hand_or_claw_visible" in failure_types
    assert "missing_relation_evidence:not_merely_near_or_supported" in failure_types


def test_relation_verification_accepts_complete_local_contact_evidence() -> None:
    parsed = parse_relation_verification_response(
        json.dumps(
            {
                "score": 0.9,
                "passed": True,
                "checks": {
                    "handle_visible": True,
                    "hand_or_claw_visible": True,
                    "visible_grip": True,
                    "physical_contact": True,
                    "handle_connected_to_umbrella": True,
                    "not_merely_near_or_supported": True,
                    "user_colors_preserved": True,
                },
            }
        ),
        pass_threshold=0.82,
    )

    assert parsed["passed"] is True
    assert parsed["evidence_quality"]["passed"] is True


def test_parse_relation_verification_response_has_text_fallback() -> None:
    parsed = parse_relation_verification_response(
        "Score: 0.35. The hand is detached from the handle, no contact is visible."
    )

    assert parsed["score"] == 0.35
    assert parsed["passed"] is False
    assert parsed["errors"]


def test_build_relation_repair_prompt_stays_compact() -> None:
    prompt = build_relation_repair_prompt(
        USER_PROMPT,
        prompt=USER_PROMPT,
        critique={"revision_hint": "make the hand and handle visibly connected"},
    )

    assert "red robot" in prompt
    assert "blue umbrella" in prompt
    assert "visible physical contact" in prompt
    assert "preserve the existing face" in prompt
    assert len(prompt.split()) <= 42


def test_relation_vlm_requests_compact_large_runtime_critique() -> None:
    critique = {
        "score": 0.2,
        "revision_hint": "make the claw visibly wrap around the handle",
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": "The robot claw is detached from the umbrella handle.",
                "prompt_span": "gripping the handle",
            }
        ],
        "candidate_arbitration": {"blob": "x" * 8000},
        "evidence_chain": [{"look": "y" * 8000}],
        "request": "raw request " * 1000,
        "raw_response": "raw response " * 1000,
    }

    localization = build_relation_region_localization_request(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT + ", " + "cinematic detail " * 600,
        image_path="/tmp/fake.png",
        image_size=(768, 768),
        critique=critique,
        layout_context={"layout": {"objects": []}},
    )
    verification = build_relation_verification_request(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT + ", " + "cinematic detail " * 600,
        image_path="/tmp/fake.png",
        critique=critique,
    )

    for request in (localization, verification):
        assert "Previous critique summary" in request
        assert "robot claw is detached" in request
        assert "candidate_arbitration" not in request
        assert "evidence_chain" not in request
        assert "raw request" not in request
        assert "raw response" not in request
        assert len(request) < 7000
