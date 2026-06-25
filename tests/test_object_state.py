import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.object_state import (
    augment_record_with_object_geometry,
    build_evidence_chain,
    parse_object_state_response,
)


def _left_of_record() -> dict:
    return {
        "image_path": "mock://cup",
        "prompt": "a blue cup on the left of three red apples",
        "questions": [
            {
                "id": "relation:cup:apples:left_of",
                "category": "spatial_relation",
                "question": "Is the cup visibly to the left of the apples?",
                "source_constraint": {
                    "subject": "cup",
                    "object": "apples",
                    "relation": "left_of",
                },
            }
        ],
        "answers": [
            {
                "id": "relation:cup:apples:left_of",
                "normalized_answer": "yes",
                "passed": True,
            }
        ],
        "summary": {
            "passed": True,
            "score": 1.0,
            "hard_failures": 0,
            "failed_constraints": [],
            "passed_constraints": ["relation:cup:apples:left_of"],
            "blocked_constraints": [],
        },
        "constraint_check": {
            "passed": True,
            "score": 1.0,
            "constraint_score": 1.0,
            "checks": [
                {
                    "type": "relation",
                    "category": "spatial_relation",
                    "target": "cup:apples:left_of",
                    "expected": "left_of",
                    "observed": "yes",
                    "passed": True,
                    "question_id": "relation:cup:apples:left_of",
                }
            ],
            "errors": [],
            "question_summary": {
                "passed": True,
                "score": 1.0,
                "hard_failures": 0,
                "failed_constraints": [],
                "passed_constraints": ["relation:cup:apples:left_of"],
                "blocked_constraints": [],
            },
        },
    }


def test_geometry_override_fails_left_of_when_subject_bbox_is_right() -> None:
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "objects": [
                        {"name": "cup", "visible": True, "bbox": [520, 260, 120, 140]},
                        {"name": "apples", "visible": True, "bbox": [120, 260, 260, 150]},
                    ]
                }
            )
        ]
    )

    record = augment_record_with_object_geometry(
        vlm,
        _left_of_record(),
        user_prompt="a blue cup on the left of three red apples",
        prompt="a blue cup on the left of three red apples",
        image_path="mock://cup-right",
    )

    check = record["constraint_check"]
    assert check["passed"] is False
    assert check["question_summary"]["hard_failures"] == 1
    assert any(error["type"] == "wrong_relation" for error in check["errors"])
    assert check["geometry_source"] == "object_state_geometry"
    assert record["evidence_chain"][0]["status"] == "failed"
    assert record["evidence_chain"][0]["verify"]["evidence_source"] == "object_state_geometry"
    assert "Targets JSON" in vlm.calls[0]["prompt"]


def test_geometry_keeps_left_of_pass_when_subject_bbox_is_left() -> None:
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "objects": [
                        {"name": "cup", "visible": True, "bbox": [80, 260, 120, 140]},
                        {"name": "apples", "visible": True, "bbox": [340, 250, 260, 160]},
                    ]
                }
            )
        ]
    )

    record = augment_record_with_object_geometry(
        vlm,
        _left_of_record(),
        user_prompt="a blue cup on the left of three red apples",
        prompt="a blue cup on the left of three red apples",
        image_path="mock://cup-left",
    )

    check = record["constraint_check"]
    assert check["passed"] is True
    assert check["errors"] == []
    assert record["geometry_verification"]["passed"] is True
    assert record["constraint_check"]["evidence_chain"][0]["status"] == "passed"


def test_parse_object_state_keeps_parts_attributes_masks_and_protection() -> None:
    state = parse_object_state_response(
        json.dumps(
            {
                "objects": [
                    {
                        "name": "umbrella",
                        "visible": True,
                        "bbox": [10, 20, 100, 80],
                        "mask_path": "mask.png",
                        "attributes": {"color": "blue"},
                        "parts": [
                            {
                                "name": "umbrella handle",
                                "visible": True,
                                "bbox": [55, 80, 10, 60],
                                "evidence": "thin dark handle",
                            }
                        ],
                        "contact_regions": [
                            {
                                "name": "hand handle contact",
                                "visible": True,
                                "bbox": [50, 90, 24, 24],
                            }
                        ],
                        "protected": True,
                        "confidence": 0.91,
                    }
                ]
            }
        ),
        targets=["umbrella"],
        image_size=(200, 200),
    )

    umbrella = state["objects"][0]
    assert umbrella["attributes"]["color"] == "blue"
    assert umbrella["parts"][0]["name"] == "umbrella handle"
    assert umbrella["parts"][0]["bbox"] == [55, 80, 10, 60]
    assert umbrella["contact_regions"][0]["bbox"] == [50, 90, 24, 24]
    assert umbrella["mask_path"] == "mask.png"
    assert umbrella["protected"] is True


def test_evidence_chain_for_count_does_not_need_extra_vlm_call() -> None:
    record = {
        "questions": [
            {
                "id": "count:birds",
                "category": "count",
                "question": "How many visible birds are in the image?",
                "expected_answer": "2",
                "source_constraint": {"object": "birds", "count": 2},
            }
        ],
        "answers": [
            {
                "id": "count:birds",
                "normalized_answer": "2",
                "passed": True,
                "evidence": "Two distinct birds are visible.",
            }
        ],
        "constraint_check": {"passed": True, "checks": [], "errors": []},
    }
    vlm = MockVLMClient()

    augmented = augment_record_with_object_geometry(
        vlm,
        record,
        user_prompt="two yellow birds",
        prompt="two yellow birds",
        image_path="mock://birds",
    )

    assert vlm.calls == []
    assert augmented["evidence_chain"][0]["think"].startswith("Need to count")
    assert augmented["evidence_chain"][0]["look_request"]["evidence_type"] == "distinct_instance_count"
    assert augmented["evidence_chain"][0]["status"] == "passed"


def test_object_state_count_evidence_overrides_vqa_yes(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "birds.png"
    Image.new("RGB", (80, 80), (0, 0, 0)).save(image_path)
    record = {
        "questions": [
            {
                "id": "count:birds",
                "category": "count",
                "question": "How many visible birds are in the image?",
                "expected_answer": "2",
                "source_constraint": {"object": "birds", "count": 2},
            }
        ],
        "answers": [
            {
                "id": "count:birds",
                "normalized_answer": "2",
                "passed": True,
                "evidence": "The VLM answer says two birds are visible.",
            }
        ],
        "constraint_check": {
            "passed": True,
            "score": 1.0,
            "constraint_score": 1.0,
            "checks": [
                {
                    "type": "count",
                    "category": "count",
                    "target": "birds",
                    "expected": "2",
                    "observed": "2",
                    "passed": True,
                    "question_id": "count:birds",
                }
            ],
            "errors": [],
            "question_summary": {
                "passed": True,
                "score": 1.0,
                "hard_failures": 0,
                "failed_constraints": [],
                "passed_constraints": ["count:birds"],
                "blocked_constraints": [],
            },
        },
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "objects": [
                        {
                            "name": "birds",
                            "visible": True,
                            "bbox": [20, 20, 30, 20],
                            "count": 1,
                        }
                    ]
                }
            )
        ]
    )

    augmented = augment_record_with_object_geometry(
        vlm,
        record,
        user_prompt="two yellow birds",
        prompt="two yellow birds",
        image_path=str(image_path),
    )

    check = augmented["constraint_check"]
    assert check["passed"] is False
    assert check["object_evidence_source"] == "object_part_state_evidence"
    assert any(error["type"] == "wrong_count" for error in check["errors"])
    assert augmented["evidence_chain"][0]["status"] == "failed"
    assert (
        augmented["evidence_chain"][0]["verify"]["evidence_source"]
        == "object_part_state_evidence"
    )


def test_missing_object_state_does_not_override_vqa_count_pass(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "birds.png"
    Image.new("RGB", (80, 80), (0, 0, 0)).save(image_path)
    record = {
        "questions": [
            {
                "id": "count:birds",
                "category": "count",
                "question": "How many visible birds are in the image?",
                "expected_answer": "2",
                "source_constraint": {"object": "birds", "count": 2},
            }
        ],
        "answers": [
            {
                "id": "count:birds",
                "normalized_answer": "2",
                "passed": True,
                "evidence": "Two birds are visible.",
            }
        ],
        "constraint_check": {
            "passed": True,
            "score": 1.0,
            "constraint_score": 1.0,
            "checks": [
                {
                    "type": "count",
                    "category": "count",
                    "target": "birds",
                    "expected": "2",
                    "observed": "2",
                    "passed": True,
                    "question_id": "count:birds",
                }
            ],
            "errors": [],
            "question_summary": {
                "passed": True,
                "score": 1.0,
                "hard_failures": 0,
                "failed_constraints": [],
                "passed_constraints": ["count:birds"],
                "blocked_constraints": [],
            },
        },
    }
    vlm = MockVLMClient(responses=[json.dumps({"objects": []})])

    augmented = augment_record_with_object_geometry(
        vlm,
        record,
        user_prompt="two yellow birds",
        prompt="two yellow birds",
        image_path=str(image_path),
    )

    assert augmented["constraint_check"]["passed"] is True
    assert augmented["constraint_check"]["errors"] == []
    assert augmented["evidence_chain"][0]["status"] == "passed"


def test_pose_action_does_not_require_contact_region(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "wizard.png"
    Image.new("RGB", (80, 80), (0, 0, 0)).save(image_path)
    record = {
        "questions": [
            {
                "id": "action:wizard:standing",
                "category": "action_relation",
                "question": "Is the wizard visibly standing upright?",
                "expected_answer": "yes",
                "source_constraint": {"subject": "wizard", "action": "standing"},
            }
        ],
        "answers": [
            {
                "id": "action:wizard:standing",
                "normalized_answer": "yes",
                "passed": True,
                "evidence": "The wizard is upright on their feet.",
            }
        ],
        "constraint_check": {
            "passed": True,
            "score": 1.0,
            "checks": [
                {
                    "type": "relation",
                    "category": "action_relation",
                    "target": "wizard:standing",
                    "expected": "yes",
                    "observed": "yes",
                    "passed": True,
                    "question_id": "action:wizard:standing",
                }
            ],
            "errors": [],
            "question_summary": {
                "passed": True,
                "score": 1.0,
                "hard_failures": 0,
                "failed_constraints": [],
                "passed_constraints": ["action:wizard:standing"],
                "blocked_constraints": [],
            },
        },
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "objects": [
                        {"name": "wizard", "visible": True, "bbox": [20, 10, 30, 60]}
                    ]
                }
            )
        ]
    )

    augmented = augment_record_with_object_geometry(
        vlm,
        record,
        user_prompt="a green wizard standing beside a barrel",
        prompt="a green wizard standing beside a barrel",
        image_path=str(image_path),
    )

    assert augmented["constraint_check"]["passed"] is True
    assert augmented["constraint_check"]["errors"] == []
    assert augmented["object_evidence_verification"]["checks"] == []
    assert augmented["object_evidence_verification"]["errors"] == []


def test_build_evidence_chain_marks_blocked_dependency() -> None:
    chain = build_evidence_chain(
        [
            {
                "id": "color:umbrella",
                "category": "color_binding",
                "question": "What color is the umbrella?",
                "expected_answer": "blue",
                "depends_on": ["existence:umbrella"],
                "source_constraint": {"object": "umbrella", "attribute": "color"},
            }
        ],
        [
            {
                "id": "color:umbrella",
                "normalized_answer": "blocked",
                "passed": None,
                "blocked_by": ["existence:umbrella"],
            }
        ],
    )

    assert chain[0]["status"] == "blocked"
    assert chain[0]["look_result"]["blocked_by"] == ["existence:umbrella"]
