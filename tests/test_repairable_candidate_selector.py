from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_constraints import extract_constraints
from src.repairable_candidate_selector import (
    build_repairability_profile,
    select_repairable_candidate,
)


USER_PROMPT = (
    "a small red robot clearly gripping the handle of a blue umbrella, "
    "cinematic rainy street photo"
)


def _color_only_wrong_umbrella_check() -> dict:
    return {
        "passed": False,
        "score": 0.6,
        "checks": [
            {
                "type": "color",
                "target": "robot",
                "expected": "red",
                "observed": "red",
                "passed": True,
            },
            {
                "type": "color",
                "target": "umbrella",
                "expected": "blue",
                "observed": "red",
                "passed": False,
            },
            {
                "type": "subject",
                "target": "robot",
                "expected": "small red robot",
                "observed": "small red robot",
                "passed": True,
            },
            {
                "type": "subject",
                "target": "handle",
                "expected": "handle of a blue umbrella",
                "observed": "handle of a red umbrella",
                "passed": False,
            },
            {
                "type": "action",
                "target": "gripping",
                "expected": "clearly gripping",
                "observed": "clearly gripping",
                "passed": True,
            },
            {
                "type": "relation",
                "target": "robot-umbrella",
                "expected": "robot gripping the handle of the umbrella",
                "observed": "robot gripping the handle of the umbrella",
                "passed": True,
            },
        ],
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The umbrella is red instead of blue.",
                "prompt_span": "umbrella",
            },
            {
                "type": "wrong_attribute",
                "evidence": "The handle is attached to a red umbrella rather than a blue one.",
                "prompt_span": "handle",
            },
        ],
    }


def _passed_but_not_repair_target_check() -> dict:
    return {
        "passed": True,
        "score": 1.0,
        "checks": [
            {
                "type": "color-object binding",
                "target": "robot",
                "expected": "red",
                "observed": "red",
                "passed": True,
            },
            {
                "type": "color-object binding",
                "target": "umbrella",
                "expected": "blue",
                "observed": "blue",
                "passed": True,
            },
            {
                "type": "subject",
                "target": "handle",
                "expected": "handle",
                "observed": "handle",
                "passed": True,
            },
            {
                "type": "action",
                "target": "clearly gripping",
                "expected": "clearly gripping",
                "observed": "clearly gripping",
                "passed": True,
            },
        ],
        "errors": [],
    }


def test_selector_keeps_current_when_current_feedback_is_equally_repairable() -> None:
    constraints = extract_constraints(USER_PROMPT)
    repair_plan = {
        "primary_action": "recolor",
        "target_object": "umbrella",
        "target_attribute": "color",
        "tool_sequence": ["recolor"],
    }
    arbitration = {
        "selected_index": 1,
        "selected_image": "image_0001.png",
        "candidate_checks": [
            {
                "index": 0,
                "image_path": "image_0000.png",
                "prompt": USER_PROMPT,
                "constraint_check": _color_only_wrong_umbrella_check(),
            },
            {
                "index": 1,
                "image_path": "image_0001.png",
                "prompt": USER_PROMPT,
                "constraint_check": _passed_but_not_repair_target_check(),
            },
            {
                "index": 2,
                "image_path": "image_0002.png",
                "prompt": USER_PROMPT,
                "constraint_check": {
                    "passed": False,
                    "score": 0.4,
                    "checks": [
                        {
                            "type": "subject",
                            "target": "umbrella",
                            "expected": "blue umbrella",
                            "observed": "not present",
                            "passed": False,
                        },
                        {
                            "type": "relation",
                            "target": "robot and umbrella",
                            "expected": "robot gripping umbrella",
                            "observed": "no relation",
                            "passed": False,
                        },
                    ],
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The blue umbrella is missing.",
                            "prompt_span": "umbrella",
                        },
                        {
                            "type": "wrong_relation",
                            "evidence": "There is no relation between the robot and umbrella.",
                            "prompt_span": "robot and umbrella",
                        },
                    ],
                },
            },
        ],
    }

    selected = select_repairable_candidate(
        arbitration=arbitration,
        current_index=1,
        critique={
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The umbrella is red instead of blue.",
                    "prompt_span": "blue umbrella",
                }
            ]
        },
        repair_plan=repair_plan,
        constraints=constraints,
    )

    assert selected is None


def test_selector_does_not_switch_when_current_candidate_is_best_for_recolor() -> None:
    constraints = extract_constraints(USER_PROMPT)
    repair_plan = {"primary_action": "recolor", "target_object": "umbrella"}
    arbitration = {
        "selected_index": 0,
        "candidate_checks": [
            {
                "index": 0,
                "image_path": "image_0000.png",
                "prompt": USER_PROMPT,
                "constraint_check": _color_only_wrong_umbrella_check(),
            },
            {
                "index": 1,
                "image_path": "image_0001.png",
                "prompt": USER_PROMPT,
                "constraint_check": _passed_but_not_repair_target_check(),
            },
        ],
    }

    assert (
        select_repairable_candidate(
            arbitration=arbitration,
            current_index=0,
            critique={},
            repair_plan=repair_plan,
            constraints=constraints,
        )
        is None
    )


def test_selector_overrides_object_insertion_when_recolorable_candidate_exists() -> None:
    constraints = extract_constraints(USER_PROMPT)
    repair_plan = {
        "primary_action": "object_insertion",
        "target_object": "umbrella",
        "target_attribute": "presence",
        "tool_sequence": ["object_insertion"],
    }
    arbitration = {
        "selected_index": 1,
        "candidate_checks": [
            {
                "index": 0,
                "image_path": "recolorable.png",
                "prompt": USER_PROMPT,
                "constraint_check": _color_only_wrong_umbrella_check(),
            },
            {
                "index": 1,
                "image_path": "missing.png",
                "prompt": USER_PROMPT,
                "constraint_check": {
                    "passed": False,
                    "score": 0.4,
                    "checks": [
                        {
                            "type": "subject",
                            "target": "umbrella",
                            "expected": "blue umbrella",
                            "observed": "not present",
                            "passed": False,
                        }
                    ],
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The umbrella is missing.",
                            "prompt_span": "umbrella",
                        }
                    ],
                },
            },
        ],
    }

    selected = select_repairable_candidate(
        arbitration=arbitration,
        current_index=1,
        critique={},
        repair_plan=repair_plan,
        constraints=constraints,
    )

    assert selected is not None
    assert selected["selected_index"] == 0
    assert selected["primary_action"] == "recolor"
    assert selected["repair_plan_override"]["primary_action"] == "recolor"
    assert selected["repair_plan_override"]["target_attribute"] == "color"


def test_relation_repair_profile_penalizes_missing_related_object() -> None:
    constraints = extract_constraints(USER_PROMPT)
    profile = build_repairability_profile(
        {
            "index": 0,
            "image_path": "bad.png",
            "prompt": USER_PROMPT,
            "constraint_check": {
                "passed": False,
                "score": 0.5,
                "checks": [
                    {
                        "type": "subject",
                        "target": "umbrella",
                        "expected": "blue umbrella",
                        "observed": "not present",
                        "passed": False,
                    },
                    {
                        "type": "relation",
                        "target": "robot and umbrella",
                        "expected": "gripping",
                        "observed": "no relation",
                        "passed": False,
                    },
                ],
                "errors": [
                    {
                        "type": "missing_object",
                        "evidence": "The umbrella is missing.",
                        "prompt_span": "umbrella",
                    }
                ],
            },
        },
        fallback_index=0,
        repair_plan={"primary_action": "relation_repair", "target_object": "gripping"},
        constraints=constraints,
    )

    assert profile["repairability"]["repairable"] is False
    assert "related objects are missing" in " ".join(
        profile["repairability"]["penalties"]
    )
