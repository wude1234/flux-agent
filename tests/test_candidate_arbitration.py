from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_arbitration import arbitrate_image_candidates, summarize_constraint_check


def test_arbitration_prefers_constraints_over_equal_reward() -> None:
    decision = arbitrate_image_candidates(
        image_paths=["image0.png", "image1.png", "image2.png"],
        prompts=["red umbrella", "blue umbrella", "blue background"],
        selection={"selected_index": 0},
        reward_ranking={
            "selected_index": 0,
            "scores": [
                {"index": 0, "score": 0.95},
                {"index": 1, "score": 0.95},
                {"index": 2, "score": 0.95},
            ],
        },
        candidate_checks=[
            {
                "index": 0,
                "constraint_check": {
                    "passed": False,
                    "score": 0.6,
                    "checks": [
                        {
                            "type": "color",
                            "target": "umbrella",
                            "passed": False,
                        }
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                },
            },
            {
                "index": 1,
                "constraint_check": {
                    "passed": True,
                    "score": 0.9,
                    "checks": [
                        {
                            "type": "color",
                            "target": "umbrella",
                            "passed": True,
                        },
                        {
                            "type": "color",
                            "target": "robot",
                            "passed": True,
                        },
                    ],
                    "errors": [],
                },
            },
            {
                "index": 2,
                "constraint_check": {
                    "passed": False,
                    "score": 0.7,
                    "checks": [
                        {
                            "type": "relation",
                            "target": "robot-umbrella",
                            "passed": False,
                        }
                    ],
                    "errors": [],
                },
            },
        ],
    )

    assert decision["selected_index"] == 1
    assert decision["overrode_reward_selection"] is True
    assert decision["ranking"][0]["constraint_summary"]["passed"] is True


def test_human_like_selector_prefers_color_binding_over_repairable_relation_gap() -> None:
    decision = arbitrate_image_candidates(
        image_paths=["red_umbrella_grip.png", "blue_umbrella_weak_grip.png"],
        prompts=["red umbrella", "blue umbrella"],
        selection={"selected_index": 0},
        reward_ranking={
            "selected_index": 0,
            "scores": [
                {"index": 0, "score": 0.95},
                {"index": 1, "score": 0.95},
            ],
        },
        constraints={
            "subjects": ["robot", "umbrella"],
            "colors": {"robot": "red", "umbrella": "blue"},
            "actions": ["gripping"],
            "relations": [],
            "protected_phrases": ["cinematic rainy street photo"],
        },
        candidate_checks=[
            {
                "index": 0,
                "constraint_check": {
                    "passed": True,
                    "score": 1.0,
                    "checks": [],
                    "errors": [],
                    "strengths": [
                        "A small red robot is visible in the rainy street.",
                        "The robot appears to grip an umbrella handle.",
                        "The cinematic rainy street photo atmosphere is strong.",
                    ],
                },
            },
            {
                "index": 1,
                "constraint_check": {
                    "passed": False,
                    "score": 0.82,
                    "checks": [
                        {
                            "type": "relation",
                            "target": "robot-umbrella",
                            "expected": "gripping handle",
                            "observed": "weak contact",
                            "passed": False,
                        }
                    ],
                    "errors": [
                        {
                            "type": "wrong_relation",
                            "prompt_span": "gripping handle",
                            "evidence": "The hand-handle contact is weak.",
                        }
                    ],
                    "strengths": [
                        "A small red robot is visible in the rainy street.",
                        "The umbrella canopy is clearly blue.",
                        "The cinematic rainy street photo atmosphere is strong.",
                    ],
                },
            },
        ],
    )

    assert decision["selected_index"] == 1
    assert decision["overrode_reward_selection"] is True
    assert decision["overrode_visual_selection"] is True
    assert decision["selection_policy"] == "m6211_rule_grounded_human_like_selector"
    trace = decision["selection_trace"]
    assert trace["selected_index"] == 1
    assert trace["candidates"][0]["tier"] == "repairable_action_or_relation_gap"
    assert trace["candidates"][1]["tier"] == "uncertain_user_attribute_binding"
    assert trace["candidates"][0]["index"] == 1
    assert trace["candidates"][1]["index"] == 0
    assert trace["candidates"][1]["missing_evidence"][0]["id"] == "color:umbrella:blue"


def test_summarize_constraint_check_counts_hard_failures() -> None:
    summary = summarize_constraint_check(
        {
            "passed": False,
            "score": 0.5,
            "checks": [
                {"type": "color", "target": "umbrella", "passed": False},
                {"type": "protected_phrase", "target": "cinematic", "passed": False},
                {"type": "subject", "target": "robot", "passed": True},
            ],
            "errors": [{"type": "wrong_relation", "prompt_span": "gripping handle"}],
        }
    )

    assert summary["passed_checks"] == 1
    assert summary["failed_checks"] == 2
    assert summary["hard_failures"] == 2
    assert summary["soft_failures"] == 1
    assert summary["failed_types"] == [
        "color",
        "protected_phrase",
        "wrong_relation",
    ]
    assert summary["major_failures"] == 1
    assert summary["repairable_failures"] == 1
    assert summary["minor_failures"] == 1
