from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_constraints import extract_constraints
from src.repair_base_selector import select_repair_base


USER_PROMPT = (
    "a small red robot clearly gripping the handle of a blue umbrella, "
    "cinematic rainy street photo"
)


def _blue_umbrella_weak_grip() -> dict:
    return {
        "passed": False,
        "score": 0.78,
        "checks": [
            {"type": "subject", "target": "robot", "expected": "small robot", "passed": True},
            {"type": "color", "target": "robot", "expected": "red", "passed": True},
            {"type": "subject", "target": "umbrella", "expected": "umbrella", "passed": True},
            {"type": "color", "target": "umbrella", "expected": "blue", "passed": True},
            {
                "type": "relation",
                "target": "robot-umbrella",
                "expected": "robot gripping umbrella handle",
                "observed": "robot near umbrella but grip is unclear",
                "passed": False,
            },
        ],
        "errors": [
            {
                "type": "wrong_relation",
                "prompt_span": "gripping handle",
                "evidence": "The robot is not clearly gripping the handle.",
            }
        ],
    }


def _pretty_brown_umbrella_better_reward() -> dict:
    return {
        "passed": False,
        "score": 0.74,
        "checks": [
            {"type": "subject", "target": "robot", "expected": "small robot", "passed": True},
            {"type": "color", "target": "robot", "expected": "red", "passed": True},
            {"type": "subject", "target": "umbrella", "expected": "umbrella", "passed": True},
            {
                "type": "color",
                "target": "umbrella",
                "expected": "blue",
                "observed": "brown red umbrella",
                "passed": False,
            },
            {
                "type": "relation",
                "target": "robot-umbrella",
                "expected": "robot gripping umbrella handle",
                "observed": "robot is closer to the handle",
                "passed": False,
            },
        ],
        "errors": [
            {
                "type": "wrong_attribute",
                "prompt_span": "blue umbrella",
                "evidence": "The umbrella is brown rather than blue.",
            },
            {
                "type": "wrong_relation",
                "prompt_span": "gripping handle",
                "evidence": "The grip is still unclear.",
            },
        ],
    }


def test_base_selector_prefers_color_correct_relation_weak_candidate() -> None:
    result = select_repair_base(
        image_paths=["sdxl_image_0000.png", "sdxl_image_0004.png"],
        prompts=[USER_PROMPT, USER_PROMPT],
        candidate_checks=[
            {"index": 0, "constraint_check": _blue_umbrella_weak_grip()},
            {"index": 1, "constraint_check": _pretty_brown_umbrella_better_reward()},
        ],
        constraints=extract_constraints(USER_PROMPT),
        repair_plan={"primary_action": "relation_repair", "target_object": "gripping"},
        reward_ranking={
            "selected_index": 1,
            "scores": [
                {"index": 0, "score": 0.62},
                {"index": 1, "score": 0.91},
            ],
        },
        current_index=1,
    )

    assert result["selected_index"] == 0
    assert result["overrode_current_selection"] is True
    assert result["intended_action"] == "relation_repair"
    assert result["ranking"][0]["hard_failures"] == 1
    assert result["ranking"][1]["hard_failures"] == 2
    assert "blue" in " ".join(result["satisfied_constraints"]).lower()
    assert result["rejected_candidates"][0]["index"] == 1
    assert "more hard user-constraint failures" in result["rejected_candidates"][0]["reason"]


def test_reward_is_only_late_tiebreaker_after_constraints_and_risk() -> None:
    result = select_repair_base(
        image_paths=["a.png", "b.png"],
        prompts=[USER_PROMPT, USER_PROMPT],
        candidate_checks=[
            {"index": 0, "constraint_check": _blue_umbrella_weak_grip()},
            {"index": 1, "constraint_check": _pretty_brown_umbrella_better_reward()},
        ],
        constraints=extract_constraints(USER_PROMPT),
        repair_plan={"primary_action": "relation_repair", "target_object": "gripping"},
        reward_ranking={
            "scores": [
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 1.0},
            ]
        },
    )

    assert result["selected_index"] == 0
    assert result["reward_score"] == 0.1


def test_vlm_pairwise_preference_can_break_near_tie_not_hard_constraints() -> None:
    class Judge:
        def judge(self, request, image_paths):
            assert "best base for local repair" in request
            assert image_paths == ["a.png", "b.png"]
            return {"selected_index": 1, "reason": "candidate 1 is easier to polish"}

    check = {
        "passed": True,
        "score": 0.9,
        "checks": [
            {"type": "color", "target": "robot", "expected": "red", "passed": True},
            {"type": "color", "target": "umbrella", "expected": "blue", "passed": True},
            {"type": "relation", "target": "robot-umbrella", "expected": "gripping", "passed": True},
        ],
        "errors": [],
    }
    result = select_repair_base(
        image_paths=["a.png", "b.png"],
        prompts=[USER_PROMPT, USER_PROMPT],
        candidate_checks=[
            {"index": 0, "constraint_check": check},
            {"index": 1, "constraint_check": check},
        ],
        constraints=extract_constraints(USER_PROMPT),
        repair_plan={"primary_action": "relation_repair", "target_object": "gripping"},
        vlm_judge=Judge(),
    )

    assert result["selected_index"] == 1
    assert result["vlm_pairwise"]["selected_index"] == 1
