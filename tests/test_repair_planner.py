import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.prompt_constraints import extract_constraints
from src.repair_planner import (
    RuleBasedRepairPlanner,
    build_repair_planning_request,
    heuristic_repair_plan,
    parse_repair_plan_response,
)


USER_PROMPT = (
    "a small red robot clearly gripping the handle of a blue umbrella, "
    "cinematic rainy street photo"
)


def test_heuristic_planner_routes_wrong_visible_color_to_recolor() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The umbrella is red instead of blue.",
                "prompt_span": "blue umbrella",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "recolor"
    assert plan["target_object"] == "umbrella"
    assert plan["target_color"] == "blue"


def test_heuristic_planner_stages_recolor_before_relation_repair() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The umbrella is red instead of blue.",
                "prompt_span": "blue umbrella",
            },
            {
                "type": "wrong_relation",
                "evidence": "The robot is not clearly gripping the umbrella handle.",
                "prompt_span": "clearly gripping handle",
            },
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "recolor"
    assert plan["tool_sequence"] == ["recolor", "relation_repair"]
    assert plan["target_object"] == "umbrella"


def test_heuristic_planner_routes_missing_subject_to_object_insertion() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The umbrella is present, but no robot is visible.",
                "prompt_span": "small red robot",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["target_object"] == "robot"


def test_heuristic_planner_does_not_insert_target_already_present() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "constraint_check": {
            "passed": False,
            "question_summary": {
                "passed_constraints": [
                    "existence:robot",
                    "existence:umbrella",
                    "color:robot",
                    "color:umbrella",
                ],
                "failed_constraints": ["action:robot:clearly_gripping"],
            },
            "checks": [
                {
                    "question_id": "existence:robot",
                    "category": "entity_existence",
                    "target": "robot",
                    "passed": True,
                },
                {
                    "question_id": "existence:umbrella",
                    "category": "entity_existence",
                    "target": "umbrella",
                    "passed": True,
                },
                {
                    "question_id": "action:robot:clearly_gripping",
                    "category": "action_relation",
                    "target": "robot:clearly gripping",
                    "passed": False,
                    "description": "The robot is not clearly gripping the umbrella handle.",
                },
            ],
            "errors": [
                {
                    "type": "wrong_relation",
                    "question_id": "action:robot:clearly_gripping",
                    "prompt_span": "clearly gripping",
                    "evidence": "The robot is not clearly gripping the umbrella handle.",
                }
            ],
        },
        "errors": [
            {
                "type": "missing_object",
                "prompt_span": "robot",
                "evidence": "A stale candidate said no robot is visible.",
            },
            {
                "type": "wrong_relation",
                "prompt_span": "clearly gripping",
                "evidence": "The robot is not clearly gripping the umbrella handle.",
            },
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "relation_repair"
    assert plan["typed_route"] == "relation_contact_repair"
    assert plan["target_attribute"] == "interaction_relation"


def test_heuristic_planner_routes_simple_under_count_to_object_insertion() -> None:
    constraints = extract_constraints(
        "three red apples on a wooden table"
    )
    critique = {
        "constraint_check": {
            "passed": False,
            "checks": [
                {
                    "category": "count",
                    "question_id": "count:apples",
                    "target": "apples",
                    "expected": "3",
                    "observed": "2",
                    "passed": False,
                    "type": "wrong_count",
                    "description": "There are two red apples visible.",
                }
            ],
            "errors": [
                {
                    "type": "wrong_count",
                    "question_id": "count:apples",
                    "prompt_span": "apples",
                    "evidence": "There are two red apples visible.",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "count_aware_regeneration"
    assert plan["target_object"] == "apples"
    assert plan["target_attribute"] == "count"
    assert plan["expected_count"] == 3
    assert plan["observed_count"] == 2


def test_heuristic_planner_routes_relation_bound_under_count_to_regenerate() -> None:
    constraints = extract_constraints(
        "two yellow birds sitting on a black bicycle near a white dog"
    )
    critique = {
        "constraint_check": {
            "passed": False,
            "checks": [
                {
                    "category": "count",
                    "question_id": "count:birds",
                    "target": "birds",
                    "expected": "2",
                    "observed": "1",
                    "passed": False,
                    "type": "wrong_count",
                    "description": "There is one yellow bird visible.",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "count_aware_regeneration"
    assert plan["target_object"] == "birds"
    assert plan["target_attribute"] == "count"


def test_heuristic_planner_parses_simple_under_count_from_evidence_text() -> None:
    constraints = extract_constraints(
        "three red apples on a wooden table"
    )
    critique = {
        "constraint_check": {
            "passed": False,
            "checks": [
                {
                    "category": "count",
                    "question_id": "count:apples",
                    "target": "apples",
                    "expected": "3",
                    "observed": "",
                    "passed": False,
                    "type": "wrong_count",
                    "description": "Only two red apples are visible.",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "count_aware_regeneration"
    assert plan["target_object"] == "apples"
    assert plan["expected_count"] == 3
    assert plan["observed_count"] == 2


def test_heuristic_planner_does_not_insert_when_count_is_satisfied() -> None:
    constraints = extract_constraints(
        "two yellow birds sitting on a black bicycle near a white dog"
    )
    critique = {
        "constraint_check": {
            "passed": True,
            "checks": [
                {
                    "category": "count",
                    "question_id": "count:birds",
                    "target": "birds",
                    "expected": "2",
                    "observed": "2",
                    "passed": True,
                    "type": "count",
                }
            ],
            "errors": [],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "none"


def test_heuristic_planner_falls_back_when_under_count_insertion_disabled() -> None:
    constraints = extract_constraints(
        "three red apples on a wooden table"
    )
    critique = {
        "constraint_check": {
            "passed": False,
            "checks": [
                {
                    "category": "count",
                    "question_id": "count:apples",
                    "target": "apples",
                    "expected": "3",
                    "observed": "2",
                    "passed": False,
                    "type": "wrong_count",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": False,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "count_aware_regeneration"
    assert "count-aware" in plan["reason"]


def test_heuristic_planner_routes_over_count_to_count_regeneration() -> None:
    constraints = extract_constraints(
        "Exactly four blue fish swim through one orange hoop, with no fifth fish "
        "and no extra hoop."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_count",
                "target": "fish",
                "expected": "4",
                "observed": "5",
                "evidence": "There are five blue fish visible.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "count_aware_regeneration"
    assert plan["target_object"] == "fish"
    assert plan["expected_count"] == 4
    assert plan["observed_count"] == 5


def test_heuristic_planner_routes_forbidden_object_to_local_removal() -> None:
    constraints = extract_constraints(
        "A yellow cereal box stands on a counter with no bowl nearby."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "target": "bowl",
                "evidence": "An extra bowl is visible, violating the no bowl constraint.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "forbidden_object_removal"
    assert plan["target_object"] == "bowl"
    assert plan["target_attribute"] == "forbidden_object"


def test_heuristic_planner_routes_forbidden_symbol_to_symbol_removal() -> None:
    constraints = extract_constraints(
        "A red box shows a white star symbol on its lid, next to a plain green box with no symbol."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "target": "green box",
                "evidence": "The plain green box contains a white star symbol, violating no symbol.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "forbidden_symbol_removal"
    assert plan["target_object"] == "green box"
    assert plan["target_attribute"] == "forbidden_symbol"


def test_heuristic_planner_routes_wrong_exact_text_to_overlay() -> None:
    constraints = extract_constraints(
        "A black sign displays the exact yellow text 'NO' above a plain blue sign with no text."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_exact_text",
                "target": "black sign",
                "evidence": "The black sign reads 'MO' instead of the exact text 'NO'.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "exact_text_overlay"
    assert plan["target_object"] == "black sign"
    assert plan["exact_text"] == "NO"


def test_heuristic_planner_does_not_confuse_missing_occluder_with_forbidden_object() -> None:
    constraints = extract_constraints(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    critique = {
        "constraint_check": {
            "errors": [
                {
                    "type": "wrong_relation",
                    "target": "screen",
                    "prompt_span": "red screen hides the lower half",
                    "evidence": "No red screen is visible occluding the suitcase.",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "occlusion_object_insertion"
    assert plan["target_object"] == "screen"


def test_heuristic_planner_routes_spatial_failure_to_layout_regeneration() -> None:
    constraints = extract_constraints(
        "A gray dog stands behind a teal bench, while a pink ball rests under the bench."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_relation",
                "target": "bench",
                "evidence": "The pink ball is resting on top of the teal bench, not under it.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "layout_guided_regeneration"
    assert plan["target_attribute"] == "spatial_relation"
    assert "ball under bench" in plan["target_object"]


def test_heuristic_planner_routes_occlusion_failure_to_early_object_insertion() -> None:
    constraints = extract_constraints(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    critique = {
        "constraint_check": {
            "passed": False,
            "checks": [
                {
                    "question_id": "existence:suitcase",
                    "category": "entity_existence",
                    "target": "suitcase",
                    "passed": True,
                    "description": "A green suitcase is visible.",
                },
                {
                    "question_id": "relation:screen:suitcase:hides_lower_half",
                    "category": "occlusion_relation",
                    "target": "screen",
                    "passed": False,
                    "description": "No red screen hides the lower half of the suitcase.",
                },
            ],
            "errors": [
                {
                    "type": "wrong_relation",
                    "target": "screen",
                    "prompt_span": "red screen hides the lower half",
                    "evidence": "There is no visible red screen occluding the suitcase.",
                }
            ],
        },
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "occlusion_object_insertion"
    assert plan["edit_timing"] == "early_edit"
    assert plan["target_object"] == "screen"
    assert plan["target_region"] == "lower_half"
    assert plan["occlusion_spec"]["target"] == "suitcase"


def test_heuristic_planner_does_not_treat_spatial_instead_of_as_recolor() -> None:
    constraints = extract_constraints(
        "A gray dog stands behind a teal bench, while a pink ball rests under the bench."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_spatial_relation",
                "target": "dog behind bench",
                "evidence": "The dog is sitting on the bench instead of behind it.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "layout_guided_regeneration"
    assert plan["target_attribute"] == "spatial_relation"


def test_heuristic_planner_routes_material_failure_before_recolor() -> None:
    constraints = extract_constraints(
        "A turquoise wooden chair, a crimson glass lamp, and a silver paper fan "
        "sit on a black rug."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_material",
                "target": "paper fan",
                "expected": "paper",
                "observed": "metal electric fan",
                "evidence": "The fan looks like a metallic electric fan, not a paper fan.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "material_guided_regeneration"
    assert plan["target_object"] == "paper fan"
    assert plan["target_attribute"] == "paper"


def test_heuristic_planner_routes_single_attribute_to_typed_patch() -> None:
    constraints = extract_constraints(
        "A turquoise wooden chair sits alone on a plain black floor."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "target": "chair",
                "evidence": "The chair is blue instead of turquoise.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "recolor"
    assert plan["typed_route"] == "single_attribute_patch"
    assert plan["target_object"] == "chair"
    assert plan["target_attribute"] == "color"


def test_heuristic_planner_routes_local_contact_relation_to_repair() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "wrong_relation",
                "target": "robot gripping handle",
                "evidence": "The robot hand is near the umbrella handle but not clearly touching it.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "relation_repair"
    assert plan["typed_route"] == "relation_contact_repair"
    assert plan["preconditions"]["relation_locally_repairable"] is True


def test_heuristic_planner_routes_missing_contact_parts_to_relation_regeneration() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "wrong_relation",
                "target": "robot gripping handle",
                "evidence": "The umbrella handle is not visible and the robot hand is hidden.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "relation_focused_regeneration"
    assert plan["preconditions"]["relation_locally_repairable"] is False


def test_heuristic_planner_routes_broad_multi_failure_to_decomposition() -> None:
    constraints = extract_constraints(
        "Exactly two cyan ceramic mugs are left of one orange wooden tray, "
        "and a purple spoon lies under the tray; no extra mug or fork is present."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_count",
                "target": "mugs",
                "expected": "2",
                "observed": "3",
                "evidence": "There are three cyan mugs visible.",
            },
            {
                "type": "wrong_spatial_relation",
                "target": "mugs left of tray",
                "evidence": "The mugs are right of the tray, not left of it.",
            },
            {
                "type": "wrong_attribute",
                "target": "spoon",
                "evidence": "The spoon is blue instead of purple.",
            },
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "multi_constraint_decompose"
    assert plan["target_attribute"] == "multi_constraint"


def test_heuristic_planner_promotes_leather_plastic_evidence_to_material_route() -> None:
    constraints = extract_constraints(
        "A turquoise wooden chair sits alone on a plain black floor; the chair "
        "must be turquoise and visibly wooden."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "target": "chair",
                "prompt_span": "wooden",
                "evidence": "The chair appears to be made of leather or plastic rather than wood.",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["typed_route"] == "material_guided_regeneration"
    assert plan["target_object"] == "chair"
    assert plan["target_attribute"] == "wooden"


def test_vlm_recolor_plan_is_overridden_for_material_failure() -> None:
    constraints = extract_constraints(
        "A turquoise wooden chair, a crimson glass lamp, and a silver paper fan "
        "sit on a black rug."
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "recolor",
                    "tool_sequence": ["recolor"],
                    "repairable": True,
                    "target_object": "paper fan",
                    "target_attribute": "color",
                    "reason": "Make the fan silver.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=constraints.original_prompt,
        prompt=constraints.original_prompt,
        image_path="/tmp/fake.png",
        critique={
            "errors": [
                {
                    "type": "wrong_material",
                    "target": "paper fan",
                    "expected": "paper",
                    "evidence": "The fan is a metallic electric fan, not paper.",
                }
            ]
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["fallback_from"] == "recolor"
    assert plan["typed_route"] == "material_guided_regeneration"


def test_heuristic_planner_routes_unrepairable_relation_to_regenerate() -> None:
    constraints = extract_constraints(USER_PROMPT)
    critique = {
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": "The handle is not visible and the hand is hidden.",
                "prompt_span": "gripping the handle",
            }
        ],
        "user_grounded": True,
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["repairable"] is False


def test_vlm_planner_can_choose_object_insertion() -> None:
    constraints = extract_constraints(USER_PROMPT)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "object_insertion",
                    "tool_sequence": ["object_insertion"],
                    "repairable": True,
                    "target_object": "robot",
                    "target_attribute": "presence",
                    "reason": "Umbrella exists but the robot is missing.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        image_path="/tmp/fake.png",
        critique={"errors": [{"type": "missing_object", "evidence": "No robot."}]},
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["source"] == "vlm_repair_planner"
    assert "Analyze the already generated image" in vlm.calls[0]["prompt"]


def test_vlm_object_insertion_plan_is_overridden_when_target_present() -> None:
    constraints = extract_constraints(USER_PROMPT)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "object_insertion",
                    "tool_sequence": ["object_insertion"],
                    "repairable": True,
                    "target_object": "robot",
                    "target_attribute": "presence",
                    "reason": "The robot appears missing.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        image_path="/tmp/fake.png",
        critique={
            "constraint_check": {
                "passed": False,
                "question_summary": {
                    "passed_constraints": ["existence:robot", "existence:umbrella"],
                    "failed_constraints": ["action:robot:clearly_gripping"],
                },
                "checks": [
                    {
                        "question_id": "existence:robot",
                        "category": "entity_existence",
                        "target": "robot",
                        "passed": True,
                    },
                    {
                        "question_id": "action:robot:clearly_gripping",
                        "category": "action_relation",
                        "target": "robot:clearly gripping",
                        "passed": False,
                        "description": "The robot is not clearly gripping the handle.",
                    },
                ],
                "errors": [
                    {
                        "type": "wrong_relation",
                        "question_id": "action:robot:clearly_gripping",
                        "evidence": "The robot is not clearly gripping the handle.",
                    }
                ],
            },
            "user_grounded": True,
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "relation_repair"
    assert plan["fallback_from"] == "object_insertion"
    assert plan["preconditions"]["planner_override"] == "target_presence_verified"


def test_repair_planning_request_compacts_current_feedback_only() -> None:
    constraints = extract_constraints(USER_PROMPT)
    request = build_repair_planning_request(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        image_path="/tmp/fake.png",
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
        critique={
            "score": 0.4,
            "raw_response": "x" * 20000,
            "request": "y" * 20000,
            "constraint_arbitration": {
                "candidate_checks": [
                    {
                        "constraint_check": {
                            "errors": [
                                {
                                    "type": "missing_object",
                                    "evidence": "No robot in a different candidate.",
                                }
                            ],
                            "raw_response": "z" * 20000,
                        }
                    }
                ]
            },
            "constraint_check": {
                "passed": False,
                "source": "question_level_vqa",
                "errors": [
                    {
                        "type": "wrong_relation",
                        "evidence": "The selected robot is not gripping the handle.",
                    }
                ],
            },
            "errors": [
                {
                    "type": "wrong_relation",
                    "evidence": "The selected robot is not gripping the handle.",
                }
            ],
        },
    )

    assert len(request) < 25000
    assert "No robot in a different candidate" not in request
    assert "raw_response" not in request
    assert "The selected robot is not gripping the handle" in request


def test_vlm_relation_plan_is_overridden_when_color_binding_still_fails() -> None:
    constraints = extract_constraints(USER_PROMPT)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "relation_repair",
                    "tool_sequence": ["relation_repair"],
                    "repairable": True,
                    "target_object": "gripping",
                    "target_attribute": "relation",
                    "reason": "The grip is weak.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=USER_PROMPT,
        prompt=USER_PROMPT,
        image_path="/tmp/fake.png",
        critique={
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The umbrella is red instead of blue.",
                    "prompt_span": "blue umbrella",
                },
                {
                    "type": "wrong_relation",
                    "evidence": "The robot is not clearly gripping the handle.",
                    "prompt_span": "clearly gripping handle",
                },
            ]
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "recolor"
    assert plan["tool_sequence"] == ["recolor", "relation_repair"]
    assert plan["fallback_from"] == "relation_repair"
    assert plan["target_object"] == "umbrella"


def test_vlm_planner_high_risk_plural_relation_insertion_is_overridden() -> None:
    user_prompt = "two yellow birds sitting on a black bicycle near a white dog"
    constraints = extract_constraints(user_prompt)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "object_insertion",
                    "tool_sequence": ["object_insertion"],
                    "repairable": True,
                    "target_object": "birds",
                    "target_attribute": "presence",
                    "reason": "The birds are missing.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=user_prompt,
        prompt=user_prompt,
        image_path="/tmp/fake.png",
        critique={
            "constraint_check": {
                "errors": [
                    {
                        "type": "missing_object",
                        "question_id": "existence:birds",
                        "prompt_span": "birds",
                        "evidence": "There are no visible birds.",
                    }
                ]
            }
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["fallback_from"] == "object_insertion"
    assert plan["preconditions"]["planner_override"] == "relation_bound_plural_insertion"
    assert plan["source"] == "vlm_repair_planner"


def test_vlm_occlusion_object_insertion_is_typed_for_bbox_editing() -> None:
    user_prompt = (
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    constraints = extract_constraints(user_prompt)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "object_insertion",
                    "tool_sequence": ["object_insertion"],
                    "repairable": True,
                    "target_object": "screen",
                    "target_attribute": "presence",
                    "reason": "The red screen is missing.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=user_prompt,
        prompt=user_prompt,
        image_path="/tmp/fake.png",
        critique={
            "constraint_check": {
                "passed": False,
                "checks": [
                    {
                        "question_id": "existence:suitcase",
                        "category": "entity_existence",
                        "target": "suitcase",
                        "passed": True,
                        "description": "A green suitcase is visible.",
                    },
                    {
                        "question_id": "relation:screen:suitcase:hides_lower_half",
                        "category": "occlusion_relation",
                        "target": "screen",
                        "passed": False,
                        "description": "No red screen hides the lower half of the suitcase.",
                    },
                ],
                "errors": [
                    {
                        "type": "missing_object",
                        "target": "screen",
                        "prompt_span": "red screen",
                        "evidence": "There is no visible red screen occluding the suitcase.",
                    }
                ],
            }
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": False,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "occlusion_object_insertion"
    assert plan["target_object"] == "screen"
    assert plan["target_attribute"] == "occlusion"
    assert plan["target_region"] == "lower_half"
    assert plan["occlusion_spec"]["target"] == "suitcase"
    assert plan["preconditions"]["planner_override"] == "typed_occlusion_object_insertion"
    assert plan["source"] == "vlm_repair_planner"


def test_vlm_occlusion_regenerate_is_overridden_to_typed_edit() -> None:
    user_prompt = (
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    constraints = extract_constraints(user_prompt)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "primary_action": "regenerate",
                    "tool_sequence": ["regenerate"],
                    "repairable": True,
                    "target_object": "screen",
                    "target_attribute": "presence",
                    "reason": "The red screen is missing; regenerate the image.",
                }
            )
        ]
    )
    planner = RuleBasedRepairPlanner(vlm)

    plan = planner.plan(
        user_prompt=user_prompt,
        prompt=user_prompt,
        image_path="/tmp/fake.png",
        critique={
            "constraint_check": {
                "passed": False,
                "checks": [
                    {
                        "question_id": "existence:suitcase",
                        "category": "entity_existence",
                        "target": "suitcase",
                        "passed": True,
                        "description": "A green suitcase is visible.",
                    },
                    {
                        "question_id": "relation:screen:suitcase:hides_lower_half",
                        "category": "occlusion_relation",
                        "target": "screen",
                        "passed": False,
                        "description": "No red screen hides the lower half of the suitcase.",
                    },
                ],
                "errors": [
                    {
                        "type": "missing_object",
                        "target": "screen",
                        "prompt_span": "red screen",
                        "evidence": "There is no visible red screen occluding the suitcase.",
                    }
                ],
            },
            "user_grounded": True,
        },
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": True,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "object_insertion"
    assert plan["typed_route"] == "occlusion_object_insertion"
    assert plan["target_object"] == "screen"
    assert plan["target_attribute"] == "occlusion"
    assert plan["fallback_from"] == "regenerate"
    assert plan["source_plan"]["primary_action"] == "regenerate"
    assert plan["preconditions"]["planner_override"] == "typed_occlusion_object_insertion"
    assert plan["source"] == "vlm_repair_planner"


def test_parse_repair_plan_falls_back_when_tool_unavailable() -> None:
    constraints = extract_constraints(USER_PROMPT)
    plan = parse_repair_plan_response(
        json.dumps({"primary_action": "object_insertion", "target_object": "robot"}),
        constraints=constraints,
        enabled_tools={
            "recolor": True,
            "relation_repair": True,
            "object_insertion": False,
            "regenerate": True,
        },
    )

    assert plan["primary_action"] == "regenerate"
    assert plan["fallback_from"] == "object_insertion"


def test_planner_routes_comparative_count_to_rerank() -> None:
    constraints = extract_constraints("A pencil holder with more pens than pencils.")
    critique = {
        "errors": [
            {
                "type": "wrong_count",
                "evidence": "The pencil holder contains more pencils than pens.",
                "prompt_span": "more pens than pencils",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "comparative_count_rerank"
    assert plan["primary_action"] == "regenerate"


def test_planner_routes_comparative_attribute_binding() -> None:
    constraints = extract_constraints(
        "A larger person in yellow clothing and a smaller person in a different color."
    )
    critique = {
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The smaller person is wearing the same yellow color as the larger person.",
                "prompt_span": "smaller person different color",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "comparative_attribute_binding"
    assert plan["primary_action"] == "regenerate"


def test_planner_routes_role_action_binding_not_forbidden_removal() -> None:
    constraints = extract_constraints(
        "The girl with glasses is drawing, and the girl without glasses is singing."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The girl without glasses is depicted as drawing rather than singing.",
                "prompt_span": "girl without glasses is singing",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "role_action_binding_regeneration"
    assert plan["typed_route"] != "forbidden_object_removal"
    assert plan["primary_action"] == "regenerate"


def test_planner_routes_normalizable_misspelling_to_lexical_grounding() -> None:
    constraints = extract_constraints("Bzaseball galove.")
    critique = {
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The image is entirely unrelated and does not match Bzaseball galove.",
                "prompt_span": "Bzaseball galove",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "lexical_grounding_regeneration"
    assert plan["primary_action"] == "regenerate"
    assert plan["normalized_prompt"] == "Baseball glove."


def test_planner_normalizes_drawbench_misspelling_before_generic_missing_object() -> None:
    constraints = extract_constraints("Tcennis rpacket.")
    critique = {
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The generated image is a circuit board and is unrelated.",
                "prompt_span": "Tcennis rpacket",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "lexical_grounding_regeneration"
    assert plan["primary_action"] == "regenerate"
    assert plan["normalized_prompt"] == "tennis racket."
    assert plan["preconditions"]["prompt_preflight"] is True


def test_planner_marks_single_unverifiable_rare_word_for_clarification() -> None:
    constraints = extract_constraints("Acersecomicke.")
    critique = {
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The generated image is unrelated to the requested rare term.",
                "prompt_span": "Acersecomicke",
            }
        ]
    }

    plan = heuristic_repair_plan(
        critique,
        constraints=constraints,
        enabled_tools={"recolor": True, "relation_repair": True, "object_insertion": True, "regenerate": True},
    )

    assert plan["typed_route"] == "unverifiable_rare_word_or_clarify"
    assert plan["primary_action"] == "none"
    assert plan["preconditions"]["needs_clarification"] is True


def test_vlm_planner_missing_object_is_overridden_by_lexical_preflight() -> None:
    constraints = extract_constraints("Tcennis rpacket.")
    planner = RuleBasedRepairPlanner(
        MockVLMClient(
            responses=[
                json.dumps(
                    {
                        "primary_action": "regenerate",
                        "repairable": False,
                        "target_object": "Tcennis rpacket",
                        "reason": "The image is missing the requested object.",
                    }
                )
            ]
        )
    )

    plan = planner.plan(
        user_prompt="Tcennis rpacket.",
        prompt="Tcennis rpacket.",
        image_path="/tmp/mock.jpg",
        critique={
            "errors": [
                {
                    "type": "missing_object",
                    "evidence": "The generated image is unrelated.",
                    "prompt_span": "Tcennis rpacket",
                }
            ]
        },
        constraints=constraints,
        enabled_tools={"recolor": False, "relation_repair": False, "object_insertion": False, "regenerate": True},
    )

    assert plan["source"] == "vlm_repair_planner"
    assert plan["typed_route"] == "lexical_grounding_regeneration"
    assert plan["normalized_prompt"] == "tennis racket."
    assert plan["preconditions"]["planner_override"] in {
        "prompt_preflight_route",
        "typed_failure_route",
    }
