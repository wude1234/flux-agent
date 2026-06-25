import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.constraint_questions import (
    VQAConstraintEvaluator,
    build_vqa_constraint_request,
    generate_constraint_questions,
)
from src.prompt_constraints import extract_constraints


def test_generates_multi_subject_questions_existence_before_count_and_relations() -> None:
    constraints = extract_constraints(
        "two yellow birds sitting on a black bicycle near a white dog"
    )

    questions = generate_constraint_questions(constraints)
    categories = [question.category for question in questions]
    ids = [question.id for question in questions]

    assert categories[:3] == [
        "entity_existence",
        "entity_existence",
        "entity_existence",
    ]
    assert ids[:3] == ["existence:birds", "existence:bicycle", "existence:dog"]
    assert ids.index("count:birds") > ids.index("existence:birds")
    assert ids.index("color:birds") > ids.index("existence:birds")
    assert ids.index("color:bicycle") > ids.index("existence:bicycle")
    assert ids.index("color:dog") > ids.index("existence:dog")
    assert "relation:birds:bicycle:on" in ids
    assert "relation:bicycle:dog:near" in ids
    assert "relation:birds:dog:on" not in ids
    bicycle_color_question = next(question for question in questions if question.id == "color:bicycle")
    assert "main visible body or surface" in bicycle_color_question.question
    assert "Ignore small accessories" in bicycle_color_question.question
    assert all(
        ids.index(question.id) > max(ids.index(dep) for dep in question.depends_on)
        for question in questions
        if question.depends_on
    )


def test_generates_left_of_relation_without_left_as_entity() -> None:
    constraints = extract_constraints(
        "a blue cup on the left of three red apples on a wooden table, clean studio photo"
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:cup" in ids
    assert "existence:apples" in ids
    assert "existence:table" in ids
    assert "existence:left" not in ids
    assert "existence:wooden" not in ids
    assert "existence:three" not in ids
    assert "relation:cup:apples:left_of" in ids
    assert "relation:apples:table:on" in ids
    left_question = next(question for question in questions if question.id == "relation:cup:apples:left_of")
    assert "left of the apples" in left_question.question
    assert left_question.depends_on == ["existence:cup", "existence:apples"]


def test_generates_to_the_left_of_relation_without_cup_to_entity() -> None:
    constraints = extract_constraints(
        "a white cup to the left of three red apples on a wooden table, realistic photo"
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:cup" in ids
    assert "existence:cup_to" not in ids
    assert "count:cup" in ids
    assert "count:apples" in ids
    assert "count:table" not in ids
    cup_count = next(question for question in questions if question.id == "count:cup")
    assert cup_count.expected_answer == "1"
    assert cup_count.depends_on == ["existence:cup"]
    assert "relation:cup:apples:left_of" in ids
    left_question = next(question for question in questions if question.id == "relation:cup:apples:left_of")
    assert left_question.depends_on == ["existence:cup", "existence:apples"]


def test_relation_connector_cleanup_is_generic_for_question_entities() -> None:
    constraints = extract_constraints(
        "a yellow bird in front of a black bicycle near a white dog"
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:bird" in ids
    assert "existence:bicycle" in ids
    assert "existence:dog" in ids
    assert "existence:bird_in" not in ids
    assert "existence:bicycle_near" not in ids
    assert "relation:bird:bicycle:in_front_of" in ids
    assert "relation:bicycle:dog:near" in ids


def test_no_color_leakage_and_while_clause_do_not_create_pseudo_entities() -> None:
    constraints = extract_constraints(
        "A cyan cat holds a red umbrella handle while sitting beside a purple "
        "teapot, no color leakage between the objects."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:cat" in ids
    assert "existence:umbrella_handle" in ids
    assert "existence:teapot" in ids
    assert "existence:umbrella_handle_while" not in ids
    assert "existence:teapot_no_color" not in ids
    assert "count:umbrella_handle_while" not in ids
    assert "count:teapot_no_color" not in ids
    assert "relation:cat:umbrella_handle:holds" in ids
    relation = next(
        question
        for question in questions
        if question.id == "relation:cat:umbrella_handle:holds"
    )
    assert "holding the umbrella handle" in relation.question


def test_occlusion_questions_use_relation_not_hidden_part_existence() -> None:
    constraints = extract_constraints(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:screen" in ids
    assert "existence:suitcase" in ids
    assert "existence:suitcase_handle" in ids
    assert "existence:screen_hides_lower" not in ids
    assert "existence:lower_half" not in ids
    relation = next(
        question for question in questions if question.category == "occlusion_relation"
    )
    assert relation.id == "relation:screen:suitcase:hides_lower_half"
    assert "lower half of the suitcase" in relation.question
    assert "suitcase handle" in relation.question
    assert relation.source_constraint["typed_relation"] == "occlusion"


def test_generates_action_object_relations_for_main_subject_accessories() -> None:
    constraints = extract_constraints(
        "a woman wearing a red hat standing beside a black cat, "
        "holding a green handbag, realistic street photo"
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "relation:woman:handbag:holding" in ids
    assert "relation:woman:hat:wearing" in ids
    assert "relation:woman:cat:beside" in ids
    assert "count:woman" in ids
    assert "count:hat" in ids
    assert "count:cat" in ids
    assert "count:handbag" in ids
    assert "relation:cat:handbag:holding" not in ids
    assert "relation:hat:cat:beside" not in ids
    assert "action:woman:standing" in ids
    standing_question = next(question for question in questions if question.id == "action:woman:standing")
    assert "no seated" in standing_question.question


def test_symbol_display_questions_use_structured_subject_and_carrier() -> None:
    constraints = extract_constraints(
        "A blue notebook shows a yellow star symbol on its cover, next to a "
        "plain green notebook with no symbol."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "existence:blue_notebook" in ids
    assert "existence:green_notebook" in ids
    assert "relation:blue_notebook:star_symbol:shows" in ids
    assert "relation:blue_notebook:green_notebook:next_to" in ids
    assert "action:plain_notebook:shows" not in ids
    display_question = next(
        question
        for question in questions
        if question.id == "relation:blue_notebook:star_symbol:shows"
    )
    assert display_question.category == "symbol_text_relation"
    assert "show or display" in display_question.question
    spatial_question = next(
        question
        for question in questions
        if question.id == "relation:blue_notebook:green_notebook:next_to"
    )
    assert "next to the green notebook" in spatial_question.question


def test_question_entities_do_not_reintroduce_action_tail_pseudo_targets() -> None:
    prompts = [
        (
            "Two red cups and one blue bowl are on a wooden table; the cups are "
            "separate and both cups are fully visible."
        ),
        (
            "A transparent box contains one silver key and one black feather, "
            "without any coins or jewelry."
        ),
        (
            "A yellow pyramid is right of a red cylinder, and the red cylinder "
            "is above a blue cube; all three objects are visible."
        ),
        (
            "A turquoise wooden chair, a crimson glass lamp, and a silver paper "
            "fan sit on a black rug."
        ),
    ]
    forbidden_ids = {
        "existence:cups_and",
        "count:cups_and",
        "existence:key_and",
        "count:key_and",
        "existence:objects_are",
        "count:objects_are",
        "existence:pyramid_is",
        "count:pyramid_is",
        "existence:paper_fan_sit",
        "count:paper_fan_sit",
    }

    for prompt in prompts:
        ids = {question.id for question in generate_constraint_questions(extract_constraints(prompt))}

        assert not (ids & forbidden_ids)


def test_question_generation_supports_holdout_color_words() -> None:
    constraints = extract_constraints(
        "A turquoise wooden chair, a crimson glass lamp, and a silver paper fan "
        "sit on a black rug."
    )

    questions = generate_constraint_questions(constraints)
    ids = {question.id for question in questions}
    colors = {
        question.id: question.expected_answer
        for question in questions
        if question.category == "color_binding"
    }

    assert "color:chair" in ids
    assert "color:glass_lamp" in ids
    assert colors["color:chair"] == "turquoise"
    assert colors["color:glass_lamp"] == "crimson"
    assert "existence:paper_fan" in ids
    assert "existence:paper_fan_sit" not in ids


def test_symbol_absence_targets_plain_comparison_object() -> None:
    constraints = extract_constraints(
        "A red lunchbox shows a white moon symbol on its lid, next to a plain "
        "yellow lunchbox with no symbol."
    )

    questions = generate_constraint_questions(constraints)
    negative = next(
        question
        for question in questions
        if question.category == "negative_symbol_text_relation"
    )

    assert negative.id == "negative_symbol:yellow_lunchbox:moon_symbol:absent"
    assert negative.depends_on == ["existence:yellow_lunchbox"]
    assert "yellow lunchbox" in negative.question


def test_unbound_action_fallback_does_not_assign_perches_to_first_subject() -> None:
    constraints = extract_constraints(
        "A white cat sits in front of a black bicycle, while a yellow bird "
        "perches above the bicycle seat."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "relation:cat:bicycle:in_front_of" in ids
    assert "relation:bird:bicycle_seat:above" in ids
    assert "action:cat:perches" not in ids
    assert "action:cat:sits" not in ids


def test_while_holding_clause_inherits_main_subject() -> None:
    constraints = extract_constraints(
        "A blue monkey touches the top of a green drum while holding a silver "
        "spoon in its other hand."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "relation:monkey:drum_top:touches" in ids
    assert "relation:monkey:spoon:holding" in ids
    assert "relation:drum:spoon:holding" not in ids
    touch_question = next(
        question
        for question in questions
        if question.id == "relation:monkey:drum_top:touches"
    )
    assert touch_question.depends_on == ["existence:monkey", "part:drum_top"]


def test_generates_attribute_relation_binding_for_color_bound_handle() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    binding_id = "binding:robot:umbrella:gripping:blue"
    assert binding_id in ids
    binding_question = next(question for question in questions if question.id == binding_id)
    assert "blue umbrella" in binding_question.question
    assert "rather than a different-colored umbrella" in binding_question.question
    assert binding_question.depends_on == [
        "relation:robot:umbrella_handle:gripping",
        "color:umbrella",
    ]


def test_evaluator_marks_uncertain_relation_as_hard_failure() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)
    response = {
        "answers": [
            {
                "id": question.id,
                "answer": _answer_for(question.id),
                "confidence": 0.9,
                "evidence": "mock evidence",
            }
            for question in questions
        ]
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/robot", constraints)

    assert record["constraint_check"]["passed"] is False
    assert record["summary"]["uncertain_hard_checks"] >= 1
    assert any(
        error["type"] == "wrong_relation"
        for error in record["constraint_check"]["errors"]
    )
    assert "strict visual constraint checker" in vlm.calls[0]["prompt"]


def test_evaluator_fails_when_relation_does_not_bind_to_colored_object() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)
    response = {
        "answers": [
            {
                "id": question.id,
                "answer": "no"
                if question.id == "binding:robot:umbrella:gripping:blue"
                else "yes"
                if question.id.startswith(("relation:", "action:"))
                else _answer_for(question.id),
                "confidence": 0.9,
                "evidence": (
                    "The robot grips a red umbrella handle while a blue canopy is nearby."
                    if question.id == "binding:robot:umbrella:gripping:blue"
                    else "mock evidence"
                ),
            }
            for question in questions
        ]
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/robot", constraints)

    assert record["constraint_check"]["passed"] is False
    assert "binding:robot:umbrella:gripping:blue" in record["summary"]["failed_constraints"]
    assert any(
        error["question_id"] == "binding:robot:umbrella:gripping:blue"
        for error in record["constraint_check"]["errors"]
    )


def test_vqa_request_compacts_prior_feedback_history() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)

    request = build_vqa_constraint_request(
        user_prompt=user_prompt,
        prompt=user_prompt,
        image_path="/tmp/fake.png",
        questions=questions,
        history=[
            {
                "round": 0,
                "source": "visual_reflector",
                "feedback": {
                    "score": 0.4,
                    "raw_response": "x" * 40000,
                    "request": "y" * 40000,
                    "constraint_check": {
                        "passed": False,
                        "source": "question_level_vqa",
                        "raw_response": "z" * 40000,
                        "errors": [
                            {
                                "type": "wrong_relation",
                                "question_id": "action:robot:clearly_gripping",
                                "evidence": "The robot is not gripping the handle.",
                            }
                        ],
                    },
                },
            }
        ],
    )

    assert len(request) < 20000
    assert "raw_response" not in request
    assert "The robot is not gripping the handle" in request


def test_vqa_request_marks_expanded_prompt_as_non_binding_context() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)

    request = build_vqa_constraint_request(
        user_prompt=user_prompt,
        prompt="a small matte crimson robot with hydraulic rivets gripping a cobalt umbrella",
        image_path="/tmp/fake.png",
        questions=questions,
        history=[],
    )

    assert "Original user prompt with binding constraints" in request
    assert "Expanded prompt for non-binding context only" in request
    assert "do not add constraints from it" in request


def test_negative_attached_relation_becomes_hard_vqa_question() -> None:
    constraints = extract_constraints(
        "A green wizard carries a silver lantern while standing beside an orange "
        "barrel; the lantern is not attached to the barrel."
    )

    questions = generate_constraint_questions(constraints)
    ids = [question.id for question in questions]

    assert "negative_relation:lantern:barrel:attached_to" in ids
    question = next(
        item
        for item in questions
        if item.id == "negative_relation:lantern:barrel:attached_to"
    )
    assert question.category == "negative_relation"
    assert "clearly separate" in question.question
    assert "no visible attached to" in question.question


def test_negative_attached_relation_failure_blocks_completion() -> None:
    user_prompt = (
        "A green wizard carries a silver lantern while standing beside an orange "
        "barrel; the lantern is not attached to the barrel."
    )
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)
    response = {
        "answers": [
            {
                "id": question.id,
                "answer": (
                    "no"
                    if question.id == "negative_relation:lantern:barrel:attached_to"
                    else _answer_for(question.id)
                ),
                "confidence": 0.9,
                "evidence": (
                    "The lantern is visibly attached to the orange barrel."
                    if question.id == "negative_relation:lantern:barrel:attached_to"
                    else _answer_for(question.id)
                ),
            }
            for question in questions
        ]
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/wizard", constraints)
    failed = {error["question_id"] for error in record["constraint_check"]["errors"]}

    assert record["constraint_check"]["passed"] is False
    assert "negative_relation:lantern:barrel:attached_to" in failed


def test_attached_or_supported_relation_evidence_does_not_pass_gripping() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)
    relation_ids = {
        "relation:robot:umbrella_handle:gripping",
        "action:robot:clearly_gripping",
    }
    response = {
        "answers": [
            {
                "id": question.id,
                "answer": "yes" if question.id in relation_ids else _answer_for(question.id),
                "confidence": 0.9,
                "evidence": (
                    "The umbrella handle is attached to and supported by the robot body, indicating a gripping relationship."
                    if question.id in relation_ids
                    else _answer_for(question.id)
                ),
            }
            for question in questions
        ]
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/robot", constraints)
    failed = {error["question_id"] for error in record["constraint_check"]["errors"]}

    assert record["constraint_check"]["passed"] is False
    assert relation_ids <= failed


def test_missing_parent_blocks_dependent_color_and_relation_questions() -> None:
    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    constraints = extract_constraints(user_prompt)
    questions = generate_constraint_questions(constraints)
    response = {
        "answers": [
            {
                "id": question.id,
                "answer": "no" if question.id == "existence:umbrella" else "yes",
                "confidence": 0.9,
                "evidence": "umbrella is not visible" if question.id == "existence:umbrella" else "",
            }
            for question in questions
        ]
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/robot", constraints)
    blocked_ids = {answer["id"] for answer in record["answers"] if answer["blocked_by"]}
    errors = record["constraint_check"]["errors"]

    assert "color:umbrella" in blocked_ids
    assert any(item.startswith("relation:") for item in blocked_ids)
    assert any(error["type"] == "missing_object" for error in errors)
    assert not any(
        error.get("question_id") == "color:umbrella"
        for error in errors
    )


def test_parser_does_not_index_fallback_when_answer_ids_are_present() -> None:
    user_prompt = "a white cup to the left of three red apples on a wooden table"
    constraints = extract_constraints(user_prompt)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "answers": [
                        {"id": "existence:cup", "answer": "yes"},
                        {"id": "existence:apples", "answer": "yes"},
                        {"id": "existence:table", "answer": "yes"},
                        {"id": "count:apples", "answer": "3"},
                        {"id": "color:cup", "answer": "white"},
                        {"id": "color:apples", "answer": "red"},
                    ]
                }
            )
        ]
    )
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(user_prompt, user_prompt, "mock://image/cup", constraints)
    count_cup = next(answer for answer in record["answers"] if answer["id"] == "count:cup")

    assert count_cup["normalized_answer"] == "uncertain"
    assert count_cup["passed"] is False


def test_evaluator_accepts_legacy_constraint_check_json() -> None:
    user_prompt = "a small red robot holding a blue umbrella"
    response = {
        "passed": False,
        "score": 0.4,
        "checks": [
            {
                "type": "color",
                "target": "umbrella",
                "expected": "blue",
                "observed": "red",
                "passed": False,
            }
        ],
        "errors": [],
        "revision_hint": "Make the umbrella blue.",
    }
    vlm = MockVLMClient(responses=[json.dumps(response)])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(
        user_prompt,
        user_prompt,
        "mock://image/robot",
        extract_constraints(user_prompt),
    )

    assert record["source"] == "legacy_constraint_check"
    assert record["constraint_check"]["passed"] is False
    assert record["constraint_check"]["errors"][0]["type"] == "wrong_attribute"


def test_single_text_answer_falls_back_to_uncertain() -> None:
    user_prompt = "a blue umbrella"
    vlm = MockVLMClient(responses=["I cannot tell from this image."])
    evaluator = VQAConstraintEvaluator(vlm)

    record = evaluator.evaluate(
        user_prompt,
        user_prompt,
        "mock://image/umbrella",
        extract_constraints(user_prompt),
    )

    assert record["constraint_check"]["passed"] is False
    assert record["answers"][0]["normalized_answer"] == "uncertain"


def test_mass_entity_bread_does_not_create_exact_count_question() -> None:
    questions = generate_constraint_questions(
        extract_constraints("A baker pulling freshly baked bread out of an oven in a bakery.")
    )
    ids = {question.id for question in questions}

    assert "existence:freshly_baked_bread" in ids
    assert "count:freshly_baked_bread" not in ids


def _answer_for(question_id: str) -> str:
    if question_id.startswith("existence:"):
        return "yes"
    if question_id in {
        "count:robot",
        "count:umbrella",
        "count:bicycle",
        "count:dog",
        "count:woman",
        "count:hat",
        "count:cat",
        "count:cup",
        "count:lantern",
        "count:barrel",
    }:
        return "1"
    if question_id == "count:birds":
        return "2"
    if question_id == "count:apples":
        return "3"
    if question_id == "color:robot":
        return "red"
    if question_id == "color:umbrella":
        return "blue"
    if question_id == "color:birds":
        return "yellow"
    if question_id == "color:bicycle":
        return "black"
    if question_id == "color:dog":
        return "white"
    if question_id == "color:hat":
        return "yellow"
    if question_id == "color:cat":
        return "black"
    if question_id == "color:cup":
        return "white"
    if question_id == "color:apples":
        return "red"
    if question_id == "color:wizard":
        return "green"
    if question_id == "color:lantern":
        return "silver"
    if question_id == "color:barrel":
        return "orange"
    if question_id.startswith("part:"):
        return "yes"
    if question_id.startswith("relation:") or question_id.startswith("action:"):
        return "uncertain"
    return "yes"
