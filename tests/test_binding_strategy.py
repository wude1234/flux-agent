from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binding_strategy import (
    build_binding_retry_prompt,
    build_negative_prompt,
    has_binding_failure,
)
from src.prompt_constraints import approx_clip_token_count, extract_constraints


def test_binding_retry_prompt_foregrounds_color_and_grip_constraints() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )
    critique = {
        "score": 0.4,
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "The umbrella canopy is red, not blue.",
                "prompt_span": "blue umbrella",
            },
            {
                "type": "wrong_relation",
                "evidence": "The hand is not clearly gripping the handle.",
                "prompt_span": "clearly gripping",
            },
        ],
    }

    retry = build_binding_retry_prompt(
        "cinematic rainy street photo of a red robot with an umbrella",
        constraints,
        critique,
        token_budget=77,
    )

    prompt = retry["prompt"].lower()
    assert "blue umbrella" in prompt
    assert "robot visibly gripping the umbrella handle" in prompt
    assert "clear physical contact between robot and the umbrella handle" in prompt
    assert "red umbrella" not in prompt
    assert approx_clip_token_count(retry["prompt"]) <= 77
    assert "red umbrella" in retry["negative_prompt"]


def test_binding_retry_uses_interaction_target_not_last_colored_object() -> None:
    constraints = extract_constraints(
        "A cyan cat holds a red umbrella handle while sitting beside a purple "
        "teapot, no color leakage between the objects."
    )
    critique = {
        "score": 0.64,
        "constraint_check": {
            "passed": False,
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The cat color needs emphasis.",
                    "prompt_span": "cyan cat",
                }
            ],
        },
    }

    retry = build_binding_retry_prompt(
        "A cyan cat sits beside a purple teapot.",
        constraints,
        critique,
        token_budget=77,
    )

    prompt = retry["prompt"].lower()
    assert "cat visibly holding the umbrella handle" in prompt
    assert "holds the teapot" not in prompt


def test_negative_prompt_targets_likely_cross_color_conflicts() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )

    negative_prompt = build_negative_prompt(constraints)

    assert "red umbrella" in negative_prompt
    assert "blue robot" in negative_prompt
    assert "hidden contact point" in negative_prompt
    assert "umbrella handle not gripped" in negative_prompt


def test_negative_prompt_does_not_forbid_required_same_class_objects() -> None:
    constraints = extract_constraints(
        "A blue notebook shows a yellow star symbol on its cover, next to a "
        "plain green notebook with no symbol."
    )

    negative_prompt = build_negative_prompt(constraints)

    assert "green notebook" not in negative_prompt
    assert "blue notebook" not in negative_prompt
    assert "yellow star symbol" not in negative_prompt
    assert "wrong notebook color" in negative_prompt
    assert "wrong symbol color" in negative_prompt


def test_binding_retry_is_generic_for_carry_relation() -> None:
    constraints = extract_constraints(
        "A green wizard carries a silver lantern while standing beside an orange barrel."
    )
    critique = {
        "score": 0.5,
        "constraint_check": {
            "passed": False,
            "errors": [
                {
                    "type": "wrong_relation",
                    "evidence": "The wizard appears to carry the barrel instead of the lantern.",
                    "prompt_span": "wizard carries lantern",
                }
            ],
        },
    }

    retry = build_binding_retry_prompt(
        "A green wizard stands beside an orange barrel.",
        constraints,
        critique,
        token_budget=77,
    )

    prompt = retry["prompt"].lower()
    assert "green wizard" in prompt
    assert "silver lantern" in prompt
    assert "orange barrel" in prompt
    assert "wizard visibly carrying the lantern" in prompt
    assert "carrying the barrel" not in prompt


def test_has_binding_failure_uses_constraint_check_result() -> None:
    constraints = extract_constraints("a red robot holding a blue umbrella")
    critique = {
        "score": 0.92,
        "errors": [],
        "constraint_check": {
            "passed": False,
            "score": 0.5,
            "errors": [{"evidence": "The umbrella is red."}],
        },
    }

    assert has_binding_failure(critique, constraints)
