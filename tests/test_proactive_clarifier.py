import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.belief_state import Attribute, BeliefState, Candidate, Entity, normalized_entropy
from src.clients import MockLLMClient
from src.proactive_clarifier import ProactiveClarifier


def test_belief_state_computes_attribute_ask_score() -> None:
    belief = BeliefState(
        entities=[
            Entity(
                name="city",
                importance_score=0.8,
                attributes=[
                    Attribute(
                        name="image style",
                        importance_score=0.9,
                        candidates=[
                            Candidate("cinematic", 0.5),
                            Candidate("illustration", 0.5),
                        ],
                    )
                ],
            )
        ],
        prompt="a futuristic city",
    )

    targets = belief.clarification_targets()

    assert normalized_entropy([0.5, 0.5]) == 1.0
    assert targets[0]["kind"] == "attribute"
    assert targets[0]["missing_slot"] == "image style"
    assert targets[0]["ask_score"] == 0.9


def test_clarifier_asks_for_underspecified_prompt_with_fallback_belief() -> None:
    llm = MockLLMClient(default_response="not json")
    clarifier = ProactiveClarifier(llm, creativity_level="medium")

    result = clarifier.decide("a futuristic city")

    assert result["status"] == "ask_user"
    assert result["missing_slot"] in {"image style", "viewpoint", "mood"}
    assert result["ask_score"] >= 0.35
    assert result["question"]
    assert len(llm.calls) == 2
    assert "belief parser" in llm.calls[0]
    assert "ClarifierAgent" in llm.calls[1]


def test_clarifier_does_not_ask_for_specific_prompt() -> None:
    llm = MockLLMClient(default_response="not json")
    clarifier = ProactiveClarifier(llm, creativity_level="medium")

    result = clarifier.decide(
        "a red vintage car parked in front of a diner at night, cinematic photo, wide view"
    )

    assert result["status"] == "do_not_ask"
    assert result["question"] is None
    assert result["ask_score"] < 0.35


def test_clarifier_parses_llm_belief_and_question() -> None:
    belief_response = json.dumps(
        {
            "prompt": "a scientist holding an unusual object",
            "entities": [
                {
                    "name": "unusual object",
                    "importance_score": 0.95,
                    "descriptions": "central object held by the scientist",
                    "entity_type": "explicit",
                    "probability": 1.0,
                    "attributes": [
                        {
                            "name": "identity",
                            "importance_score": 0.95,
                            "candidates": {
                                "crystal device": 0.34,
                                "alien artifact": 0.33,
                                "lab instrument": 0.33,
                            },
                        }
                    ],
                }
            ],
            "relations": [],
        }
    )
    llm = MockLLMClient(
        responses=[
            belief_response,
            "<question>What unusual object should the scientist be holding?</question>",
        ]
    )
    clarifier = ProactiveClarifier(llm, creativity_level="medium")

    result = clarifier.decide("a scientist holding an unusual object")

    assert result["status"] == "ask_user"
    assert result["missing_slot"] == "identity"
    assert result["question"] == "What unusual object should the scientist be holding?"
    assert result["target"]["entity"] == "unusual object"


def test_high_creativity_auto_fills_more_often() -> None:
    llm = MockLLMClient(default_response="not json")
    clarifier = ProactiveClarifier(llm, creativity_level="high")

    result = clarifier.decide("a futuristic city")

    assert result["status"] == "do_not_ask"
    assert 0.70 < result["ask_score"] < 0.95


def test_merge_answer_uses_llm_and_returns_state_update() -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "merged_prompt": (
                        "a futuristic city in a cinematic realistic style, "
                        "with a street-level view"
                    )
                }
            )
        ]
    )
    clarifier = ProactiveClarifier(llm)

    result = clarifier.merge_answer(
        "a futuristic city",
        "What image style should the city have?",
        "cinematic realistic, street-level view",
    )

    assert result["status"] == "merged"
    assert result["merged_prompt"].startswith("a futuristic city in a cinematic")
    assert result["update"] == {"user_prompt": result["merged_prompt"]}
    assert len(llm.calls) == 1
    assert "Merge the answer" in llm.calls[0]


def test_merge_answer_falls_back_when_llm_is_unstructured() -> None:
    llm = MockLLMClient(default_response="plain text without prompt tag")
    clarifier = ProactiveClarifier(llm)

    result = clarifier.merge_answer(
        "a scientist holding an unusual object",
        "What is the unusual object?",
        "an alien crystal artifact",
    )

    assert result["merged_prompt"] == (
        "a scientist holding an unusual object, "
        "with this clarified detail: an alien crystal artifact"
    )
