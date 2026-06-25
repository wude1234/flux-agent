import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockLLMClient
from src.prompt_reviser import PromptReviser


def test_generate_initial_prompts_uses_llm_and_json_response() -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "prompts": [
                        "A blue Labrador retriever sitting on green grass, sunny light.",
                        "A vivid blue dog on grass, realistic photo style.",
                    ]
                }
            )
        ]
    )
    reviser = PromptReviser(llm)

    prompts = reviser.generate_initial_prompts("blue dog on grass", n=2)

    assert prompts == [
        "A blue Labrador retriever sitting on green grass, sunny light.",
        "A vivid blue dog on grass, realistic photo style.",
    ]
    assert len(llm.calls) == 1
    assert "PromptAgent" in llm.calls[0]
    assert "Write exactly 2 diverse prompts" in llm.calls[0]


def test_revise_returns_first_revised_prompt_from_start_end_response() -> None:
    llm = MockLLMClient(
        responses=[
            "<START>Exactly eight red apples arranged in one row on a wooden table, clean studio light.</END>"
        ]
    )
    reviser = PromptReviser(llm)

    prompt = reviser.revise(
        "8 apples on the table",
        "red apples on a table",
        {
            "score": 0.4,
            "errors": [
                {
                    "type": "wrong_count",
                    "evidence": "There are too many apples.",
                    "prompt_span": "8 apples",
                }
            ],
            "strengths": ["The table is clear."],
            "revision_hint": "Specify exactly eight apples in one row.",
        },
        history=[{"round": 0, "prompt": "red apples"}],
    )

    assert prompt.startswith("Exactly eight red apples")
    assert len(llm.calls) == 1
    assert "Visual critique" in llm.calls[0]
    assert "exactly eight apples" in llm.calls[0].lower()


def test_revise_candidates_parses_json_and_deduplicates() -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "prompts": [
                        "A person in boat pose forming a V shape on the beach.",
                        "A person in boat pose forming a V shape on the beach.",
                        "Beach yoga scene with legs and torso lifted in a V shape.",
                    ]
                }
            )
        ]
    )
    reviser = PromptReviser(llm)

    prompts = reviser.revise_candidates(
        "a person practicing yoga boat pose at beach",
        "a person doing yoga at beach",
        "Describe boat pose as balancing on sit bones with legs and torso raised.",
        n=2,
    )

    assert prompts == [
        "A person in boat pose forming a V shape on the beach.",
        "Beach yoga scene with legs and torso lifted in a V shape.",
    ]


def test_revise_falls_back_to_current_prompt_plus_hint_when_llm_is_unstructured() -> None:
    llm = MockLLMClient(responses=[""])
    reviser = PromptReviser(llm)

    prompt = reviser.revise(
        "blue dog",
        "dog on grass",
        {"revision_hint": "make the dog blue"},
    )

    assert prompt == "dog on grass, refined to address: make the dog blue"

