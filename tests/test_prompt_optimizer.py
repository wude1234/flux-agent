import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockLLMClient, MockVLMClient
from src.candidate_scorer import CandidateScorer
from src.error_analyzer import ErrorAnalyzer, PromptError
from src.prompt_optimizer import PromptOptimizer


def test_error_analyzer_maps_visual_critique_to_prompt_error() -> None:
    analyzer = ErrorAnalyzer()
    critique = {
        "errors": [
            {
                "type": "wrong_count",
                "evidence": "The image shows 12 apples instead of exactly 8.",
                "prompt_span": "8 apples on the table",
            }
        ],
        "revision_hint": "Specify exactly eight apples in one row.",
    }

    errors = analyzer.analyze(
        "8 apples on the table. A blue wall in the background.",
        critique,
    )

    assert len(errors) == 1
    assert errors[0].failed_sentence == "8 apples on the table"
    assert errors[0].error_type == "wrong_count"
    assert errors[0].aspect == "attribute_binding"


def test_prompt_optimizer_generates_unique_candidates_and_avoids_memory() -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "modified_sentence": "Exactly 8 apples arranged in one clear row on the table.",
                    "prompt": "Exactly 8 apples arranged in one clear row on the table. A blue wall in the background.",
                    "fixes": ["wrong_count"],
                    "expected_improvement": "Makes the apple count explicit.",
                    "risk": "May reduce natural composition.",
                },
                {
                    "modified_sentence": "There are eight and only eight apples on the table.",
                    "prompt": "There are eight and only eight apples on the table. A blue wall in the background.",
                    "fixes": ["wrong_count"],
                    "expected_improvement": "Repeats the target count.",
                },
            ]
        }
    )
    llm = MockLLMClient(responses=[response])
    optimizer = PromptOptimizer(llm)
    error = PromptError(
        original_prompt="8 apples on the table. A blue wall in the background.",
        failed_sentence="8 apples on the table.",
        error="The image shows 12 apples instead of exactly 8.",
        error_type="wrong_count",
    )

    candidates = optimizer.generate_candidate_prompts(
        error,
        num_candidates=3,
        memory=[
            {
                "prompt": "Exactly 8 apples arranged in one clear row on the table. A blue wall in the background."
            }
        ],
    )

    prompts = [candidate["prompt"] for candidate in candidates]
    assert len(candidates) == 3
    assert prompts[0].startswith("There are eight and only eight apples")
    assert all("Exactly 8 apples arranged" not in prompt for prompt in prompts)
    assert candidates[0]["fixes"] == ["wrong_count"]
    assert len(llm.calls) == 1
    assert "PromptOptimizerAgent" in llm.calls[0]


def test_candidate_scorer_ranks_from_mocked_vlm_json() -> None:
    response = json.dumps(
        {
            "scores": [
                {
                    "index": 0,
                    "subscores": {
                        "alignment": 0.6,
                        "attribute_binding": 0.5,
                        "object_relationship": 0.5,
                        "background_consistency": 0.6,
                        "aesthetic": 0.7,
                    },
                    "reason": "Still ambiguous.",
                },
                {
                    "index": 1,
                    "subscores": {
                        "alignment": 0.9,
                        "attribute_binding": 1.0,
                        "object_relationship": 0.8,
                        "background_consistency": 0.8,
                        "aesthetic": 0.7,
                    },
                    "reason": "Best count and relation fix.",
                },
            ]
        }
    )
    vlm = MockVLMClient(responses=[response])
    scorer = CandidateScorer(vlm)

    ranked = scorer.score_candidates(
        "8 apples on the table",
        [
            {"prompt": "apples on a table", "fixes": []},
            {"prompt": "exactly 8 apples in one row on the table", "fixes": ["wrong_count"]},
        ],
        image_paths=["mock://image/0", "mock://image/1"],
    )

    assert ranked[0]["prompt"] == "exactly 8 apples in one row on the table"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["reason"] == "Best count and relation fix."
    assert len(vlm.calls) == 1
    assert "MLLM scorer" in vlm.calls[0]["prompt"]
    assert vlm.calls[0]["image_paths"] == ["mock://image/0", "mock://image/1"]


def test_stage1_feedback_feeds_stage2_candidate_generator() -> None:
    critique = {
        "score": 0.4,
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": "The person is not balancing in a V-shaped boat pose.",
                "prompt_span": "a person practicing yoga boat pose at beach",
            }
        ],
        "revision_hint": "Describe boat pose as legs and torso lifted in a V shape.",
    }
    errors = ErrorAnalyzer().analyze(
        "a person practicing yoga boat pose at beach",
        critique,
    )
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": "A person balances on sit bones with legs and torso raised in a V shape on the beach.",
                            "prompt": "A person balances on sit bones with legs and torso raised in a V shape on the beach.",
                            "fixes": ["wrong_relation"],
                            "expected_improvement": "Explains the boat pose visually.",
                        }
                    ]
                }
            )
        ]
    )

    candidates = PromptOptimizer(llm).optimize(errors, num_candidates=2)

    assert len(errors) == 1
    assert len(candidates) == 2
    assert candidates[0]["prompt"].startswith("A person balances on sit bones")
    assert candidates[1]["source"] == "genpilot"


def test_fallback_prompt_does_not_embed_raw_failure_text() -> None:
    error = PromptError(
        original_prompt=(
            "A turquoise wooden chair, a crimson glass lamp, and a silver "
            "paper fan sit on a black rug."
        ),
        failed_sentence=(
            "A turquoise wooden chair, a crimson glass lamp, and a silver "
            "paper fan sit on a black rug."
        ),
        error="The silver paper fan is not present in the image.",
        error_type="missing_object",
    )
    optimizer = PromptOptimizer(MockLLMClient(responses=[json.dumps({"candidates": []})]))

    candidate = optimizer.generate_candidate_prompts(error, num_candidates=1)[0]
    prompt = candidate["prompt"].lower()

    assert "silver paper fan" in prompt
    assert "not present" not in prompt
    assert "violating" not in prompt
    assert "(" not in prompt
    assert ")" not in prompt


def test_candidate_scorer_has_deterministic_fallback_without_vlm() -> None:
    scorer = CandidateScorer()

    ranked = scorer.score_candidates(
        "red car on rainy street",
        [
            {"prompt": "car on street", "fixes": []},
            {
                "prompt": "clearly visible red car on a rainy street",
                "fixes": ["wrong_attribute"],
                "expected_improvement": "Emphasizes red car and rainy street.",
            },
        ],
    )

    assert ranked[0]["prompt"] == "clearly visible red car on a rainy street"
    assert ranked[0]["score"] > ranked[1]["score"]
