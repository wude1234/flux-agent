import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockLLMClient, MockVLMClient
from src.evaluators import (
    VLMJudgeEvaluator,
    evaluation_to_optimizer_errors,
    normalize_error_type,
    parse_vlm_judge_response,
)
from src.factuality_qa import (
    FactualQuestion,
    FactualityQAEvaluator,
    build_vqa_answer_request,
    infer_prompt_domain,
    parse_answer_response,
    parse_questions_response,
    should_run_factuality_qa,
)
from src.reward_reranker import (
    MockRewardBackend,
    RewardReranker,
    VLMRewardBackend,
    pairwise_preference_probability,
    parse_vlm_reward_response,
    ranking_to_evaluation,
)


def test_parse_vlm_judge_response_normalizes_schema() -> None:
    response = json.dumps(
        {
            "score": 4,
            "criteria_scores": {
                "alignment": 0.8,
                "attribute-binding": 0.25,
                "artifact": 5,
            },
            "errors": [
                {
                    "type": "color",
                    "evidence": "The umbrella is red, not blue.",
                    "prompt_span": "blue umbrella",
                }
            ],
            "strengths": "Robot is visible.",
            "revision_hint": "Make the umbrella blue.",
        }
    )

    parsed = parse_vlm_judge_response(
        response,
        criteria=("alignment", "attribute_binding", "artifact_quality"),
    )

    assert parsed["score"] == 0.8
    assert parsed["passed"] is True
    assert parsed["criteria_scores"]["attribute_binding"] == 0.25
    assert parsed["criteria_scores"]["artifact_quality"] == 1.0
    assert parsed["errors"][0]["type"] == "wrong_attribute"
    assert parsed["strengths"] == ["Robot is visible."]


def test_vlm_judge_evaluator_calls_vlm_and_returns_optimizer_errors() -> None:
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "score": 0.42,
                    "criteria_scores": {"alignment": 0.5},
                    "errors": [{"type": "relation", "evidence": "The robot is not holding the umbrella."}],
                    "strengths": ["Rainy street is present."],
                }
            )
        ]
    )
    evaluator = VLMJudgeEvaluator(vlm, criteria=("alignment", "object_relationship"))

    result = evaluator.evaluate(
        "a red robot holding a blue umbrella",
        "red robot, blue umbrella",
        "mock://image/0",
    )
    optimizer_errors = evaluation_to_optimizer_errors(
        result,
        original_prompt="a red robot holding a blue umbrella",
    )

    assert result["evaluator"] == "vlm_judge"
    assert result["score"] == 0.42
    assert result["passed"] is False
    assert "original user prompt first" in vlm.calls[0]["prompt"]
    assert optimizer_errors[0]["error_type"] == "wrong_relation"


def test_normalize_error_type_aliases() -> None:
    assert normalize_error_type("counting") == "wrong_count"
    assert normalize_error_type("hallucination") == "factuality"
    assert normalize_error_type("unknown") == "wrong_attribute"


def test_vlm_judge_response_text_fallback_extracts_score_and_errors() -> None:
    parsed = parse_vlm_judge_response(
        "Score: 6/10\nIssue: wrong color, the umbrella is red instead of blue."
    )

    assert parsed["score"] == 0.6
    assert parsed["passed"] is False
    assert parsed["errors"][0]["type"] == "wrong_attribute"


def test_factuality_domain_gate() -> None:
    assert infer_prompt_domain("an educational diagram of a plant cell") == "science"
    assert should_run_factuality_qa("a roman battle scene") is True
    assert should_run_factuality_qa("a cute robot in rain") is False
    assert should_run_factuality_qa("anything", domain="medical_diagram") is True


def test_parse_questions_and_answer_response() -> None:
    raw = json.dumps(
        {
            "questions": [
                {
                    "question": "How many moons are shown?",
                    "choices": {"A": "one", "B": "two", "C": "three", "D": "four", "E": "None"},
                    "answer": "B",
                    "coi": "counting",
                }
            ]
        }
    )

    questions = parse_questions_response(raw)

    assert questions[0].answer == "B"
    assert parse_answer_response("Answer: B") == "B"
    assert parse_answer_response("unclear") == "E"


def test_build_vqa_answer_request_requires_single_letter() -> None:
    question = FactualQuestion(
        question="Which organ is highlighted?",
        choices={"A": "heart", "B": "lung", "C": "brain", "D": "kidney", "E": "None"},
        answer="A",
    )

    request = build_vqa_answer_request(question)

    assert "Return only one character" in request
    assert "A) heart" in request


def test_factuality_qa_evaluator_scores_question_failures() -> None:
    question = FactualQuestion(
        question="Which molecule is depicted?",
        choices={"A": "water", "B": "methane", "C": "oxygen", "D": "salt", "E": "None"},
        answer="A",
        coi="factuality",
    )
    vlm = MockVLMClient(responses=["A", "C"])
    evaluator = FactualityQAEvaluator(vlm)

    passed = evaluator.evaluate(
        "science diagram of a water molecule",
        "mock://water.png",
        questions=[question],
    )
    failed = evaluator.evaluate(
        "science diagram of a water molecule",
        "mock://wrong.png",
        questions=[question],
    )

    assert passed["score"] == 1.0
    assert passed["errors"] == []
    assert failed["score"] == 0.0
    assert failed["errors"][0]["type"] == "factuality"


def test_factuality_qa_generates_questions_with_llm() -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question": "What phase of the moon is shown?",
                            "choices": {"A": "full", "B": "new", "C": "crescent", "D": "gibbous", "E": "None"},
                            "answer": "C",
                            "coi": "shape",
                        }
                    ]
                }
            )
        ]
    )
    evaluator = FactualityQAEvaluator(MockVLMClient(default_response="C"), llm=llm)

    result = evaluator.evaluate("science diagram of a crescent moon", "mock://moon.png")

    assert result["score"] == 1.0
    assert result["questions"][0]["answer"] == "C"
    assert "I-HallA-style QA builder" in llm.calls[0]


def test_factuality_qa_skips_creative_prompts() -> None:
    evaluator = FactualityQAEvaluator(MockVLMClient())

    result = evaluator.evaluate("a dreamy robot in the rain", "mock://robot.png")

    assert result["skipped"] is True
    assert result["score"] is None


def test_reward_reranker_ranks_images_and_exports_evaluation() -> None:
    backend = MockRewardBackend(scores={"b.png": 0.9, "a.png": 0.3})
    reranker = RewardReranker(backend, aspects=("overall",))

    ranking = reranker.rank("a red robot holding a blue umbrella", ["a.png", "b.png"])
    evaluation = ranking_to_evaluation(ranking)

    assert ranking["selected_image"] == "b.png"
    assert ranking["selected_index"] == 1
    assert ranking["scores"][0]["score"] == 0.9
    assert evaluation["passed"] is True
    assert backend.calls[0]["aspect"] == "overall"


def test_vlm_reward_backend_uses_vlm_api_proxy() -> None:
    vlm = MockVLMClient(responses=[json.dumps({"score": 0.82, "reason": "good"})])
    backend = VLMRewardBackend(vlm)

    score = backend.score("a prompt", "mock://image.png", aspect="alignment")

    assert score == 0.82
    assert "API reward model proxy" in vlm.calls[0]["prompt"]
    assert backend.calls[0]["aspect"] == "alignment"


def test_parse_vlm_reward_response_text_fallback() -> None:
    assert parse_vlm_reward_response("Reward: 8/10") == 0.8
    assert parse_vlm_reward_response("score=0.7") == 0.7


def test_reward_reranker_weighted_aspects() -> None:
    class AspectBackend:
        def score(self, prompt, image_path, *, aspect="overall"):
            del prompt
            values = {
                ("a.png", "alignment"): 0.9,
                ("a.png", "fidelity"): 0.2,
                ("b.png", "alignment"): 0.4,
                ("b.png", "fidelity"): 0.9,
            }
            return values[(image_path, aspect)]

    reranker = RewardReranker(
        AspectBackend(),
        aspects=("alignment", "fidelity"),
        weights={"alignment": 0.25, "fidelity": 0.75},
    )

    ranking = reranker.rank("prompt", ["a.png", "b.png"])

    assert ranking["selected_image"] == "b.png"
    assert ranking["weights"] == {"alignment": 0.25, "fidelity": 0.75}


def test_pairwise_preference_probability() -> None:
    assert pairwise_preference_probability(1.0, 1.0) == 0.5
    assert pairwise_preference_probability(2.0, 0.0) > 0.8
