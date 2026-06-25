"""I-HallA-style factual QA evaluator for M6."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence

from .clients import LLMClient, VLMClient


FACTUAL_DOMAINS = {"science", "history", "education", "medical_diagram"}
DEFAULT_CHOICES = {
    "E": "None of the above",
}


@dataclass(frozen=True)
class FactualQuestion:
    """One visual multiple-choice factuality question."""

    question: str
    choices: dict[str, str]
    answer: str
    coi: str = "factuality"
    rationale: str = ""

    def __post_init__(self) -> None:
        _clean_text(self.question, "question")
        if not isinstance(self.choices, Mapping) or not self.choices:
            raise ValueError("choices must be a non-empty mapping")
        normalized_choices = {
            _normalize_answer_key(key): _clean_text(str(value), "choice")
            for key, value in self.choices.items()
        }
        object.__setattr__(self, "choices", normalized_choices)
        answer = _normalize_answer_key(self.answer)
        if answer not in normalized_choices:
            raise ValueError("answer must be one of choices")
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "coi", str(self.coi or "factuality"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "choices": dict(self.choices),
            "answer": self.answer,
            "coi": self.coi,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactualQuestion":
        return cls(
            question=str(data["question"]),
            choices=dict(data["choices"]),
            answer=str(data["answer"]),
            coi=str(data.get("coi", "factuality")),
            rationale=str(data.get("rationale", "")),
        )


class FactualityQAEvaluator:
    """Evaluate factual prompts by generating or receiving visual QA checks."""

    def __init__(
        self,
        vlm: VLMClient,
        *,
        llm: LLMClient | None = None,
        question_count: int = 5,
    ) -> None:
        if question_count < 1:
            raise ValueError("question_count must be at least 1")
        self.vlm = vlm
        self.llm = llm
        self.question_count = question_count

    def should_run(self, prompt: str, *, domain: str | None = None) -> bool:
        return should_run_factuality_qa(prompt, domain=domain)

    def generate_questions(
        self,
        user_prompt: str,
        *,
        domain: str | None = None,
        facts: Sequence[str] | None = None,
    ) -> list[FactualQuestion]:
        user_prompt = _clean_text(user_prompt, "user_prompt")
        if self.llm is None:
            return build_fallback_questions(user_prompt, facts=facts, count=self.question_count)
        request = build_question_generation_request(
            user_prompt,
            domain=domain,
            facts=facts,
            question_count=self.question_count,
        )
        raw_response = self.llm.text(request)
        questions = parse_questions_response(raw_response)
        return questions[: self.question_count] or build_fallback_questions(
            user_prompt,
            facts=facts,
            count=self.question_count,
        )

    def evaluate(
        self,
        user_prompt: str,
        image_path: str,
        *,
        domain: str | None = None,
        facts: Sequence[str] | None = None,
        questions: Sequence[FactualQuestion | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        user_prompt = _clean_text(user_prompt, "user_prompt")
        image_path = _clean_text(image_path, "image_path")
        run_qa = self.should_run(user_prompt, domain=domain)
        normalized_questions = _normalize_questions(questions or [])
        if run_qa and not normalized_questions:
            normalized_questions = self.generate_questions(
                user_prompt,
                domain=domain,
                facts=facts,
            )
        if not run_qa:
            return {
                "evaluator": "factuality_qa",
                "skipped": True,
                "domain": domain or infer_prompt_domain(user_prompt),
                "score": None,
                "questions": [],
                "results": [],
                "errors": [],
                "revision_hint": "",
            }

        results: list[dict[str, Any]] = []
        for question in normalized_questions:
            request = build_vqa_answer_request(question)
            raw_response = self.vlm.vision(request, [image_path])
            predicted = parse_answer_response(raw_response)
            passed = predicted == question.answer
            results.append(
                {
                    "question": question.to_dict(),
                    "predicted_answer": predicted,
                    "passed": passed,
                    "raw_response": raw_response,
                    "request": request,
                }
            )
        score = _score_results(results)
        errors = [
            {
                "type": "factuality",
                "evidence": _failed_question_evidence(item),
                "prompt_span": user_prompt,
                "severity": "major",
            }
            for item in results
            if not item["passed"]
        ]
        return {
            "evaluator": "factuality_qa",
            "skipped": False,
            "domain": domain or infer_prompt_domain(user_prompt),
            "score": score,
            "questions": [question.to_dict() for question in normalized_questions],
            "results": results,
            "errors": errors,
            "revision_hint": _revision_hint(errors),
        }


def should_run_factuality_qa(prompt: str, *, domain: str | None = None) -> bool:
    """Return true when I-HallA-style QA is appropriate."""

    if domain:
        return domain.strip().lower() in FACTUAL_DOMAINS
    return infer_prompt_domain(prompt) in FACTUAL_DOMAINS


def infer_prompt_domain(prompt: str) -> str:
    prompt = _clean_text(prompt, "prompt").lower()
    if any(word in prompt for word in ("diagram", "anatomy", "molecule", "cell", "planet", "map")):
        return "science"
    if any(word in prompt for word in ("historical", "ancient", "battle", "dynasty", "roman")):
        return "history"
    if any(word in prompt for word in ("textbook", "educational", "lesson", "classroom")):
        return "education"
    if any(word in prompt for word in ("medical", "surgery", "organ", "x-ray", "mri")):
        return "medical_diagram"
    return "creative"


def build_question_generation_request(
    user_prompt: str,
    *,
    domain: str | None = None,
    facts: Sequence[str] | None = None,
    question_count: int = 5,
) -> str:
    facts_blob = json.dumps(list(facts or []), ensure_ascii=False)
    schema = {
        "questions": [
            {
                "question": "visually answerable factual question",
                "choices": {"A": "choice", "B": "choice", "C": "choice", "D": "choice", "E": "None of the above"},
                "answer": "A",
                "coi": "existence|counting|color|shape|relation|scene|factuality",
                "rationale": "why this checks factuality",
            }
        ]
    }
    return "\n".join(
        [
            "You are an I-HallA-style QA builder for text-to-image factuality evaluation.",
            "Create visual multiple-choice questions whose answers can be verified from the image.",
            "Use E for None of the above when needed. Return exactly one JSON object.",
            f"Question count: {question_count}",
            f"Domain: {domain or infer_prompt_domain(user_prompt)}",
            f"Prompt: {user_prompt}",
            f"Known facts: {facts_blob}",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
        ]
    )


def build_vqa_answer_request(question: FactualQuestion) -> str:
    choices = "\n".join(f"{key}) {value}" for key, value in sorted(question.choices.items()))
    return "\n".join(
        [
            "You are an agent who answers questions based only on the given image.",
            "Choose the best answer choice or E for None of the above.",
            "Return only one character: A, B, C, D, or E.",
            f"Question: {question.question}",
            "Choices:",
            choices,
            "Answer:",
        ]
    )


def parse_questions_response(response: str) -> list[FactualQuestion]:
    data = _extract_json_object(_clean_text(response, "response"))
    if not data:
        return []
    raw_questions = data.get("questions", data.get("qas", []))
    if isinstance(raw_questions, Mapping):
        raw_questions = [raw_questions]
    if not isinstance(raw_questions, list):
        return []
    questions: list[FactualQuestion] = []
    for item in raw_questions:
        if not isinstance(item, Mapping):
            continue
        try:
            questions.append(FactualQuestion.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return questions


def parse_answer_response(response: str) -> str:
    text = _clean_text(response, "response").strip()
    match = re.search(r"\b([A-E])\b", text.upper())
    if match:
        return match.group(1)
    return "E"


def build_fallback_questions(
    user_prompt: str,
    *,
    facts: Sequence[str] | None = None,
    count: int = 5,
) -> list[FactualQuestion]:
    user_prompt = _clean_text(user_prompt, "user_prompt")
    facts = [str(item).strip() for item in facts or [] if str(item).strip()]
    base_fact = facts[0] if facts else user_prompt
    question = FactualQuestion(
        question="Which choice best matches the factual content requested by the prompt?",
        choices={
            "A": base_fact[:160],
            "B": "A visually unrelated scene",
            "C": "A scene with the main factual object missing",
            "D": "A scene with incorrect factual details",
            **DEFAULT_CHOICES,
        },
        answer="A",
        coi="factuality",
        rationale="Fallback factuality check built without an LLM question generator.",
    )
    return [deepcopy(question) for _ in range(max(1, count))]


def _normalize_questions(
    questions: Sequence[FactualQuestion | Mapping[str, Any]],
) -> list[FactualQuestion]:
    normalized: list[FactualQuestion] = []
    for question in questions:
        if isinstance(question, FactualQuestion):
            normalized.append(question)
        elif isinstance(question, Mapping):
            normalized.append(FactualQuestion.from_dict(question))
        else:
            raise TypeError("questions must contain FactualQuestion or mappings")
    return normalized


def _score_results(results: Sequence[Mapping[str, Any]]) -> float:
    if not results:
        return 0.0
    return round(sum(1 for item in results if item.get("passed")) / len(results), 6)


def _failed_question_evidence(result: Mapping[str, Any]) -> str:
    question = result.get("question", {})
    if not isinstance(question, Mapping):
        return "Factual QA failed."
    return (
        f"Question failed: {question.get('question')}; expected "
        f"{question.get('answer')}, got {result.get('predicted_answer')}."
    )


def _revision_hint(errors: Sequence[Mapping[str, Any]]) -> str:
    if not errors:
        return "Factual QA passed."
    return "Revise the prompt or regenerate the image to satisfy failed factual QA checks."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_answer_key(value: Any) -> str:
    key = str(value or "").strip().upper()[:1]
    if key not in {"A", "B", "C", "D", "E"}:
        raise ValueError("answer choices must use keys A-E")
    return key


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned
