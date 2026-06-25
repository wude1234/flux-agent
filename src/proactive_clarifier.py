"""Proactive clarification agent built on the project LLM adapter."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .belief_state import BeliefState, heuristic_belief_state
from .clients import LLMClient
from .state import CreativityLevel


THRESHOLDS = {
    CreativityLevel.LOW: 0.18,
    CreativityLevel.MEDIUM: 0.35,
    CreativityLevel.HIGH: 0.95,
}


class ProactiveClarifier:
    """Decide whether to ask one high-value clarification question."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        creativity_level: CreativityLevel | str = CreativityLevel.MEDIUM,
        ask_threshold: float | None = None,
    ) -> None:
        self.llm = llm
        self.creativity_level = CreativityLevel(creativity_level)
        self.ask_threshold = (
            float(ask_threshold)
            if ask_threshold is not None
            else THRESHOLDS[self.creativity_level]
        )

    def parse_belief_state(self, user_prompt: str) -> BeliefState:
        request = _belief_parse_request(user_prompt)
        response = self.llm.text(request)
        data = _extract_json_object(response)
        if data:
            try:
                return BeliefState.from_dict(data)
            except (KeyError, TypeError, ValueError):
                pass
        return heuristic_belief_state(user_prompt)

    def decide(
        self,
        user_prompt: str,
        *,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        belief = _ensure_belief_state(belief_state) if belief_state is not None else self.parse_belief_state(user_prompt)
        history_records = [deepcopy(dict(item)) for item in history or []]
        target = _first_unasked_target(belief.clarification_targets(), history_records)
        if target is None:
            return _do_not_ask_result(belief, 0.0, "No uncertain high-value target.")

        ask_score = float(target["ask_score"])
        if ask_score < self.ask_threshold:
            return _do_not_ask_result(
                belief,
                ask_score,
                f"Top target is below {self.creativity_level.value} threshold.",
                target,
            )

        question = self._question_for_target(user_prompt, belief, target, history_records)
        return {
            "status": "ask_user",
            "belief_state": belief.to_dict(),
            "question": question,
            "ask_score": ask_score,
            "missing_slot": target["missing_slot"],
            "target": deepcopy(target),
            "update": {
                "feedback": [
                    {
                        "source": "proactive_clarifier",
                        "question": question,
                        "ask_score": ask_score,
                        "missing_slot": target["missing_slot"],
                    }
                ]
            },
            "reasons": [
                "Target uncertainty and importance justify asking before generation."
            ],
        }

    def merge_answer(
        self,
        user_prompt: str,
        question: str,
        answer: str,
        *,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_prompt = _clean_text(user_prompt, "user_prompt")
        question = _clean_text(question, "question")
        answer = _clean_text(answer, "answer")
        belief = _ensure_belief_state(belief_state) if belief_state is not None else None
        request = _merge_answer_request(user_prompt, question, answer, belief)
        response = self.llm.text(request)
        merged_prompt = _extract_merged_prompt(response) or _fallback_merge(user_prompt, question, answer)
        return {
            "status": "merged",
            "merged_prompt": merged_prompt,
            "answer": answer,
            "question": question,
            "belief_state": belief.to_dict() if belief is not None else None,
            "update": {"user_prompt": merged_prompt},
            "raw_response": response,
        }

    def _question_for_target(
        self,
        user_prompt: str,
        belief: BeliefState,
        target: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        request = _question_request(user_prompt, belief, target, history)
        response = self.llm.text(request)
        question = _extract_tag(response, "question")
        if question:
            return question
        data = _extract_json_object(response)
        if data and data.get("question"):
            return str(data["question"]).strip()
        return _fallback_question(target)


def _belief_parse_request(user_prompt: str) -> str:
    return "\n".join(
        [
            "You are a proactive text-to-image belief parser.",
            "Return JSON with entities, relations, and prompt.",
            "Each entity needs name, importance_score, descriptions, entity_type,",
            "probability, and attributes. Each attribute needs name,",
            "importance_score, and candidates [{name, probability}].",
            "Use probabilities to represent uncertainty in user intent.",
            f"User prompt: {user_prompt}",
        ]
    )


def _question_request(
    user_prompt: str,
    belief: BeliefState,
    target: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> str:
    return "\n".join(
        [
            "You are the ClarifierAgent for a text-to-image system.",
            "Ask exactly one concise question that reduces the largest important uncertainty.",
            "Do not ask about details already present in the user prompt or dialogue.",
            "Wrap the question in <question> and </question>.",
            f"User prompt: {user_prompt}",
            f"Belief state: {json.dumps(belief.to_dict(), ensure_ascii=False, sort_keys=True)}",
            f"Selected target: {json.dumps(dict(target), ensure_ascii=False, sort_keys=True)}",
            f"Dialogue history: {json.dumps(list(history), ensure_ascii=False, sort_keys=True)}",
        ]
    )


def _merge_answer_request(
    user_prompt: str,
    question: str,
    answer: str,
    belief: BeliefState | None,
) -> str:
    belief_blob = json.dumps(belief.to_dict(), ensure_ascii=False, sort_keys=True) if belief else "{}"
    return "\n".join(
        [
            "You are writing a prompt for a text-to-image model based on user feedback.",
            "Merge the answer into the original prompt without adding unrelated details.",
            "Return JSON with merged_prompt, or wrap the merged prompt in <prompt> tags.",
            f"Original prompt: {user_prompt}",
            f"Question: {question}",
            f"Answer: {answer}",
            f"Belief state: {belief_blob}",
        ]
    )


def _first_unasked_target(
    targets: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    asked_slots = {
        str(item.get("missing_slot", "")).lower()
        for item in history
        if item.get("missing_slot")
    }
    asked_questions = {
        str(item.get("question", "")).lower()
        for item in history
        if item.get("question")
    }
    for target in targets:
        slot = str(target.get("missing_slot", "")).lower()
        question = _fallback_question(target).lower()
        if slot in asked_slots or question in asked_questions:
            continue
        return deepcopy(dict(target))
    return None


def _do_not_ask_result(
    belief: BeliefState,
    ask_score: float,
    reason: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "do_not_ask",
        "belief_state": belief.to_dict(),
        "question": None,
        "ask_score": ask_score,
        "missing_slot": target.get("missing_slot") if target else None,
        "target": deepcopy(dict(target)) if target else None,
        "update": {},
        "reasons": [reason],
    }


def _fallback_question(target: Mapping[str, Any]) -> str:
    kind = target.get("kind")
    candidates = [
        str(candidate.get("name"))
        for candidate in target.get("candidates", [])
        if isinstance(candidate, Mapping) and candidate.get("name")
    ]
    options = f" Options: {', '.join(candidates)}." if candidates else ""
    if kind == "attribute":
        return f"What {target.get('attribute')} should the {target.get('entity')} have?{options}"
    if kind == "entity":
        return f"Should the {target.get('entity')} appear in the image?{options}"
    if kind == "relation":
        return (
            f"What spatial relation should {target.get('entity_1')} have to"
            f" {target.get('entity_2')}?{options}"
        )
    return f"What detail should be specified for {target.get('missing_slot', 'the image')}?{options}"


def _fallback_merge(user_prompt: str, question: str, answer: str) -> str:
    del question
    if answer.lower() in user_prompt.lower():
        return user_prompt
    return f"{user_prompt}, with this clarified detail: {answer}"


def _extract_merged_prompt(response: str) -> str | None:
    data = _extract_json_object(response)
    if data and data.get("merged_prompt"):
        return str(data["merged_prompt"]).strip()
    return _extract_tag(response, "prompt")


def _ensure_belief_state(value: BeliefState | Mapping[str, Any]) -> BeliefState:
    if isinstance(value, BeliefState):
        return value
    if isinstance(value, Mapping):
        return BeliefState.from_dict(value)
    raise TypeError("belief_state must be BeliefState or mapping")


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


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value
