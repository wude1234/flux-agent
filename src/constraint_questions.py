"""Question-level user-constraint checks for M6.19.

The evaluator turns extracted prompt constraints into small VQA-style questions
with dependencies. This keeps base-image selection grounded in visible evidence:
first required objects, then counts/attributes/parts, then actions/relations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence

from .clients import VLMClient
from .prompt_constraints import (
    COLOR_WORDS,
    RELATION_CONNECTOR_WORDS,
    PromptConstraints,
    extract_constraints,
)


YES_NO_CHOICES = ["yes", "no", "uncertain"]
COLOR_CHOICES = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "grey",
    "purple",
    "pink",
    "orange",
    "brown",
    "lavender",
    "amber",
    "navy",
    "mint",
    "ruby",
    "bronze",
    "lime",
    "maroon",
    "ivory",
    "olive",
    "violet",
    "aqua",
    "silver",
    "gold",
    "golden",
    "teal",
    "cyan",
    "magenta",
    "crimson",
    "turquoise",
    "uncertain",
]

HARD_CATEGORIES = {
    "entity_existence",
    "count",
    "color_binding",
    "part_visibility",
    "action_relation",
    "spatial_relation",
    "symbol_text_relation",
    "negative_symbol_text_relation",
    "negative_object_existence",
    "attribute_relation_binding",
}

ENTITY_DROP_WORDS = {
    "a",
    "an",
    "the",
    "small",
    "large",
    "tiny",
    "big",
    "visible",
    "clearly",
    "dominant",
    "clean",
    "plain",
    "studio",
    "realistic",
    "outdoor",
    "wooden",
    "leakage",
    "no",
    *COLOR_WORDS,
}

ENTITY_TRAILING_STOP_WORDS = {
    "and",
    "are",
    "is",
    "be",
    "been",
    "being",
    "contain",
    "contains",
    "containing",
    "display",
    "displays",
    "displaying",
    "hang",
    "hangs",
    "hanging",
    "lie",
    "lies",
    "lying",
    "perch",
    "perches",
    "perching",
    "remain",
    "remains",
    "rest",
    "rests",
    "resting",
    "show",
    "shows",
    "showing",
    "sit",
    "sits",
    "sitting",
    "stand",
    "stands",
    "standing",
    "swim",
    "swims",
    "swimming",
}

NON_ENTITY_TERMS = {
    "left",
    "wooden",
    "leakage",
    "no",
    *RELATION_CONNECTOR_WORDS,
}

NUMBER_ENTITY_TERMS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}

ENTITY_DROP_WORDS.update(NUMBER_ENTITY_TERMS)
NON_ENTITY_TERMS.update(NUMBER_ENTITY_TERMS)

SUPPORT_OBJECTS = {
    "table",
    "wooden table",
    "floor",
    "ground",
    "grass",
    "road",
    "street",
    "sidewalk",
}

DISPLAY_ACTIONS = {
    "show",
    "shows",
    "showing",
    "display",
    "displays",
    "displaying",
}

PART_NAME_WORDS = {
    "beak",
    "claw",
    "handle",
    "hand",
    "leash",
    "seat",
    "tail",
    "top",
    "wheel",
    "wing",
}


@dataclass(frozen=True)
class ConstraintQuestion:
    """One atomic visual question derived from the original user prompt."""

    id: str
    category: str
    question: str
    answer_type: str
    expected_answer: str
    choices: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    priority: str = "hard"
    source_constraint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "answer_type": self.answer_type,
            "choices": list(self.choices),
            "expected_answer": self.expected_answer,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "source_constraint": deepcopy(self.source_constraint),
        }


@dataclass(frozen=True)
class ConstraintAnswer:
    """Normalized answer to one ``ConstraintQuestion``."""

    id: str
    raw_answer: str
    normalized_answer: str
    passed: bool | None
    confidence: float | None = None
    evidence: str = ""
    blocked_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_answer": self.raw_answer,
            "normalized_answer": self.normalized_answer,
            "passed": self.passed,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "blocked_by": list(self.blocked_by),
        }


class VQAConstraintEvaluator:
    """Evaluate prompt constraints through the existing ``VLMClient`` adapter."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm

    def evaluate(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        constraints: PromptConstraints | Mapping[str, Any] | str | None = None,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        constraints = _ensure_constraints(constraints or user_prompt)
        questions = generate_constraint_questions(constraints)
        if not questions:
            return _empty_record(user_prompt, prompt, image_path)

        request = build_vqa_constraint_request(
            user_prompt=user_prompt,
            prompt=prompt,
            image_path=image_path,
            questions=questions,
            history=history,
        )
        raw_response = self.vlm.vision(request, [image_path])
        parsed = parse_vqa_constraint_response(raw_response, questions)
        if parsed.get("legacy_constraint_check"):
            check = deepcopy(dict(parsed["legacy_constraint_check"]))
            check.update(
                {
                    "image_path": image_path,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "request": request,
                    "source": "legacy_constraint_check",
                    "question_count": len(questions),
                }
            )
            return {
                "image_path": image_path,
                "prompt": prompt,
                "questions": [question.to_dict() for question in questions],
                "answers": [],
                "summary": {
                    "passed": check.get("passed"),
                    "hard_failures": _hard_failure_count(check),
                    "failed_constraints": _failed_constraint_ids(check),
                    "passed_constraints": _passed_constraint_ids(check),
                    "blocked_constraints": [],
                    "source": "legacy_constraint_check",
                },
                "constraint_check": check,
                "raw_response": raw_response,
                "request": request,
                "source": "legacy_constraint_check",
            }

        answers = _apply_dependencies(parsed["answers"], questions)
        summary = summarize_question_answers(questions, answers)
        check = question_answers_to_constraint_check(
            questions,
            answers,
            summary,
            prompt=prompt,
            image_path=image_path,
            raw_response=raw_response,
            request=request,
        )
        return {
            "image_path": image_path,
            "prompt": prompt,
            "questions": [question.to_dict() for question in questions],
            "answers": [answer.to_dict() for answer in answers],
            "summary": summary,
            "constraint_check": check,
            "raw_response": raw_response,
            "request": request,
            "source": "question_level_vqa",
        }


def generate_constraint_questions(
    constraints: PromptConstraints | Mapping[str, Any] | str,
) -> list[ConstraintQuestion]:
    """Create dependency-ordered questions from user-intent constraints."""

    constraints = _ensure_constraints(constraints)
    prompt = constraints.original_prompt
    colors = _clean_color_constraints(constraints.colors)
    counts = _canonical_count_constraints(
        _intent_counts(constraints),
        _extract_count_constraints(prompt),
        constraints.subjects,
        colors,
    )
    entities = _ordered_entities(prompt, colors, constraints.subjects, counts)
    parts = _extract_part_constraints(prompt, entities)
    color_entities = {_normalize_entity_name_keep_color(item) for item in colors.keys()}
    questions: list[ConstraintQuestion] = []
    seen: set[str] = set()

    for entity in entities:
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=f"existence:{_slug(entity)}",
                category="entity_existence",
                question=f"Is there a visible {entity} in the image?",
                answer_type="choice",
                choices=list(YES_NO_CHOICES),
                expected_answer="yes",
                source_constraint={"object": entity},
            ),
        )

    for entity, count in counts.items():
        if entity not in colors and _covered_by_color_entity(entity, color_entities):
            continue
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=f"count:{_slug(entity)}",
                category="count",
                question=f"How many visible {entity} are in the image?",
                answer_type="number",
                expected_answer=str(count),
                choices=["0", "1", "2", "3", "4", "5", "uncertain"],
                depends_on=[f"existence:{_slug(entity)}"],
                source_constraint={"object": entity, "count": count},
            ),
        )

    for entity, color in colors.items():
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=f"color:{_slug(entity)}",
                category="color_binding",
                question=_color_question(entity),
                answer_type="short_text",
                choices=list(COLOR_CHOICES),
                expected_answer=color,
                depends_on=[f"existence:{_slug(entity)}"],
                source_constraint={"object": entity, "attribute": "color", "value": color},
            ),
        )

    for part in parts:
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=part["id"],
                category="part_visibility",
                question=f"Is the {part['name']} clearly visible in the image?",
                answer_type="choice",
                choices=list(YES_NO_CHOICES),
                expected_answer="yes",
                depends_on=[f"existence:{_slug(part['parent'])}"],
                source_constraint=deepcopy(part),
            ),
        )

    relations = _extract_action_relation_constraints(
        prompt,
        entities,
        parts,
        constraints.actions,
        counts,
        intent_spec=(
            constraints.intent_spec.to_dict()
            if constraints.intent_spec is not None
            else None
        ),
    )
    relations.extend(
        _negative_relation_constraints(
            (
                constraints.intent_spec.negative_constraints
                if constraints.intent_spec is not None
                else []
            ),
            entities,
        )
    )
    relations.extend(
        _negative_symbol_text_constraints(
            (
                constraints.intent_spec.negative_constraints
                if constraints.intent_spec is not None
                else []
            ),
            entities,
            (
                constraints.intent_spec.to_dict()
                if constraints.intent_spec is not None
                else None
            ),
        )
    )
    relations.extend(
        _negative_object_existence_constraints(
            (
                constraints.intent_spec.negative_constraints
                if constraints.intent_spec is not None
                else []
            ),
            entities,
        )
    )
    relations.extend(_attribute_relation_binding_constraints(relations, colors))
    for relation in relations:
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=relation["id"],
                category=relation.get("category", "action_relation"),
                question=relation["question"],
                answer_type="choice",
                choices=list(YES_NO_CHOICES),
                expected_answer="yes",
                depends_on=list(relation.get("depends_on", [])),
                source_constraint=deepcopy(relation),
            ),
        )

    for phrase in _scene_style_phrases(constraints.protected_phrases, colors):
        _append_question(
            questions,
            seen,
            ConstraintQuestion(
                id=f"scene:{_slug(phrase)}",
                category="scene_style",
                question=f"Does the image visibly match this scene/style phrase: {phrase}?",
                answer_type="choice",
                choices=list(YES_NO_CHOICES),
                expected_answer="yes",
                priority="soft",
                source_constraint={"phrase": phrase},
            ),
        )

    return questions


def build_vqa_constraint_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    questions: Sequence[ConstraintQuestion],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    question_payload = [question.to_dict() for question in questions]
    history_blob = json.dumps(
        _compact_history_for_request(history or []),
        ensure_ascii=False,
        sort_keys=True,
    )
    return "\n".join(
        [
            "You are a strict visual constraint checker and question-level VQA verifier.",
            "Answer from the image only. Do not infer from the prompt.",
            "The original user prompt is only used to know what to check.",
            "If visible evidence is unclear, answer uncertain.",
            "Use the dependency order: objects first, then counts, attributes, parts, relations.",
            "For count questions, count only distinct visible instances of the requested target class.",
            "Do not count other species/objects, hidden objects, shadows, or inferred offscreen objects.",
            "For action questions, require visible pose/contact evidence; do not infer action from prompt context.",
            "For gripping/holding/carrying, the relevant hand, claw, or contact point must visibly grasp, wrap around, or hold the target part/object.",
            "Do not mark gripping/holding true merely because an object is attached, mounted, supported by the body, or visually close.",
            "For symbol_text_relation questions, check whether the requested visible text, symbol, logo, or mark appears on the specified carrier object; do not require physical contact or hand-object grasp evidence.",
            "For negative_symbol_text_relation questions, answer yes only if the forbidden text/symbol/logo/mark is absent from the specified comparison object.",
            "For negative_object_existence questions, answer yes only if the specified forbidden object is completely absent from the entire image; any visible instance of that object class means the answer is no.",
            "For left/right questions, compare the subject against the referenced object group in the image.",
            "Return exactly one JSON object with this shape:",
            '{"answers":[{"id":"...","answer":"yes|no|uncertain or short value","confidence":0.0,"evidence":"visible evidence"}]}',
            f"Original user prompt with binding constraints: {user_prompt}",
            f"Expanded prompt for non-binding context only, do not add constraints from it: {prompt}",
            f"Image path: {image_path}",
            f"Prior feedback JSON: {history_blob}",
            "Questions JSON:",
            json.dumps(question_payload, ensure_ascii=False, sort_keys=True),
        ]
    )


def _compact_history_for_request(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(history)[-2:]:
        if not isinstance(item, Mapping):
            continue
        feedback = item.get("feedback")
        if not isinstance(feedback, Mapping):
            feedback = item.get("critique")
        entry: dict[str, Any] = {
            "round": item.get("round"),
            "source": item.get("source"),
        }
        if isinstance(feedback, Mapping):
            check = feedback.get("constraint_check")
            entry["score"] = feedback.get("score")
            entry["errors"] = _compact_error_records(feedback.get("errors", []))
            entry["revision_hint"] = _truncate_request_text(
                str(feedback.get("revision_hint") or ""),
                800,
            )
            if isinstance(check, Mapping):
                entry["constraint_check"] = {
                    "passed": check.get("passed"),
                    "failed": check.get("failed"),
                    "score": check.get("score"),
                    "source": check.get("source"),
                    "question_summary": deepcopy(
                        dict(check.get("question_summary", {}))
                    )
                    if isinstance(check.get("question_summary"), Mapping)
                    else {},
                    "errors": _compact_error_records(check.get("errors", [])),
                }
        selection = item.get("selection")
        if isinstance(selection, Mapping):
            entry["selection"] = {
                "selected_index": selection.get("selected_index"),
                "selected_image": selection.get("selected_image"),
            }
        compact.append(entry)
    return compact


def _compact_error_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        records = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = [item for item in value if isinstance(item, Mapping)]
    else:
        records = []
    compact: list[dict[str, Any]] = []
    for item in records[:10]:
        compact.append(
            {
                "type": item.get("type"),
                "category": item.get("category"),
                "question_id": item.get("question_id"),
                "target": item.get("target"),
                "expected": item.get("expected"),
                "observed": item.get("observed"),
                "prompt_span": item.get("prompt_span"),
                "evidence": _truncate_request_text(
                    str(item.get("evidence") or item.get("description") or ""),
                    500,
                ),
            }
        )
    return compact


def _truncate_request_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def parse_vqa_constraint_response(
    response: str,
    questions: Sequence[ConstraintQuestion],
) -> dict[str, Any]:
    data = _extract_json(response)
    if isinstance(data, Mapping) and "checks" in data and not _answer_items(data):
        return {"answers": [], "legacy_constraint_check": _legacy_constraint_check(data, response)}

    raw_answers = _answer_items(data)
    if not raw_answers:
        raw_answers = [_single_text_answer(response, questions[0])] if questions else []

    answers_by_id: dict[str, ConstraintAnswer] = {}
    for index, question in enumerate(questions):
        raw = _matching_answer(raw_answers, question, index)
        answers_by_id[question.id] = _normalize_answer(raw, question)
    return {"answers": [answers_by_id[question.id] for question in questions]}


def summarize_question_answers(
    questions: Sequence[ConstraintQuestion],
    answers: Sequence[ConstraintAnswer],
) -> dict[str, Any]:
    question_by_id = {question.id: question for question in questions}
    passed_constraints: list[str] = []
    failed_constraints: list[str] = []
    blocked_constraints: list[str] = []
    hard_failures = 0
    uncertain_hard_checks = 0
    soft_failures = 0

    for answer in answers:
        question = question_by_id.get(answer.id)
        if question is None:
            continue
        if answer.blocked_by:
            blocked_constraints.append(answer.id)
            continue
        if answer.passed is True:
            passed_constraints.append(answer.id)
            continue
        failed_constraints.append(answer.id)
        if question.priority == "hard":
            hard_failures += 1
            if answer.normalized_answer == "uncertain":
                uncertain_hard_checks += 1
        else:
            soft_failures += 1

    passed = hard_failures == 0
    score = _score_from_summary(
        passed=len(passed_constraints),
        failed=len(failed_constraints),
        blocked=len(blocked_constraints),
        hard_failures=hard_failures,
    )
    return {
        "passed": passed,
        "score": score,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "uncertain_hard_checks": uncertain_hard_checks,
        "failed_constraints": failed_constraints,
        "passed_constraints": passed_constraints,
        "blocked_constraints": blocked_constraints,
    }


def question_answers_to_constraint_check(
    questions: Sequence[ConstraintQuestion],
    answers: Sequence[ConstraintAnswer],
    summary: Mapping[str, Any],
    *,
    prompt: str,
    image_path: str,
    raw_response: str = "",
    request: str = "",
) -> dict[str, Any]:
    question_by_id = {question.id: question for question in questions}
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for answer in answers:
        question = question_by_id.get(answer.id)
        if question is None or answer.blocked_by:
            continue
        check = _check_from_answer(question, answer)
        checks.append(check)
        if answer.passed is False:
            errors.append(_error_from_answer(question, answer))

    revision_hint = _revision_hint(errors)
    return {
        "passed": bool(summary.get("passed")),
        "score": float(summary.get("score", 0.0) or 0.0),
        "constraint_score": float(summary.get("score", 0.0) or 0.0),
        "checks": checks,
        "errors": _dedupe_errors(errors),
        "strengths": _strengths_from_checks(checks),
        "revision_hint": revision_hint,
        "user_grounded": True,
        "question_summary": deepcopy(dict(summary)),
        "image_path": image_path,
        "prompt": prompt,
        "raw_response": raw_response,
        "request": request,
        "source": "question_level_vqa",
    }


def _apply_dependencies(
    answers: Sequence[ConstraintAnswer],
    questions: Sequence[ConstraintQuestion],
) -> list[ConstraintAnswer]:
    answer_by_id = {answer.id: answer for answer in answers}
    result: list[ConstraintAnswer] = []
    for question in questions:
        answer = answer_by_id.get(question.id)
        if answer is None:
            answer = ConstraintAnswer(
                id=question.id,
                raw_answer="uncertain",
                normalized_answer="uncertain",
                passed=False,
                confidence=None,
                evidence="No answer returned by VLM.",
            )
        blocked_by = [
            dep
            for dep in question.depends_on
            if (answer_by_id.get(dep) is None or answer_by_id[dep].passed is not True)
        ]
        if blocked_by:
            answer = ConstraintAnswer(
                id=answer.id,
                raw_answer=answer.raw_answer,
                normalized_answer="blocked",
                passed=None,
                confidence=answer.confidence,
                evidence=answer.evidence,
                blocked_by=blocked_by,
            )
        answer_by_id[question.id] = answer
        result.append(answer)
    return result


def _normalize_answer(raw: Mapping[str, Any], question: ConstraintQuestion) -> ConstraintAnswer:
    raw_answer = str(
        raw.get("answer", raw.get("normalized_answer", raw.get("value", "")))
        or ""
    ).strip()
    if not raw_answer:
        raw_answer = "uncertain"
    normalized = _normalize_answer_value(raw_answer, question)
    passed = _answer_matches(normalized, question)
    confidence = _coerce_float(raw.get("confidence"), default=None)
    evidence = str(raw.get("evidence", raw.get("reason", raw.get("description", ""))) or "")
    if passed and _unsupported_contact_relation(question, evidence):
        normalized = "no"
        passed = False
    return ConstraintAnswer(
        id=question.id,
        raw_answer=raw_answer,
        normalized_answer=normalized,
        passed=passed,
        confidence=confidence,
        evidence=evidence,
    )


def _normalize_answer_value(raw_answer: str, question: ConstraintQuestion) -> str:
    text = raw_answer.strip().lower()
    if not text:
        return "uncertain"
    if question.answer_type == "number":
        number = _number_from_text(text)
        return str(number) if number is not None else "uncertain"
    if question.answer_type == "choice":
        if _looks_uncertain(text):
            return "uncertain"
        if _looks_yes(text):
            return "yes"
        if _looks_no(text):
            return "no"
        for choice in question.choices:
            if choice.lower() in text:
                return choice.lower()
        return "uncertain"
    if question.answer_type == "short_text":
        if _looks_uncertain(text):
            return "uncertain"
        for color in sorted(COLOR_WORDS | {"gray", "grey"}, key=len, reverse=True):
            if re.search(rf"\b{re.escape(color)}\b", text):
                return color
        return text.split()[0] if text.split() else "uncertain"
    return text


def _answer_matches(normalized: str, question: ConstraintQuestion) -> bool:
    expected = str(question.expected_answer).strip().lower()
    if normalized in {"uncertain", "blocked"}:
        return False
    if question.answer_type == "number":
        return normalized == expected
    if question.category == "color_binding":
        if expected == "gray":
            return normalized in {"gray", "grey"}
        if expected == "grey":
            return normalized in {"gray", "grey"}
    return normalized == expected


def _unsupported_contact_relation(question: ConstraintQuestion, evidence: str) -> bool:
    """Reject relation yes answers that only prove attachment/support.

    This is intentionally generic for hand-object action prompts. A relation
    like gripping/holding requires visible effector evidence, not just a pole
    attached to a body or an object sitting near the subject.
    """

    if question.category not in {"action_relation", "attribute_relation_binding"}:
        return False
    source = question.source_constraint
    action_text = " ".join(
        str(source.get(key) or "") for key in ("action", "relation")
    ).lower()
    question_text = question.question.lower()
    relation_text = f"{action_text} {question_text}"
    if not any(word in relation_text for word in ("grip", "grasp", "hold", "holding", "carry", "carrying")):
        return False
    evidence_text = str(evidence or "").lower()
    if not evidence_text:
        return False
    weak_markers = (
        "attached",
        "connected",
        "mounted",
        "supported",
        "rests on",
        "resting on",
        "inserted",
        "stuck",
        "fixed to",
        "emerging from",
        "on top of",
    )
    strong_markers = (
        "hand",
        "claw",
        "gripper",
        "finger",
        "fingers",
        "paw",
        "arm",
        "wrap",
        "wrapped",
        "around",
        "contact point",
        "end-effector",
    )
    has_weak = any(_contains_evidence_marker(evidence_text, marker) for marker in weak_markers)
    has_strong = any(_contains_evidence_marker(evidence_text, marker) for marker in strong_markers)
    return has_weak and not has_strong


def _contains_evidence_marker(text: str, marker: str) -> bool:
    marker_pattern = re.escape(marker).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){marker_pattern}(?![a-z0-9])", text))


def _check_from_answer(
    question: ConstraintQuestion,
    answer: ConstraintAnswer,
) -> dict[str, Any]:
    check_type = _check_type(question)
    target = _target_from_question(question)
    return {
        "type": check_type,
        "target": target,
        "expected": question.expected_answer,
        "observed": answer.normalized_answer,
        "passed": answer.passed is True,
        "description": answer.evidence or question.question,
        "question_id": question.id,
        "category": question.category,
    }


def _error_from_answer(
    question: ConstraintQuestion,
    answer: ConstraintAnswer,
) -> dict[str, Any]:
    error_type = _error_type(question)
    target = _target_from_question(question)
    evidence = answer.evidence.strip()
    if not evidence:
        evidence = (
            f"Question {question.id} expected {question.expected_answer!r} "
            f"but got {answer.normalized_answer!r}."
        )
    return {
        "type": error_type,
        "evidence": evidence,
        "prompt_span": target,
        "question_id": question.id,
    }


def _legacy_constraint_check(data: Mapping[str, Any], response: str) -> dict[str, Any]:
    checks = _list_records(data.get("checks", data.get("constraints", [])))
    errors = _list_records(data.get("errors", []))
    for check in checks:
        if check.get("passed") is False:
            errors.append(
                {
                    "type": _legacy_error_type(check),
                    "evidence": str(
                        check.get("description")
                        or check.get("observed")
                        or "Constraint check failed."
                    ),
                    "prompt_span": str(check.get("target") or check.get("expected") or ""),
                }
            )
    errors = _dedupe_errors(errors)
    score = _normalize_score(data.get("score", data.get("constraint_score", 0.5)))
    passed_value = data.get("passed", data.get("all_passed"))
    passed = bool(passed_value) if passed_value is not None else score >= 0.85
    if errors:
        passed = False
    return {
        "passed": passed,
        "score": score,
        "constraint_score": score,
        "checks": checks,
        "errors": errors,
        "strengths": _list_text(data.get("strengths", [])),
        "revision_hint": str(
            data.get("revision_hint")
            or data.get("feedback")
            or _first_sentence(response)
            or ""
        ),
        "user_grounded": True,
    }


def _empty_record(user_prompt: str, prompt: str, image_path: str) -> dict[str, Any]:
    del user_prompt
    check = {
        "passed": True,
        "score": 1.0,
        "constraint_score": 1.0,
        "checks": [],
        "errors": [],
        "strengths": [],
        "revision_hint": "No explicit user constraints were extracted.",
        "user_grounded": True,
        "image_path": image_path,
        "prompt": prompt,
        "source": "question_level_vqa",
    }
    return {
        "image_path": image_path,
        "prompt": prompt,
        "questions": [],
        "answers": [],
        "summary": {
            "passed": True,
            "score": 1.0,
            "hard_failures": 0,
            "soft_failures": 0,
            "uncertain_hard_checks": 0,
            "failed_constraints": [],
            "passed_constraints": [],
            "blocked_constraints": [],
        },
        "constraint_check": check,
        "source": "question_level_vqa",
    }


def _clean_color_constraints(colors: Mapping[str, Any]) -> dict[str, str]:
    raw_items: list[tuple[str, str, str]] = []
    head_counts: dict[str, int] = {}
    for raw_object, raw_color in colors.items():
        raw_entity = _normalize_entity_name_keep_color(str(raw_object))
        clean_entity = _clean_entity_name(str(raw_object))
        color = str(raw_color or "").strip().lower()
        if not clean_entity or not color:
            continue
        raw_items.append((raw_entity, clean_entity, color))
        head_counts[clean_entity] = head_counts.get(clean_entity, 0) + 1

    result: dict[str, str] = {}
    for raw_entity, clean_entity, color in raw_items:
        entity = raw_entity if head_counts.get(clean_entity, 0) > 1 else clean_entity
        if entity and color and entity not in result:
            result[entity] = color
    return result


def _intent_counts(constraints: PromptConstraints) -> dict[str, int]:
    intent = constraints.intent_spec
    if intent is None:
        return {}
    return {
        _clean_entity_name(str(entity)): int(count)
        for entity, count in intent.counts.items()
        if _clean_entity_name(str(entity))
    }


def _canonical_count_constraints(
    intent_counts: Mapping[str, int],
    extracted_counts: Mapping[str, int],
    subjects: Sequence[str],
    colors: Mapping[str, str],
) -> dict[str, int]:
    """Prefer IntentSpec noun targets and strip grammar fragments from counts."""

    canonical_entities = [
        _canonical_entity_fragment(entity)
        for entity in [*subjects, *colors.keys(), *intent_counts.keys()]
    ]
    canonical_entities = [
        entity
        for entity in dict.fromkeys(canonical_entities)
        if entity and not _is_style_or_attribute(entity)
    ]
    counts: dict[str, int] = {}
    for raw_entity, raw_count in [*extracted_counts.items(), *intent_counts.items()]:
        entity = _canonical_entity_fragment(str(raw_entity))
        if not entity:
            continue
        if entity in extracted_counts and entity not in intent_counts and _is_mass_or_scene_entity(entity):
            continue
        matched = _best_entity_match(entity, canonical_entities)
        entity = matched or entity
        if str(raw_entity) in extracted_counts and str(raw_entity) not in intent_counts and _is_mass_or_scene_entity(entity):
            continue
        if not entity or _is_style_or_attribute(entity):
            continue
        if entity in SUPPORT_OBJECTS and entity not in intent_counts:
            continue
        try:
            counts[entity] = int(raw_count)
        except (TypeError, ValueError):
            continue
    for entity, count in intent_counts.items():
        cleaned = _canonical_entity_fragment(entity)
        if cleaned:
            counts[cleaned] = int(count)
    return counts


def _ordered_entities(
    prompt: str,
    colors: Mapping[str, str],
    subjects: Sequence[str],
    counts: Mapping[str, int],
) -> list[str]:
    candidates: list[str] = []
    color_entities = {_normalize_entity_name_keep_color(item) for item in colors.keys()}
    for value in [*counts.keys(), *colors.keys(), *subjects]:
        raw_entity = _normalize_entity_name_keep_color(str(value))
        entity = raw_entity if raw_entity in color_entities else _clean_entity_name(str(value))
        entity = _canonical_entity_fragment(entity)
        if (
            entity
            and entity not in candidates
            and not _is_style_or_attribute(entity)
            and not _is_minor_part_name(entity)
            and not _covered_by_color_entity(entity, color_entities)
        ):
            candidates.append(entity)
    lowered = prompt.lower()
    original_order = {item: index for index, item in enumerate(candidates)}
    candidates.sort(
        key=lambda item: (
            _find_entity_position(lowered, item),
            original_order.get(item, 10**8),
        )
    )
    return candidates[:10]


def _extract_count_constraints(prompt: str) -> dict[str, int]:
    lowered = prompt.lower()
    number_alt = "|".join(sorted(_NUMBER_WORDS.keys(), key=len, reverse=True))
    modifiers = "|".join(sorted(COLOR_WORDS | {"small", "large", "tiny", "big", "visible"}))
    pattern = re.compile(
        rf"\b(?P<count>\d+|{number_alt})\s+"
        rf"(?:(?:{modifiers})\s+)?"
        r"(?P<object>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})"
    )
    counts: dict[str, int] = {}
    for match in pattern.finditer(lowered):
        number = _number_from_text(match.group("count"))
        entity = _canonical_entity_fragment(_clean_entity_name(match.group("object")))
        if number is not None and entity:
            counts[entity] = number
    for entity in _extract_singular_count_constraints(lowered):
        counts.setdefault(entity, 1)
    return counts


MASS_OR_SCENE_COUNT_TARGETS = {
    "bread",
    "freshly baked bread",
    "food",
    "fruit",
    "rice",
    "grass",
    "water",
    "sand",
    "snow",
    "bakery",
    "garden",
    "greenhouse",
    "street",
    "sky",
    "cloud",
    "clouds",
}


def _is_mass_or_scene_entity(entity: str) -> bool:
    entity = _canonical_entity_fragment(entity)
    if not entity:
        return False
    if entity in MASS_OR_SCENE_COUNT_TARGETS:
        return True
    return any(
        entity.endswith(f" {target}")
        for target in ("bread", "food", "fruit", "grass", "water", "sand", "snow")
    )


def _extract_singular_count_constraints(prompt: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b(?:a|an|single)\b", prompt):
        raw_object = _next_entity_phrase(prompt[match.end() :])
        entity = _canonical_entity_fragment(_clean_entity_name(raw_object))
        if not entity:
            continue
        if entity in SUPPORT_OBJECTS or _is_style_or_attribute(entity):
            continue
        if entity in {"group", "pair", "couple", "set", "bunch", "cluster"}:
            continue
        if _looks_plural_entity(entity):
            continue
        if _is_mass_or_scene_entity(entity):
            continue
        if entity not in entities:
            entities.append(entity)
    return entities


def _next_entity_phrase(text: str, *, max_words: int = 5) -> str:
    # Singular-count extraction starts immediately after "a/an/single".  Stop
    # at punctuation and local clause boundaries so trailing modifiers such as
    # "while sitting..." or "no color leakage" cannot become pseudo-entities.
    text = re.split(r"[,;.]", text, maxsplit=1)[0]
    text = re.split(
        r"\b(?:while|without|avoid|not|no)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    words = re.findall(r"[a-z0-9-]+", text.lower())
    return " ".join(words[:max_words])


def _extract_part_constraints(prompt: str, entities: Sequence[str]) -> list[dict[str, Any]]:
    lowered = prompt.lower()
    parts: list[dict[str, Any]] = []
    for match in re.finditer(
        r"\b(handle|hand|claw|wheel|seat|leash|tail|wing|beak|top)\s+of\s+(?:an?\s+|the\s+)?(?:(?:"
        + "|".join(sorted(COLOR_WORDS))
        + r")\s+)?(?P<object>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})",
        lowered,
    ):
        part_type = match.group(1)
        parent = _best_entity_match(_clean_entity_name(match.group("object")), entities)
        if parent:
            parts.append(_part_record(parent, part_type))
    for entity in entities:
        for part_type in ("handle", "wheel", "seat"):
            if re.search(rf"\b{re.escape(entity)}\s+{part_type}\b", lowered):
                parts.append(_part_record(entity, part_type))
    return _dedupe_parts(parts)


def _extract_action_relation_constraints(
    prompt: str,
    entities: Sequence[str],
    parts: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    counts: Mapping[str, int],
    *,
    intent_spec: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    lowered = prompt.lower()
    relations: list[dict[str, Any]] = []
    for relation in _structured_relation_constraints(intent_spec, counts, parts):
        relations.append(relation)
    handle_part = next((part for part in parts if part.get("part_type") == "handle"), None)
    if handle_part and any(action in lowered for action in ("gripping", "holding")):
        target_parent = str(handle_part["parent"])
        subject = next((entity for entity in entities if entity != target_parent), "")
        if subject:
            action = "gripping" if "gripping" in lowered else "holding"
            part_id = str(handle_part["id"])
            relations.append(
                {
                    "id": f"relation:{_slug(subject)}:{_slug(target_parent)}_handle:{_slug(action)}",
                    "category": "action_relation",
                    "question": (
                        f"Are the {subject}'s relevant hands, claws, or contact points "
                        f"visibly {action} the {target_parent} handle, with clear grasp/contact "
                        "rather than the handle merely attached to, mounted on, or supported by "
                        f"the {subject}'s body?"
                    ),
                    "depends_on": [
                        f"existence:{_slug(subject)}",
                        f"existence:{_slug(target_parent)}",
                        part_id,
                    ],
                    "subject": subject,
                    "object": target_parent,
                    "part": f"{target_parent} handle",
                    "action": action,
                }
            )

    for relation in _explicit_action_object_relations(lowered, entities):
        if (
            relation["id"] not in {item["id"] for item in relations}
            and not _overlaps_existing_relation(relation, relations)
        ):
            relations.append(relation)

    for relation in _explicit_spatial_relations_from_prompt(lowered, entities, counts):
        if relation["id"] not in {item["id"] for item in relations}:
            relations.append(relation)

    for relation in _binary_relations_from_prompt(lowered, entities, counts):
        if relation["id"] not in {item["id"] for item in relations}:
            relations.append(relation)

    for action in actions:
        action = str(action).strip().lower()
        if (
            not action
            or action in {"hide", "hides", "cover", "covers", "occlude", "occludes", "occluding", "covering"}
            or _is_display_action(action)
            or _action_has_relation_context(action, lowered)
            or any(action in str(item.get("action", "")) for item in relations)
        ):
            continue
        subject = entities[0] if entities else ""
        if not subject:
            continue
        relations.append(
            {
                "id": f"action:{_slug(subject)}:{_slug(action)}",
                "category": "action_relation",
                "question": _action_question(subject, action),
                "depends_on": [f"existence:{_slug(subject)}"],
                "subject": subject,
                "action": action,
            }
        )
    return relations


def _attribute_relation_binding_constraints(
    relations: Sequence[Mapping[str, Any]],
    colors: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Ask whether an action/relation applies to the same color-bound object."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relation in relations:
        if str(relation.get("category") or "") == "symbol_text_relation":
            continue
        subject = str(relation.get("subject") or "").strip()
        obj = str(relation.get("object") or "").strip()
        part = str(relation.get("part") or "").strip()
        action = str(
            relation.get("action") or relation.get("relation") or ""
        ).strip()
        relation_id = str(relation.get("id") or "").strip()
        if not subject or not action or not relation_id:
            continue
        if not _is_physical_binding_action(action):
            continue
        for target, color in colors.items():
            target = str(target).strip()
            color = str(color).strip()
            if not target or not color:
                continue
            part_mentions_target = part and target.lower() in part.lower()
            if obj != target and not part_mentions_target:
                continue
            key = f"{subject}:{target}:{action}:{color}"
            if key in seen:
                continue
            seen.add(key)
            part_phrase = part or target
            action_phrase = _progressive_action(action)
            if part_mentions_target:
                question = (
                    f"Is the {subject} visibly {action_phrase} the {color} {target}'s "
                    f"{part_phrase.split()[-1]}, rather than a different-colored "
                    f"{target} or nearby object?"
                )
            else:
                question = (
                    f"Is the {subject} visibly {action_phrase} the {color} {target}, "
                    f"rather than a different-colored {target} or nearby object?"
                )
            result.append(
                {
                    "id": (
                        f"binding:{_slug(subject)}:{_slug(target)}:"
                        f"{_slug(action)}:{_slug(color)}"
                    ),
                    "category": "attribute_relation_binding",
                    "question": question,
                    "depends_on": [
                        relation_id,
                        f"color:{_slug(target)}",
                    ],
                    "subject": subject,
                    "object": target,
                    "part": part,
                    "action": action,
                    "attribute": "color",
                    "value": color,
                }
            )
    return result


def _negative_relation_constraints(
    negative_constraints: Sequence[str],
    entities: Sequence[str],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for raw_constraint in negative_constraints:
        text = str(raw_constraint or "").strip().lower()
        if not text:
            continue
        match = re.search(
            r"\b(?P<subject>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,3})\s+"
            r"(?:is|are|be|being)?\s*not\s+"
            r"(?P<relation>attached\s+to|connected\s+to|mounted\s+on|stuck\s+to|touching)\s+"
            r"(?:an?\s+|the\s+)?(?P<object>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,3})\b",
            text,
        )
        if not match:
            continue
        subject = _best_entity_for_negative_relation(match.group("subject"), entities)
        obj = _best_entity_for_negative_relation(match.group("object"), entities)
        relation = _normalize_negative_relation(match.group("relation"))
        if not subject or not obj or not relation or subject == obj:
            continue
        relations.append(
            {
                "id": (
                    f"negative_relation:{_slug(subject)}:{_slug(obj)}:"
                    f"{_relation_label_slug(relation)}"
                ),
                "category": "negative_relation",
                "question": (
                    f"Is the {subject} clearly separate from the {obj}, with no visible "
                    f"{relation.replace('_', ' ')} between them?"
                ),
                "depends_on": [
                    f"existence:{_slug(subject)}",
                    f"existence:{_slug(obj)}",
                ],
                "subject": subject,
                "object": obj,
                "relation": relation,
                "negative": True,
            }
        )
    return relations


def _negative_symbol_text_constraints(
    negative_constraints: Sequence[str],
    entities: Sequence[str],
    intent_spec: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    requested_symbols = _requested_symbol_text_objects(intent_spec)
    for raw_constraint in negative_constraints:
        text = str(raw_constraint or "").strip().lower()
        if not text or not re.search(r"\b(no|without)\b", text):
            continue
        if not re.search(r"\b(symbol|text|logo|mark|letter|word|moon|star)\b", text):
            continue
        target = _best_symbol_absence_target(text, entities)
        if not target:
            continue
        forbidden = _best_forbidden_symbol_text(text, requested_symbols)
        relation_id = (
            f"negative_symbol:{_slug(target)}:"
            f"{_relation_slug(forbidden or 'symbol_text')}:absent"
        )
        question = (
            f"Is the {target} free of the forbidden "
            f"{forbidden or 'text, symbol, logo, or mark'}?"
        )
        relations.append(
            {
                "id": relation_id,
                "category": "negative_symbol_text_relation",
                "question": question,
                "depends_on": [f"existence:{_slug(target)}"],
                "subject": target,
                "object": forbidden or "symbol/text",
                "relation": "absent",
                "negative": True,
            }
        )
    return relations


def _negative_object_existence_constraints(
    negative_constraints: Sequence[str],
    entities: Sequence[str],
) -> list[dict[str, Any]]:
    """
    Generate absence checks for "no X" / "without X" object existence negations.

    Handles the most common negation type that was previously missing:
    - "no bowl and no spoon nearby"
    - "no window and no sign"
    - "without sitting inside it"
    - "with no visible zipper pull and no side pocket"

    This is distinct from:
    - _negative_relation_constraints: "X is not touching Y" (relation negation)
    - _negative_symbol_text_constraints: "no text/symbol on X" (text negation)
    """
    checks: list[dict[str, Any]] = []
    for raw_constraint in negative_constraints:
        text = str(raw_constraint or "").strip().lower()
        if not text:
            continue
        # Skip if already handled by symbol/text handler (has symbol/text keywords)
        if re.search(r"\b(symbol|text|logo|mark|letter|word)\b", text):
            continue
        # Skip if already handled by relation handler (has relation keywords)
        if re.search(
            r"\b(?:is|are|be|being)?\s*not\s+(?:attached|connected|mounted|stuck|touching)\b",
            text
        ):
            continue

        # Extract forbidden objects from patterns like:
        # "no bowl and no spoon" → ["bowl", "spoon"]
        # "without sitting inside" → skip (action, not object)
        # "with no visible zipper pull and no side pocket" → ["zipper pull", "side pocket"]
        forbidden_objects = []

        # Pattern 1: "no X" / "and no Y" / "with no Z"
        # Match all "no <object>" patterns, stop at conjunctions
        for match in re.finditer(
            r"\bno\s+(?:visible\s+)?(?P<obj>(?:[a-z0-9-]+)(?:\s+(?!and\b|or\b|but\b|with\b)[a-z0-9-]+){0,3})\b",
            text
        ):
            obj = match.group("obj").strip()
            # Filter out verbs/actions (sitting, standing, etc.)
            if obj and not re.match(r"^(sitting|standing|lying|running|walking|moving)$", obj):
                forbidden_objects.append(obj)

        # Pattern 2: "without X" (only if X is a noun, not verb phrase)
        for match in re.finditer(
            r"\bwithout\s+(?:a|an|the|any)?\s*(?P<obj>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})\b",
            text
        ):
            obj = match.group("obj").strip()
            # Only accept if looks like noun phrase, not verb phrase
            if obj and not re.match(r"^(sitting|standing|lying|running|walking|being|having)", obj):
                forbidden_objects.append(obj)

        # Deduplicate and generate one check per forbidden object
        for forbidden in dict.fromkeys(forbidden_objects):
            # Clean up trailing adverbs/prepositions
            forbidden = re.sub(r'\s+(nearby|inside|outside|around|above|below|next)$', '', forbidden)
            forbidden = forbidden.strip()
            if not forbidden:
                continue

            # Try to match against known entities (but don't require it)
            matched_entity = _best_entity_for_negative_relation(forbidden, entities)
            obj_name = matched_entity if matched_entity else forbidden

            check_id = f"negative_existence:{_slug(obj_name)}:absent"
            question = f"Is the {obj_name} absent from the image?"

            checks.append(
                {
                    "id": check_id,
                    "category": "negative_object_existence",
                    "question": question,
                    "expected": "yes",  # yes, it's absent
                    "object": obj_name,
                    "negative": True,
                    "type": "forbidden_object",
                }
            )
    return checks


def _requested_symbol_text_objects(
    intent_spec: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(intent_spec, Mapping):
        return []
    result: list[str] = []
    for item in _mapping_records(intent_spec.get("interaction_relations")):
        action = str(item.get("action") or "").strip().lower()
        obj = _entity_from_spec(item.get("object"))
        if _is_display_action(action) and obj and _looks_like_symbol_text_target(obj):
            result.append(obj)
    return list(dict.fromkeys(result))


def _best_symbol_absence_target(text: str, entities: Sequence[str]) -> str:
    no_pos = text.find(" no ")
    if no_pos < 0:
        no_pos = text.find(" without ")
    prefix = text[:no_pos] if no_pos >= 0 else text
    nearest_full = _nearest_full_entity_in_text(prefix, entities)
    if nearest_full:
        return nearest_full
    plain_match = re.search(
        r"\b(?:plain\s+)?(?P<object>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,3})\s+"
        r"(?:with|has|have|showing|displaying)?\s*no\s+"
        r"(?:text|symbol|logo|mark|letter|word)s?\b",
        text,
    )
    if plain_match:
        matched = _best_entity_for_negative_relation(plain_match.group("object"), entities)
        if matched:
            return matched
    best = ""
    best_pos = -1
    for entity in entities:
        for term in [entity, *_entity_tail_terms(entity)]:
            for match in re.finditer(rf"\b{re.escape(term)}\b", prefix):
                if match.start() > best_pos:
                    best = entity
                    best_pos = match.start()
    return best


def _nearest_full_entity_in_text(text: str, entities: Sequence[str]) -> str:
    best = ""
    best_pos = -1
    for entity in entities:
        entity = str(entity or "").strip()
        if not entity:
            continue
        for match in re.finditer(rf"\b{re.escape(entity)}\b", text):
            if match.start() > best_pos:
                best = entity
                best_pos = match.start()
    return best


def _best_forbidden_symbol_text(text: str, requested_symbols: Sequence[str]) -> str:
    for symbol in requested_symbols:
        if _looks_like_symbol_text_target(symbol):
            return symbol
    match = re.search(
        r"\b(?:no|without)\s+(?P<object>[a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})\b",
        text,
    )
    if match:
        candidate = _canonical_entity_fragment(_clean_entity_name(match.group("object")))
        if candidate and _looks_like_symbol_text_target(candidate):
            return candidate
    if "symbol" in text:
        return "symbol"
    if "text" in text:
        return "text"
    return ""


def _looks_like_symbol_text_target(value: str) -> bool:
    return bool(
        re.search(
            r"\b(text|symbol|logo|mark|letter|word|sign|star|moon|number)\b",
            str(value or "").lower(),
        )
    )


def _entity_tail_terms(entity: str) -> list[str]:
    words = [word for word in str(entity or "").split() if word]
    return words[-1:] if len(words) > 1 else []


def _best_entity_for_negative_relation(value: str, entities: Sequence[str]) -> str:
    cleaned = _canonical_entity_fragment(_clean_entity_name(value))
    if not cleaned:
        return ""
    best = _best_entity_match(cleaned, entities)
    return best or cleaned


def _normalize_negative_relation(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return {
        "attached to": "attached_to",
        "connected to": "connected_to",
        "mounted on": "mounted_on",
        "stuck to": "stuck_to",
        "touching": "touching",
    }.get(text, "")


def _relation_label_slug(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def _structured_relation_constraints(
    intent_spec: Mapping[str, Any] | None,
    counts: Mapping[str, int],
    parts: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    if not isinstance(intent_spec, Mapping):
        return []
    relations: list[dict[str, Any]] = []
    for item in _mapping_records(intent_spec.get("interaction_relations")):
        subject = _entity_from_spec(item.get("subject"))
        obj = _entity_from_spec(item.get("object"))
        action = str(item.get("action") or "").strip().lower()
        if not subject or not obj or not action or subject == obj:
            continue
        relation_type = str(item.get("type") or "").strip().lower()
        if relation_type == "occlusion":
            hidden_part = str(item.get("hidden_part") or "requested part").strip()
            visible_part = str(item.get("visible_part") or "").strip()
            relation_id = (
                f"relation:{_slug(subject)}:{_relation_slug(obj)}:"
                f"{_slug(action)}_{_slug(hidden_part)}"
            )
            question = (
                f"Does the {subject} visibly {action} the {hidden_part} of the {obj}, "
                f"so that this part is occluded rather than fully visible?"
            )
            if visible_part:
                question += f" Is the {visible_part} still clearly visible?"
            relations.append(
                {
                    "id": relation_id,
                    "category": "occlusion_relation",
                    "question": question,
                    "depends_on": [
                        f"existence:{_slug(subject)}",
                        f"existence:{_slug(obj)}",
                    ],
                    "subject": subject,
                    "object": obj,
                    "action": action,
                    "hidden_part": hidden_part,
                    "visible_part": visible_part,
                    "typed_relation": "occlusion",
                }
            )
            continue
        relation_id = f"relation:{_slug(subject)}:{_relation_slug(obj)}:{_slug(action)}"
        relations.append(
            {
                "id": relation_id,
                "category": (
                    "symbol_text_relation"
                    if _is_display_action(action)
                    else "action_relation"
                ),
                "question": (
                    _display_relation_question(subject, action, obj)
                    if _is_display_action(action)
                    else _action_object_question(subject, action, obj)
                ),
                "depends_on": [
                    f"existence:{_slug(subject)}",
                    _dependency_for_relation_object(obj, parts),
                ],
                "subject": subject,
                "object": obj,
                "action": action,
                "typed_relation": "display" if _is_display_action(action) else "interaction",
            }
        )
    for item in _mapping_records(intent_spec.get("relations")):
        subject = _entity_from_spec(item.get("subject"))
        obj = _entity_from_spec(item.get("object"))
        phrase = str(item.get("phrase") or "").strip().lower()
        label = _relation_label(phrase)
        if not subject or not obj or not label or subject == obj:
            continue
        relations.append(
            {
                "id": f"relation:{_slug(subject)}:{_relation_slug(obj)}:{_slug(label)}",
                "category": "spatial_relation",
                "question": _relation_question(subject, obj, label, counts),
                "depends_on": [
                    f"existence:{_slug(subject)}",
                    _dependency_for_relation_object(obj, parts),
                ],
                "subject": subject,
                "object": obj,
                "relation": label,
                "typed_relation": "spatial",
            }
        )
    return relations


def _mapping_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _overlaps_existing_relation(
    relation: Mapping[str, Any],
    existing: Sequence[Mapping[str, Any]],
) -> bool:
    subject = str(relation.get("subject") or "").strip()
    obj = str(relation.get("object") or "").strip()
    action = str(relation.get("action") or relation.get("relation") or "").strip()
    if not subject or not obj or not action:
        return False
    for item in existing:
        item_subject = str(item.get("subject") or "").strip()
        item_obj = str(item.get("object") or "").strip()
        item_action = str(item.get("action") or item.get("relation") or "").strip()
        if subject != item_subject or action != item_action:
            continue
        if obj == item_obj or obj in item_obj or item_obj in obj:
            return True
    return False


def _dependency_for_relation_object(
    obj: str,
    parts: Sequence[Mapping[str, Any]],
) -> str:
    obj_slug = _slug(obj)
    for part in parts:
        part_id = str(part.get("id") or "")
        part_name = str(part.get("name") or "")
        if part_id == f"part:{obj_slug}" or _slug(part_name) == obj_slug:
            return part_id
    return f"existence:{obj_slug}"


def _entity_from_spec(value: Any) -> str:
    return _normalize_relation_entity_name(str(value or ""))


def _relation_label(phrase: str) -> str:
    phrase = str(phrase or "").strip().lower()
    return {
        "on the left of": "left_of",
        "to the left of": "left_of",
        "left of": "left_of",
        "on the right of": "right_of",
        "to the right of": "right_of",
        "right of": "right_of",
        "in front of": "in_front_of",
        "next to": "next_to",
        "beside": "beside",
        "near": "near",
        "behind": "behind",
        "under": "under",
        "above": "above",
        "on top of": "on",
        "on": "on",
    }.get(phrase, "")


def _explicit_action_object_relations(
    prompt: str,
    entities: Sequence[str],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    positions = [(entity, _find_entity_position(prompt, entity)) for entity in entities]
    positions = [(entity, pos) for entity, pos in positions if pos < 10**8]
    positions.sort(key=lambda item: item[1])
    for action in (
        "holding",
        "holds",
        "gripping",
        "grasping",
        "wearing",
        "carrying",
        "riding",
        "touching",
    ):
        action_match = re.search(rf"\b{re.escape(action)}\b", prompt)
        if not action_match:
            continue
        subject = _subject_for_action(prompt, positions, action_match.start())
        obj = _nearest_entity_after(positions, action_match.end())
        if not subject or not obj or subject == obj:
            continue
        relation_id = f"relation:{_slug(subject)}:{_relation_slug(obj)}:{_slug(action)}"
        relations.append(
            {
                "id": relation_id,
                "category": "action_relation",
                "question": _action_object_question(subject, action, obj),
                "depends_on": [
                    f"existence:{_slug(subject)}",
                    f"existence:{_slug(obj)}",
                ],
                "subject": subject,
                "object": obj,
                "action": action,
            }
        )
    return relations


def _explicit_spatial_relations_from_prompt(
    prompt: str,
    entities: Sequence[str],
    counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    positions = [(entity, _find_entity_position(prompt, entity)) for entity in entities]
    positions = [(entity, pos) for entity, pos in positions if pos < 10**8]
    positions.sort(key=lambda item: item[1])
    terms = [
        ("on the left of", "left_of"),
        ("to the left of", "left_of"),
        ("left of", "left_of"),
        ("on the right of", "right_of"),
        ("to the right of", "right_of"),
        ("right of", "right_of"),
        ("in front of", "in_front_of"),
        ("next to", "next_to"),
        ("beside", "beside"),
        ("near", "near"),
    ]
    for needle, label in terms:
        for match in re.finditer(rf"\b{re.escape(needle)}\b", prompt):
            subject = _nearest_entity_before(positions, match.start())
            obj = _nearest_entity_after(positions, match.end())
            if not subject or not obj or subject == obj:
                continue
            if _looks_like_worn_or_held_accessory(subject, prompt) and positions:
                subject = positions[0][0]
            relations.append(
                {
                    "id": f"relation:{_slug(subject)}:{_relation_slug(obj)}:{_slug(label)}",
                    "category": "spatial_relation",
                    "question": _relation_question(subject, obj, label, counts),
                    "depends_on": [
                        f"existence:{_slug(subject)}",
                        f"existence:{_slug(obj)}",
                    ],
                    "subject": subject,
                    "object": obj,
                    "relation": label,
                }
            )
    return relations


def _binary_relations_from_prompt(
    prompt: str,
    entities: Sequence[str],
    counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    positions = [(entity, _find_entity_position(prompt, entity)) for entity in entities]
    positions = [(entity, pos) for entity, pos in positions if pos < 10**8]
    positions.sort(key=lambda item: item[1])
    relation_terms = [
        ("on the left of", "left_of", "spatial_relation"),
        ("to the left of", "left_of", "spatial_relation"),
        ("left of", "left_of", "spatial_relation"),
        ("on the right of", "right_of", "spatial_relation"),
        ("to the right of", "right_of", "spatial_relation"),
        ("right of", "right_of", "spatial_relation"),
        ("in front of", "in_front_of", "spatial_relation"),
        ("next to", "next_to", "spatial_relation"),
        ("beside", "beside", "spatial_relation"),
        ("on", "on"),
        ("near", "near"),
        ("behind", "behind"),
        ("under", "under"),
        ("above", "above"),
    ]
    for index in range(len(positions) - 1):
        left, left_pos = positions[index]
        right, right_pos = positions[index + 1]
        between = prompt[left_pos + len(left) : right_pos]
        if _has_intervening_entity(between, entities):
            continue
        for item in relation_terms:
            needle = item[0]
            label = item[1]
            category = item[2] if len(item) > 2 else ("spatial_relation" if label != "on" else "action_relation")
            if re.search(rf"\b{re.escape(needle)}\b", between):
                if category == "spatial_relation" and _looks_like_worn_or_held_accessory(left, prompt):
                    break
                relations.append(
                    {
                        "id": f"relation:{_slug(left)}:{_relation_slug(right)}:{_slug(label)}",
                        "category": category,
                        "question": _relation_question(left, right, label, counts),
                        "depends_on": [
                            f"existence:{_slug(left)}",
                            f"existence:{_slug(right)}",
                        ],
                        "subject": left,
                        "object": right,
                        "relation": label,
                    }
                )
                break
    return relations


def _looks_like_worn_or_held_accessory(entity: str, prompt: str) -> bool:
    pos = _find_entity_position(prompt, entity)
    if pos >= 10**8:
        return False
    window = prompt[max(0, pos - 80) : pos]
    return bool(
        re.search(
            r"\b(wearing|holding|holds?|gripping|grasping|carrying)\b",
            window,
        )
    )


def _nearest_entity_before(positions: Sequence[tuple[str, int]], offset: int) -> str:
    before = [(entity, pos) for entity, pos in positions if pos < offset]
    return before[-1][0] if before else ""


def _subject_for_action(
    prompt: str,
    positions: Sequence[tuple[str, int]],
    action_offset: int,
) -> str:
    if re.search(r"\bwhile\s+$", prompt[max(0, action_offset - 16) : action_offset]):
        return positions[0][0] if positions else ""
    boundary = max(
        prompt.rfind(",", 0, action_offset),
        prompt.rfind(";", 0, action_offset),
        prompt.rfind(" while ", 0, action_offset),
    )
    local_before = prompt[boundary + 1 : action_offset] if boundary >= 0 else prompt[:action_offset]
    if boundary >= 0 and not any(
        re.search(rf"\b{re.escape(entity)}\b", local_before) for entity, _ in positions
    ):
        return positions[0][0] if positions else ""
    subject = _nearest_entity_before(positions, action_offset)
    if subject and _looks_like_worn_or_held_accessory(subject, prompt):
        return positions[0][0] if positions else subject
    return subject


def _nearest_entity_after(positions: Sequence[tuple[str, int]], offset: int) -> str:
    after = [(entity, pos) for entity, pos in positions if pos >= offset]
    return after[0][0] if after else ""


def _relation_question(
    subject: str,
    obj: str,
    relation: str,
    counts: Mapping[str, int],
) -> str:
    subject_text = subject
    if counts.get(subject, 0) > 1:
        subject_text = f"all {counts[subject]} required {subject}"
    if relation == "left_of":
        return f"Is the {subject_text} visibly to the left of the {obj}?"
    if relation == "right_of":
        return f"Is the {subject_text} visibly to the right of the {obj}?"
    if relation == "in_front_of":
        return f"Is the {subject_text} visibly in front of the {obj}?"
    if relation == "next_to":
        return f"Is the {subject_text} visibly next to the {obj}?"
    if relation == "on":
        return f"Is the {subject_text} visibly on the {obj}?"
    if relation in {"holds", "holding"}:
        return f"Is the {subject_text} visibly holding the {obj}?"
    if relation in {"gripping", "grasping"}:
        return f"Is the {subject_text} visibly {relation} the {obj}?"
    return f"Is the {subject_text} visibly {relation.replace('_', ' ')} the {obj}?"


def _action_object_question(subject: str, action: str, obj: str) -> str:
    action_phrase = _progressive_action(action)
    return f"Is the {subject} visibly {action_phrase} the {obj}?"


def _display_relation_question(subject: str, action: str, obj: str) -> str:
    del action
    return (
        f"Does the {subject} visibly show or display the {obj} on the requested "
        "surface, mark, cover, sign, or label?"
    )


def _progressive_action(action: str) -> str:
    action = str(action or "").strip().lower()
    if action in {"hold", "holds", "holding"}:
        return "holding"
    if action in {"grip", "grips", "gripping"}:
        return "gripping"
    if action in {"grasp", "grasps", "grasping"}:
        return "grasping"
    if action in {"carry", "carries", "carrying"}:
        return "carrying"
    if action in {"touch", "touches", "touching"}:
        return "touching"
    return action


def _action_question(subject: str, action: str) -> str:
    if action == "standing":
        return (
            f"Is the {subject} visibly standing upright on their feet, with no seated, "
            "crouching, kneeling, or leaning posture?"
        )
    if action == "sitting":
        return (
            f"Is the {subject} visibly sitting with body support/contact, not merely "
            "standing, floating, or placed nearby?"
        )
    if "hold" in action or "carrying" in action:
        return (
            f"Is the {subject} visibly {action} the requested object with clear "
            "hand, claw, or contact-point grasp evidence, not merely attached, "
            "mounted, supported, or nearby?"
        )
    if "grip" in action or "grasp" in action:
        return (
            f"Is the {subject} visibly {action} the requested object or part with "
            "a hand, claw, or contact point wrapped around or firmly grasping it, "
            "not merely touching, attached, mounted, supported, or nearby?"
        )
    return f"Is the {subject} visibly {action} as requested?"


def _is_display_action(action: str) -> bool:
    return str(action or "").strip().lower() in DISPLAY_ACTIONS


def _is_physical_binding_action(action: str) -> bool:
    action = str(action or "").strip().lower()
    return any(
        token in action
        for token in (
            "carry",
            "grasp",
            "grip",
            "hold",
            "riding",
            "ride",
            "touch",
            "wear",
        )
    )


def _action_has_relation_context(action: str, prompt: str) -> bool:
    action = str(action or "").strip().lower()
    if not action:
        return False
    if action not in {"sit", "sits", "sitting", "perch", "perches", "perching"}:
        return False
    relation_after_action = (
        "above",
        "behind",
        "beside",
        "in front of",
        "left of",
        "near",
        "next to",
        "on",
        "right of",
        "under",
    )
    action_pattern = re.escape(action).replace(r"\ ", r"\s+")
    relation_alt = "|".join(re.escape(item) for item in relation_after_action)
    return bool(re.search(rf"\b{action_pattern}\b\s+(?:{relation_alt})\b", prompt))


def _scene_style_phrases(
    protected_phrases: Sequence[str],
    colors: Mapping[str, str],
) -> list[str]:
    color_phrases = {f"{color} {entity}" for entity, color in colors.items()}
    result: list[str] = []
    for phrase in protected_phrases:
        cleaned = str(phrase).strip().lower()
        if cleaned and cleaned not in color_phrases and cleaned not in result:
            result.append(cleaned)
    return result


def _append_question(
    questions: list[ConstraintQuestion],
    seen: set[str],
    question: ConstraintQuestion,
) -> None:
    if question.id in seen:
        return
    seen.add(question.id)
    questions.append(question)


def _color_question(entity: str) -> str:
    return (
        f"What is the dominant visible color of the main visible body or surface "
        f"of the {entity}? Ignore small accessories, highlights, shadows, labels, "
        "eyes, and background regions."
    )


def _has_intervening_entity(between: str, entities: Sequence[str]) -> bool:
    text = between.lower()
    return any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for entity in entities
        for term in [entity, *entity.split()]
        if term and len(term) > 2
    )


def _part_record(parent: str, part_type: str) -> dict[str, Any]:
    return {
        "id": f"part:{_slug(parent)}_{_slug(part_type)}",
        "name": f"{parent} {part_type}",
        "parent": parent,
        "part_type": part_type,
    }


def _dedupe_parts(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for part in parts:
        part_id = str(part.get("id") or "")
        if part_id and part_id not in seen:
            seen.add(part_id)
            result.append(deepcopy(dict(part)))
    return result


def _clean_entity_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9 -]+", " ", value.lower())
    words = [word for word in value.split() if word]
    stop_after = {
        "avoid",
        "sitting",
        "standing",
        "gripping",
        "holding",
        "touches",
        "touching",
        "touch",
        "riding",
        "carrying",
        "balancing",
        "wearing",
        "top",
        "photo",
        "cinematic",
        "rainy",
        "street",
        "no",
        "not",
        "while",
        "without",
        *RELATION_CONNECTOR_WORDS,
    }
    drop = ENTITY_DROP_WORDS
    kept: list[str] = []
    for word in words:
        if word.isdigit():
            continue
        if word in stop_after and kept:
            break
        if word in stop_after or word in drop:
            continue
        kept.append(word)
    cleaned = " ".join(kept[:3]).strip()
    cleaned = _canonical_entity_fragment(cleaned)
    return "" if cleaned in NON_ENTITY_TERMS else cleaned


def _canonical_entity_fragment(value: str) -> str:
    words = [word for word in str(value or "").lower().split() if word]
    if not words:
        return ""
    while words and words[-1] in ENTITY_TRAILING_STOP_WORDS:
        words.pop()
    while words and words[0] in {"and", "or"}:
        words.pop(0)
    cleaned = " ".join(words).strip()
    if cleaned in NON_ENTITY_TERMS or cleaned in {"objects", "object", "exact text"}:
        return ""
    return cleaned


def _normalize_entity_name_keep_color(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9 -]+", " ", str(value or "").lower())
    words = [word for word in value.split() if word]
    stop_after = {
        "avoid",
        "sitting",
        "standing",
        "gripping",
        "holding",
        "touches",
        "touching",
        "touch",
        "riding",
        "carrying",
        "balancing",
        "wearing",
        "top",
        "photo",
        "cinematic",
        "rainy",
        "street",
        "no",
        "not",
        "while",
        "without",
        *RELATION_CONNECTOR_WORDS,
    }
    kept: list[str] = []
    for word in words:
        if word.isdigit():
            continue
        if word in stop_after and kept:
            break
        if word in stop_after:
            continue
        if word in ENTITY_DROP_WORDS and word not in COLOR_WORDS:
            continue
        kept.append(word)
    cleaned = " ".join(kept[:4]).strip()
    cleaned = _canonical_entity_fragment(cleaned)
    return "" if cleaned in NON_ENTITY_TERMS else cleaned


def _normalize_relation_entity_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9 -]+", " ", str(value or "").lower())
    words = [word for word in value.split() if word]
    stop_after = {
        "avoid",
        "sitting",
        "standing",
        "gripping",
        "holding",
        "touches",
        "touching",
        "touch",
        "riding",
        "carrying",
        "balancing",
        "wearing",
        "photo",
        "cinematic",
        "rainy",
        "street",
        "no",
        "not",
        "while",
        "without",
        *RELATION_CONNECTOR_WORDS,
    }
    kept: list[str] = []
    for word in words:
        if word.isdigit():
            continue
        if word in stop_after and kept:
            break
        if word in stop_after:
            continue
        if word in ENTITY_DROP_WORDS and word not in COLOR_WORDS and word not in PART_NAME_WORDS:
            continue
        kept.append(word)
    cleaned = " ".join(kept[:4]).strip()
    cleaned = _canonical_entity_fragment(cleaned)
    return "" if cleaned in NON_ENTITY_TERMS else cleaned


def _covered_by_color_entity(entity: str, color_entities: set[str]) -> bool:
    if not entity or entity in color_entities:
        return False
    for color_entity in color_entities:
        parts = color_entity.split()
        if len(parts) > 1 and parts[0] in COLOR_WORDS and " ".join(parts[1:]) == entity:
            return True
    return False


def _best_entity_match(candidate: str, entities: Sequence[str]) -> str:
    if candidate in entities:
        return candidate
    candidate_terms = set(candidate.split())
    for entity in entities:
        terms = set(entity.split())
        if candidate_terms & terms:
            return entity
    return candidate


def _find_entity_position(prompt: str, entity: str) -> int:
    terms = [entity, *entity.split()]
    positions = [prompt.find(term) for term in terms if term and prompt.find(term) >= 0]
    return min(positions) if positions else 10**9


def _is_style_or_attribute(entity: str) -> bool:
    return entity in {
        "photo",
        "street",
        "rainy",
        "cinematic",
        "film",
        "grain",
        *NON_ENTITY_TERMS,
    }


def _is_minor_part_name(entity: str) -> bool:
    return entity in {"handle", "hand", "hands", "claw", "claws", "grip"}


def _looks_plural_entity(entity: str) -> bool:
    head = entity.split()[-1] if entity.split() else entity
    if head.endswith("ss"):
        return False
    if head.endswith("ies") and len(head) > 3:
        return True
    return head.endswith("s") and len(head) > 3


def _slug(value: str) -> str:
    raw_value = _normalize_entity_name_keep_color(value)
    if raw_value.split() and raw_value.split()[0] in COLOR_WORDS:
        value = raw_value
    else:
        value = _clean_entity_name(value) or str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "unknown"


def _relation_slug(value: str) -> str:
    value = _normalize_relation_entity_name(value) or _clean_entity_name(value) or str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "unknown"


def _extract_json(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _answer_items(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        answers = data.get("answers", data.get("question_answers", data.get("results")))
        if isinstance(answers, Mapping):
            return [answers]
        if isinstance(answers, list):
            return [item for item in answers if isinstance(item, Mapping)]
        if "answer" in data:
            return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, Mapping)]
    return []


def _matching_answer(
    raw_answers: Sequence[Mapping[str, Any]],
    question: ConstraintQuestion,
    index: int,
) -> Mapping[str, Any]:
    has_explicit_ids = any(
        str(item.get("id", item.get("question_id", ""))).strip()
        for item in raw_answers
    )
    for item in raw_answers:
        if str(item.get("id", item.get("question_id", ""))) == question.id:
            return item
    if has_explicit_ids:
        return {
            "id": question.id,
            "answer": "uncertain",
            "evidence": f"No matching answer returned for {question.id}.",
        }
    if index < len(raw_answers):
        return raw_answers[index]
    return {"id": question.id, "answer": "uncertain", "evidence": "No matching answer."}


def _single_text_answer(response: str, question: ConstraintQuestion) -> dict[str, Any]:
    text = str(response or "").strip()
    if question.answer_type == "number":
        answer = str(_number_from_text(text)) if _number_from_text(text) is not None else "uncertain"
    elif question.answer_type == "short_text":
        answer = _normalize_answer_value(text, question)
    elif _looks_yes(text):
        answer = "yes"
    elif _looks_no(text):
        answer = "no"
    else:
        answer = "uncertain"
    return {"id": question.id, "answer": answer, "evidence": text}


def _looks_yes(text: str) -> bool:
    return bool(re.search(r"\b(yes|true|pass|passes|present|visible|correct)\b", text))


def _looks_no(text: str) -> bool:
    return bool(re.search(r"\b(no|false|fail|fails|missing|absent|not visible|incorrect)\b", text))


def _looks_uncertain(text: str) -> bool:
    return bool(re.search(r"\b(uncertain|unclear|ambiguous|maybe|not sure|cannot tell)\b", text))


def _number_from_text(text: str) -> int | None:
    text = text.strip().lower()
    digit = re.search(r"\b\d+\b", text)
    if digit:
        return int(digit.group(0))
    for word, number in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return number
    return None


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _score_from_summary(
    *,
    passed: int,
    failed: int,
    blocked: int,
    hard_failures: int,
) -> float:
    total = max(1, passed + failed + blocked)
    base = passed / total
    penalty = min(0.35, 0.12 * hard_failures)
    return max(0.0, min(1.0, base - penalty))


def _check_type(question: ConstraintQuestion) -> str:
    return {
        "entity_existence": "subject",
        "count": "wrong_count",
        "color_binding": "color",
        "part_visibility": "part",
        "action_relation": "relation",
        "occlusion_relation": "relation",
        "spatial_relation": "relation",
        "symbol_text_relation": "relation",
        "negative_symbol_text_relation": "relation",
        "negative_object_existence": "relation",
        "negative_relation": "relation",
        "attribute_relation_binding": "relation_binding",
        "scene_style": "style",
    }.get(question.category, question.category)


def _error_type(question: ConstraintQuestion) -> str:
    return {
        "entity_existence": "missing_object",
        "count": "wrong_count",
        "color_binding": "wrong_attribute",
        "part_visibility": "missing_object",
        "action_relation": "wrong_relation",
        "occlusion_relation": "wrong_relation",
        "spatial_relation": "wrong_relation",
        "symbol_text_relation": "wrong_relation",
        "negative_symbol_text_relation": "wrong_relation",
        "negative_object_existence": "forbidden_object_present",
        "negative_relation": "wrong_relation",
        "attribute_relation_binding": "wrong_relation",
        "scene_style": "style_mismatch",
    }.get(question.category, "wrong_attribute")


def _target_from_question(question: ConstraintQuestion) -> str:
    source = question.source_constraint
    return str(
        source.get("part")
        or source.get("object")
        or source.get("phrase")
        or source.get("target")
        or question.id.split(":", 1)[-1].replace("_", " ")
    )


def _revision_hint(errors: Sequence[Mapping[str, Any]]) -> str:
    if not errors:
        return "All hard user-grounded VQA constraints passed."
    spans = [str(item.get("prompt_span") or "") for item in errors if item.get("prompt_span")]
    if spans:
        return "Fix user-grounded visual constraints: " + "; ".join(spans[:4]) + "."
    return "Fix failed user-grounded visual constraints."


def _strengths_from_checks(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"Passed {item.get('category')}: {item.get('target')}"
        for item in checks
        if item.get("passed") is True
    ][:6]


def _hard_failure_count(check: Mapping[str, Any]) -> int:
    count = 0
    for item in _list_records(check.get("checks", [])):
        if item.get("passed") is False and str(item.get("type", "")).lower() != "style":
            count += 1
    count += len(_list_records(check.get("errors", [])))
    return count


def _failed_constraint_ids(check: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in _list_records(check.get("checks", [])):
        if item.get("passed") is False:
            ids.append(str(item.get("question_id") or item.get("target") or item.get("type") or ""))
    return _dedupe_text(ids)


def _passed_constraint_ids(check: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in _list_records(check.get("checks", [])):
        if item.get("passed") is True:
            ids.append(str(item.get("question_id") or item.get("target") or item.get("type") or ""))
    return _dedupe_text(ids)


def _legacy_error_type(check: Mapping[str, Any]) -> str:
    check_type = str(check.get("type") or "").lower()
    target_text = " ".join(
        str(check.get(key) or "").lower()
        for key in ("target", "expected", "observed", "description")
    )
    if "handle" in target_text and any(color in target_text for color in COLOR_WORDS):
        return "wrong_attribute"
    if any(token in target_text for token in ("handle", "hand", "claw", "grip")):
        return "wrong_relation"
    if "color" in check_type or "attribute" in check_type:
        return "wrong_attribute"
    if "relation" in check_type or "action" in check_type or "grip" in check_type:
        return "wrong_relation"
    if "count" in check_type:
        return "wrong_count"
    if "subject" in check_type or "object" in check_type:
        return "missing_object"
    return "wrong_attribute"


def _list_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, list):
        return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _list_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _dedupe_errors(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in errors:
        error = deepcopy(dict(item))
        key = (
            str(error.get("type", "")),
            str(error.get("prompt_span", "")),
            str(error.get("evidence", "")),
        )
        if key not in seen:
            seen.add(key)
            result.append(error)
    return result


def _dedupe_text(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _normalize_score(value: Any) -> float:
    score = _coerce_float(value, default=0.5)
    if score is None:
        return 0.5
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else 1.0
    return max(0.0, min(float(score), 1.0))


def _coerce_float(value: Any, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_sentence(text: str) -> str:
    text = str(text or "").strip()
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    return match.group(1).strip() if match else text[:160]


def _ensure_constraints(
    value: PromptConstraints | Mapping[str, Any] | str,
) -> PromptConstraints:
    if isinstance(value, PromptConstraints):
        return value
    if isinstance(value, str):
        return extract_constraints(value)
    if isinstance(value, Mapping):
        return PromptConstraints(
            original_prompt=str(value.get("original_prompt", "")),
            colors=dict(value.get("colors", {})),
            subjects=[str(item) for item in value.get("subjects", [])],
            actions=[str(item) for item in value.get("actions", [])],
            relations=[str(item) for item in value.get("relations", [])],
            protected_phrases=[str(item) for item in value.get("protected_phrases", [])],
        )
    raise TypeError("constraints must be PromptConstraints, mapping, or str")
