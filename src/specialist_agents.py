"""Specialist-agent observation and local constraint arbitration.

The first version intentionally uses one VLM call for a structured visual
observation, then performs subject/attribute/spatial/interaction analysis
locally. This keeps API cost bounded while making failures auditable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence

from .clients import VLMClient
from .prompt_constraints import PromptConstraints, extract_constraints


SPECIALIST_NAMES = (
    "SubjectExistenceAgent",
    "AttributeBindingAgent",
    "SpatialLayoutAgent",
    "InteractionRelationAgent",
    "SymbolTextVisibilityAgent",
    "StyleBackgroundAgent",
)

DISPLAY_ACTIONS = {"show", "shows", "showing", "display", "displays", "displaying"}


@dataclass(frozen=True)
class ConstraintFailure:
    """One typed failure found by a specialist agent."""

    type: str
    target: str
    evidence: str
    confidence: float | None = None
    attribute: str = ""
    expected: str = ""
    observed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "attribute": self.attribute,
            "expected": self.expected,
            "observed": self.observed,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SpecialistReport:
    """Report produced by one local specialist agent."""

    agent: str
    passed: bool
    failures: list[ConstraintFailure] = field(default_factory=list)
    uncertain: list[ConstraintFailure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "passed": self.passed,
            "failures": [item.to_dict() for item in self.failures],
            "uncertain": [item.to_dict() for item in self.uncertain],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SpecialistArbitration:
    """Merged decision from all specialist reports."""

    global_passed: bool
    dominant_failure: str
    selected_action: str
    fallback_action: str
    prompt_patch: str
    forbidden_phrases: list[str] = field(default_factory=list)
    protected_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_passed": self.global_passed,
            "dominant_failure": self.dominant_failure,
            "selected_action": self.selected_action,
            "fallback_action": self.fallback_action,
            "prompt_patch": self.prompt_patch,
            "forbidden_phrases": list(self.forbidden_phrases),
            "protected_constraints": deepcopy(self.protected_constraints),
        }


def run_specialist_observation(
    *,
    vlm: VLMClient,
    user_prompt: str,
    image_path: str,
    generated_prompt: str = "",
    constraints: PromptConstraints | Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Run one VLM observation, then split analysis into local specialists."""

    constraints = _ensure_constraints(constraints or user_prompt)
    request = build_specialist_observation_request(
        user_prompt=user_prompt,
        generated_prompt=generated_prompt,
        constraints=constraints,
    )
    raw_response = vlm.vision(request, [image_path])
    return analyze_specialist_observation(
        user_prompt=user_prompt,
        image_path=image_path,
        generated_prompt=generated_prompt,
        raw_response=raw_response,
        request=request,
        constraints=constraints,
        api_call_count=1,
    )


def analyze_specialist_observation(
    *,
    user_prompt: str,
    image_path: str,
    generated_prompt: str = "",
    raw_response: str = "",
    observation: Mapping[str, Any] | None = None,
    request: str = "",
    constraints: PromptConstraints | Mapping[str, Any] | str | None = None,
    api_call_count: int = 0,
) -> dict[str, Any]:
    """Analyze an existing VLM observation with local specialist agents."""

    constraints = _ensure_constraints(constraints or user_prompt)
    if observation is None:
        observation = parse_specialist_observation_response(raw_response)
    else:
        observation = _normalize_observation(observation)
    reports = build_specialist_reports(
        observation,
        constraints,
        generated_prompt=generated_prompt,
    )
    arbitration = arbitrate_specialist_reports(reports, constraints)
    return {
        "image_path": image_path,
        "user_prompt": user_prompt,
        "generated_prompt": generated_prompt,
        "constraints": constraints.to_dict(),
        "request": request,
        "raw_response": raw_response,
        "observation": observation,
        "reports": [report.to_dict() for report in reports],
        "arbitration": arbitration.to_dict(),
        "api_call_count": int(api_call_count),
    }


def build_specialist_observation_request(
    *,
    user_prompt: str,
    generated_prompt: str,
    constraints: PromptConstraints,
) -> str:
    intent = constraints.intent_spec.to_dict() if constraints.intent_spec else None
    schema = {
        "subjects": [
            {
                "name": "object name",
                "visible": True,
                "count": 1,
                "confidence": 0.0,
                "evidence": "short visual evidence",
            }
        ],
        "attributes": [
            {
                "object": "object name",
                "attribute": "color",
                "expected": "expected value if known",
                "observed": "observed value",
                "passed": True,
                "confidence": 0.0,
                "evidence": "short visual evidence",
            }
        ],
        "spatial_relations": [
            {
                "subject": "object",
                "relation": "left_of",
                "object": "object",
                "passed": True,
                "confidence": 0.0,
                "evidence": "short visual evidence",
            }
        ],
        "interaction_relations": [
            {
                "subject": "object",
                "action": "holds",
                "object": "target object or part",
                "passed": True,
                "confidence": 0.0,
                "evidence": "short visual evidence",
                "confused_with": "wrong object if any",
            }
        ],
        "symbol_text_relations": [
            {
                "subject": "object",
                "action": "shows",
                "object": "text or symbol",
                "passed": True,
                "confidence": 0.0,
                "evidence": "short visual evidence",
            }
        ],
        "negative_constraints": [
            {
                "constraint": "no leakage",
                "passed": True,
                "confidence": 0.0,
                "evidence": "short visual evidence",
            }
        ],
        "summary": {
            "global_passed": False,
            "dominant_failure": "subject|attribute|spatial|interaction|style|none",
            "repair_hint": "one concise repair instruction",
        },
    }
    return (
        "You are a structured visual observation module for a text-to-image agent.\n"
        "Use exactly one image. Answer from visible evidence only.\n"
        "Do not infer success from the prompt. If uncertain, set passed=false and confidence below 0.6.\n"
        "Separate subject existence, attributes, spatial relations, and interaction relations.\n"
        "Treat text, symbols, logos, marks, signs, and 'shows/displays/on cover' as symbol_text_relations, not physical contact.\n"
        "For interaction relations, explicitly say if the target is confused with another object or part.\n"
        "Return exactly one JSON object with this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Original user prompt: {user_prompt}\n"
        f"Generated prompt: {generated_prompt or user_prompt}\n"
        f"Structured user intent: {json.dumps(intent, ensure_ascii=False)}\n"
    )


def parse_specialist_observation_response(raw_response: str) -> dict[str, Any]:
    """Parse a VLM JSON response, tolerating fenced JSON."""

    text = str(raw_response or "").strip()
    if not text:
        return _empty_observation("empty VLM response")
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return _normalize_observation(parsed)
    return _empty_observation(f"failed to parse VLM response: {text[:300]}")


def build_specialist_reports(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
    *,
    generated_prompt: str = "",
) -> list[SpecialistReport]:
    """Build specialist reports from one structured observation."""

    return [
        _subject_report(observation, constraints),
        _attribute_report(observation, constraints),
        _spatial_report(observation, constraints),
        _interaction_report(observation, constraints, generated_prompt=generated_prompt),
        _symbol_text_report(observation, constraints),
        _style_report(observation, constraints),
    ]


def arbitrate_specialist_reports(
    reports: Sequence[SpecialistReport],
    constraints: PromptConstraints,
) -> SpecialistArbitration:
    failures = [
        failure
        for report in reports
        for failure in [*report.failures, *report.uncertain]
    ]
    global_passed = not failures
    if global_passed:
        return SpecialistArbitration(
            global_passed=True,
            dominant_failure="none",
            selected_action="complete",
            fallback_action="none",
            prompt_patch="",
            protected_constraints=_protected_constraints(constraints),
        )

    priority = {
        "missing_object": 0,
        "wrong_count": 1,
        "wrong_symbol_text": 2,
        "wrong_spatial_relation": 3,
        "wrong_relation": 4,
        "wrong_object_type": 4,
        "wrong_material": 4,
        "wrong_attribute": 5,
        "style_mismatch": 6,
        "uncertain": 6,
    }
    high_confidence_missing = _highest_confidence_missing_failure(failures)
    relation_conflict = _high_confidence_relation_conflict(failures)
    high_confidence_typed = _highest_confidence_typed_failure(failures)
    if relation_conflict is not None:
        dominant = relation_conflict
    elif high_confidence_missing is not None:
        dominant = high_confidence_missing
    elif high_confidence_typed is not None:
        dominant = high_confidence_typed
    else:
        dominant = sorted(failures, key=lambda item: priority.get(item.type, 99))[0]
    forbidden = _forbidden_phrases(failures, constraints)
    prompt_patch = _prompt_patch_for_failure(dominant, constraints, forbidden)
    selected_action = _selected_action_for_failure(dominant)
    fallback_action = (
        "layout_guided_regenerate"
        if dominant.type
        in {
            "wrong_relation",
            "wrong_spatial_relation",
            "missing_object",
            "wrong_count",
            "wrong_material",
            "wrong_object_type",
        }
        else "regenerate"
    )
    return SpecialistArbitration(
        global_passed=False,
        dominant_failure=_dominant_failure_family(dominant.type),
        selected_action=selected_action,
        fallback_action=fallback_action,
        prompt_patch=prompt_patch,
        forbidden_phrases=forbidden,
        protected_constraints=_protected_constraints(constraints),
    )


def _subject_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> SpecialistReport:
    observed = {
        _norm(item.get("name")): item
        for item in _records(observation.get("subjects"))
        if item.get("name")
    }
    failures: list[ConstraintFailure] = []
    uncertain: list[ConstraintFailure] = []
    failed_missing_targets: set[str] = set()
    for subject in constraints.subjects:
        if _is_support_subject(subject):
            continue
        item = _match_observed(subject, observed)
        if item and bool(item.get("visible", False)):
            continue
        failed_missing_targets.add(_norm(subject))
        failure = ConstraintFailure(
            type="missing_object",
            target=subject,
            evidence=str((item or {}).get("evidence") or f"{subject} is not visibly confirmed."),
            confidence=_float_or_none((item or {}).get("confidence")),
        )
        if item and _float_or_none(item.get("confidence")) is not None and _float_or_none(item.get("confidence")) < 0.6:
            uncertain.append(failure)
        else:
            failures.append(failure)
    counts = constraints.intent_spec.counts if constraints.intent_spec is not None else {}
    for object_name, expected_count in counts.items():
        if _is_support_subject(object_name) or _norm(object_name) in failed_missing_targets:
            continue
        item = _match_observed(object_name, observed)
        observed_count = _int_or_none((item or {}).get("count"))
        if item and bool(item.get("visible", False)) and observed_count == expected_count:
            continue
        if item and bool(item.get("visible", False)):
            evidence = (
                f"Expected exactly {expected_count} {object_name}, "
                f"observed {observed_count if observed_count is not None else 'unverified'}."
            )
            failure = ConstraintFailure(
                type="wrong_count",
                target=object_name,
                attribute="count",
                expected=str(expected_count),
                observed="" if observed_count is None else str(observed_count),
                evidence=str(item.get("evidence") or evidence),
                confidence=_float_or_none(item.get("confidence")),
            )
        else:
            failure = ConstraintFailure(
                type="missing_object",
                target=object_name,
                attribute="count",
                expected=str(expected_count),
                observed="0",
                evidence=str((item or {}).get("evidence") or f"{object_name} count target is not visibly confirmed."),
                confidence=_float_or_none((item or {}).get("confidence")),
            )
        if failure.confidence is not None and failure.confidence < 0.6:
            uncertain.append(failure)
        else:
            failures.append(failure)
    return SpecialistReport(
        agent="SubjectExistenceAgent",
        passed=not failures and not uncertain,
        failures=failures,
        uncertain=uncertain,
    )


def _attribute_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> SpecialistReport:
    attributes = _records(observation.get("attributes"))
    failures: list[ConstraintFailure] = []
    uncertain: list[ConstraintFailure] = []
    for object_name, expected_color in constraints.colors.items():
        item = _find_attribute(attributes, object_name, "color")
        if item and bool(item.get("passed", False)):
            continue
        failure = ConstraintFailure(
            type="wrong_attribute",
            target=object_name,
            attribute="color",
            expected=expected_color,
            observed=str((item or {}).get("observed") or ""),
            evidence=str((item or {}).get("evidence") or f"{object_name} color is not verified as {expected_color}."),
            confidence=_float_or_none((item or {}).get("confidence")),
        )
        if item and _float_or_none(item.get("confidence")) is not None and _float_or_none(item.get("confidence")) < 0.6:
            uncertain.append(failure)
        else:
            failures.append(failure)
    for object_name, expected_values in _expected_attribute_constraints(constraints).items():
        for expected in expected_values:
            item = _find_attribute(attributes, object_name, expected)
            if item is None and expected in _material_words():
                item = _find_attribute(attributes, object_name, "material")
            if item is None:
                item = _find_attribute(attributes, object_name, "object_type")
            if item and bool(item.get("passed", False)):
                continue
            failure_type = (
                "wrong_material"
                if expected in _material_words()
                else "wrong_object_type"
            )
            failure = ConstraintFailure(
                type=failure_type,
                target=object_name,
                attribute=expected,
                expected=expected,
                observed=str((item or {}).get("observed") or ""),
                evidence=str(
                    (item or {}).get("evidence")
                    or f"{object_name} is not verified as {expected}."
                ),
                confidence=_float_or_none((item or {}).get("confidence")),
            )
            if item and _float_or_none(item.get("confidence")) is not None and _float_or_none(item.get("confidence")) < 0.6:
                uncertain.append(failure)
            else:
                failures.append(failure)
    for item in _records(observation.get("negative_constraints")):
        if bool(item.get("passed", False)):
            continue
        constraint_text = str(item.get("constraint") or "negative_constraint")
        failures.append(
            ConstraintFailure(
                type=_negative_constraint_failure_type(constraint_text, item),
                target=constraint_text,
                attribute="negative_constraint",
                expected="pass",
                observed="failed",
                evidence=str(item.get("evidence") or "Negative constraint failed."),
                confidence=_float_or_none(item.get("confidence")),
            )
        )
    return SpecialistReport(
        agent="AttributeBindingAgent",
        passed=not failures and not uncertain,
        failures=failures,
        uncertain=uncertain,
    )


def _spatial_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> SpecialistReport:
    expected = (
        constraints.intent_spec.relations
        if constraints.intent_spec is not None
        else []
    )
    observed = _records(observation.get("spatial_relations"))
    failures: list[ConstraintFailure] = []
    for relation in expected:
        phrase = str(relation.get("phrase") or "")
        subject = str(relation.get("subject") or "")
        target = str(relation.get("object") or "")
        if not phrase:
            continue
        item = _find_relation(observed, subject, phrase, target)
        if item and bool(item.get("passed", False)):
            continue
        failures.append(
            ConstraintFailure(
                type="wrong_spatial_relation",
                target=f"{subject} {phrase} {target}".strip(),
                expected=phrase,
                observed=str((item or {}).get("evidence") or ""),
                evidence=str((item or {}).get("evidence") or f"Spatial relation not verified: {subject} {phrase} {target}"),
                confidence=_float_or_none((item or {}).get("confidence")),
            )
        )
    return SpecialistReport(
        agent="SpatialLayoutAgent",
        passed=not failures,
        failures=failures,
    )


def _interaction_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
    *,
    generated_prompt: str = "",
) -> SpecialistReport:
    expected = (
        constraints.intent_spec.interaction_relations
        if constraints.intent_spec is not None
        else []
    )
    observed = _records(observation.get("interaction_relations"))
    failures: list[ConstraintFailure] = []
    for relation in expected:
        subject = str(relation.get("subject") or "")
        action = str(relation.get("action") or "")
        target = str(relation.get("object") or "")
        if _is_display_relation(relation):
            continue
        item = _find_interaction(observed, subject, action, target)
        if item and bool(item.get("passed", False)):
            continue
        confused = str((item or {}).get("confused_with") or "")
        evidence = str((item or {}).get("evidence") or f"Interaction not verified: {subject} {action} {target}")
        if confused:
            evidence = f"{evidence} Confused with: {confused}."
        failures.append(
            ConstraintFailure(
                type="wrong_relation",
                target=target or f"{subject} {action}".strip(),
                attribute="interaction",
                expected=f"{subject} {action} {target}".strip(),
                observed=confused,
                evidence=evidence,
                confidence=_float_or_none((item or {}).get("confidence")),
            )
        )
    failures.extend(_generated_prompt_relation_conflicts(generated_prompt, constraints))
    return SpecialistReport(
        agent="InteractionRelationAgent",
        passed=not failures,
        failures=failures,
    )


def _symbol_text_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> SpecialistReport:
    expected = (
        constraints.intent_spec.interaction_relations
        if constraints.intent_spec is not None
        else []
    )
    observed = [
        *_records(observation.get("symbol_text_relations")),
        *_records(observation.get("interaction_relations")),
    ]
    failures: list[ConstraintFailure] = []
    for relation in expected:
        if not _is_display_relation(relation):
            continue
        subject = str(relation.get("subject") or "")
        action = str(relation.get("action") or "")
        target = str(relation.get("object") or "")
        item = _find_interaction(observed, subject, action, target)
        if item and bool(item.get("passed", False)):
            continue
        confused = str((item or {}).get("confused_with") or "")
        evidence = str((item or {}).get("evidence") or f"Symbol/text visibility not verified: {subject} {action} {target}")
        if confused:
            evidence = f"{evidence} Confused with: {confused}."
        failures.append(
            ConstraintFailure(
                type="wrong_symbol_text",
                target=target or f"{subject} {action}".strip(),
                attribute="symbol_text_visibility",
                expected=f"{subject} {action} {target}".strip(),
                observed=confused,
                evidence=evidence,
                confidence=_float_or_none((item or {}).get("confidence")),
            )
        )
    return SpecialistReport(
        agent="SymbolTextVisibilityAgent",
        passed=not failures,
        failures=failures,
    )


def _style_report(
    observation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> SpecialistReport:
    del observation, constraints
    return SpecialistReport(agent="StyleBackgroundAgent", passed=True)


def _normalize_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subjects": _records(value.get("subjects")),
        "attributes": _records(value.get("attributes")),
        "spatial_relations": _records(value.get("spatial_relations")),
        "interaction_relations": _records(value.get("interaction_relations")),
        "symbol_text_relations": _records(value.get("symbol_text_relations")),
        "negative_constraints": _records(value.get("negative_constraints")),
        "summary": deepcopy(dict(value.get("summary", {}))) if isinstance(value.get("summary"), Mapping) else {},
    }


def _empty_observation(reason: str) -> dict[str, Any]:
    return {
        "subjects": [],
        "attributes": [],
        "spatial_relations": [],
        "interaction_relations": [],
        "symbol_text_relations": [],
        "negative_constraints": [],
        "summary": {"global_passed": False, "dominant_failure": "unknown", "repair_hint": reason},
    }


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _find_attribute(
    attributes: Sequence[Mapping[str, Any]],
    object_name: str,
    attribute: str,
) -> Mapping[str, Any] | None:
    for item in attributes:
        if _norm(item.get("attribute")) != _norm(attribute):
            continue
        if _name_matches(object_name, item.get("object")):
            return item
    return None


def _expected_attribute_constraints(
    constraints: PromptConstraints,
) -> dict[str, list[str]]:
    intent = constraints.intent_spec
    if intent is None:
        return {}
    raw = getattr(intent, "attributes", {})
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, list[str]] = {}
    for object_name, values in raw.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        cleaned = [
            _norm_attribute(value)
            for value in values
            if _norm_attribute(value)
        ]
        if cleaned:
            result[str(object_name)] = list(dict.fromkeys(cleaned))
    return result


def _norm_attribute(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _material_words() -> set[str]:
    return {
        "ceramic",
        "fabric",
        "glass",
        "metal",
        "metallic",
        "paper",
        "plastic",
        "rubber",
        "steel",
        "stone",
        "wood",
        "wooden",
    }


def _find_relation(
    relations: Sequence[Mapping[str, Any]],
    subject: str,
    phrase: str,
    target: str,
) -> Mapping[str, Any] | None:
    phrase_norm = _relation_norm(phrase)
    for item in relations:
        if subject and not _name_matches(subject, item.get("subject")):
            continue
        if target and not _name_matches(target, item.get("object")):
            continue
        observed_phrase = item.get("relation") or item.get("phrase")
        if phrase_norm and phrase_norm not in _relation_norm(observed_phrase):
            continue
        return item
    return None


def _find_interaction(
    relations: Sequence[Mapping[str, Any]],
    subject: str,
    action: str,
    target: str,
) -> Mapping[str, Any] | None:
    action_norm = _action_norm(action)
    fallback: Mapping[str, Any] | None = None
    for item in relations:
        if subject and not _name_matches(subject, item.get("subject")):
            continue
        if action_norm and _action_norm(item.get("action")) != action_norm:
            continue
        if target and _name_matches(target, item.get("object")):
            return item
        if fallback is None:
            fallback = item
    return fallback


def _match_observed(
    subject: str,
    observed: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for name, item in observed.items():
        if _name_matches(subject, name):
            return item
    return None


def _forbidden_phrases(
    failures: Sequence[ConstraintFailure],
    constraints: PromptConstraints,
) -> list[str]:
    phrases: list[str] = []
    for failure in failures:
        phrases.extend(_generic_forbidden_phrases_for_failure(failure, constraints))
    return list(dict.fromkeys(phrases))


def _generic_forbidden_phrases_for_failure(
    failure: ConstraintFailure,
    constraints: PromptConstraints,
) -> list[str]:
    if failure.type not in {"missing_object", "wrong_relation"}:
        return []
    text = " ".join([failure.observed, failure.evidence]).lower()
    phrases: list[str] = []
    relations = (
        constraints.intent_spec.interaction_relations
        if constraints.intent_spec is not None
        else []
    )
    for relation in relations:
        action = str(relation.get("action") or "").strip()
        subject = str(relation.get("subject") or "").strip()
        target = str(relation.get("object") or "").strip()
        if _is_display_relation(relation):
            continue
        if not action or not target:
            continue
        for wrong_object in _candidate_wrong_objects(text, target, constraints, subject=subject):
            phrases.extend(_action_target_phrases(action, wrong_object))
            if _looks_like_part_confusion(text, wrong_object, target):
                phrases.append(f"{wrong_object} {_target_head(target)}".strip())
            if "attached" in text:
                phrases.append(f"attached to the {wrong_object}")
    if failure.observed:
        phrases.append(failure.observed)
    return _dedupe_text(phrases)


def _prompt_patch_for_failure(
    failure: ConstraintFailure,
    constraints: PromptConstraints,
    forbidden_phrases: Sequence[str],
) -> str:
    if failure.type == "missing_object":
        typed_target = _typed_object_phrase(failure.target, constraints)
        return (
            f"include the missing {typed_target} as a clearly visible separate object; "
            f"do not replace it with another object; keep all original required objects visible"
        )
    if failure.type == "wrong_count":
        target = failure.target or "target object"
        return f"show exactly {failure.expected} {target}, no extra or missing {target}"
    if failure.type == "wrong_spatial_relation":
        layout_hint = _layout_patch_for_spatial_failure(failure, constraints)
        if layout_hint:
            return layout_hint
        return (
            f"enforce the spatial layout from the original prompt: {failure.target}; "
            "keep all required objects visible and separated"
        )
    if failure.type == "wrong_relation":
        relation = _matching_interaction(failure.target, constraints)
        if relation:
            subject = relation.get("subject", "subject")
            action = relation.get("action", "holds")
            target = relation.get("object", failure.target)
            color = _color_for_target(str(target), constraints)
            colored_target = f"{color} {target}".strip()
            suffix = _safe_forbidden_suffix(forbidden_phrases, relation, constraints)
            return _relation_patch_phrase(subject, action, colored_target, suffix)
    if failure.type == "wrong_symbol_text":
        relation = _matching_symbol_text(failure.target, constraints)
        if relation:
            subject = str(relation.get("subject") or "object").strip() or "object"
            target = str(relation.get("object") or failure.target or "symbol/text").strip()
            color = _color_for_target(target, constraints)
            colored_target = f"{color} {target}".strip()
            return (
                f"{subject} clearly shows one visible {colored_target} in the requested place; "
                f"keep other required objects visible and do not add the symbol/text to the wrong object"
            )
        return f"make the requested text or symbol clearly visible: {failure.target}"
    if failure.type == "wrong_material":
        color = _color_for_target(failure.target, constraints)
        colored_target = f"{color} {failure.target}".strip()
        return (
            f"{colored_target} must visibly be made of {failure.expected}; "
            "make the material visually unambiguous, preserve its original color, "
            "and keep the required object separate"
        )
    if failure.type == "wrong_object_type":
        return (
            f"make the object type unambiguous: {failure.target} is a "
            f"{failure.expected} object, not a visually similar substitute"
        )
    if failure.type == "wrong_attribute" and failure.expected:
        return f"{failure.target} must visibly remain {failure.expected}"
    return failure.evidence


def _high_confidence_relation_conflict(
    failures: Sequence[ConstraintFailure],
) -> ConstraintFailure | None:
    relation_failures = [item for item in failures if item.type == "wrong_relation"]
    for failure in relation_failures:
        text = " ".join([failure.observed, failure.evidence]).lower()
        if (
            "prompt drift contradicts" in text
            or "confused with" in text
            or "rather than" in text
        ):
            return failure
    for failure in failures:
        if failure.type != "missing_object":
            continue
        text = " ".join([failure.target, failure.observed, failure.evidence]).lower()
        if any(marker in text for marker in ("instead", "rather than", "confused with")) and any(
            token in text
            for token in (
                "handle",
                "attached",
                "touch",
                "hold",
                "holding",
                "grip",
                "gripping",
                "carry",
                "carrying",
            )
        ):
            return ConstraintFailure(
                type="wrong_relation",
                target=failure.target,
                attribute="interaction",
                expected=failure.expected,
                observed=failure.observed,
                evidence=failure.evidence,
                confidence=failure.confidence,
            )
    return None


def _highest_confidence_typed_failure(
    failures: Sequence[ConstraintFailure],
) -> ConstraintFailure | None:
    typed = [
        item
        for item in failures
        if item.confidence is not None
        and item.confidence >= 0.75
        and item.type in {
            "wrong_count",
            "wrong_spatial_relation",
            "wrong_symbol_text",
            "wrong_relation",
            "wrong_attribute",
            "wrong_material",
            "wrong_object_type",
        }
    ]
    if not typed:
        return None
    priority = {
        "wrong_count": 0,
        "wrong_symbol_text": 1,
        "wrong_spatial_relation": 2,
        "wrong_relation": 3,
        "wrong_material": 4,
        "wrong_object_type": 4,
        "wrong_attribute": 5,
    }
    return sorted(
        typed,
        key=lambda item: (
            priority.get(item.type, 99),
            -(item.confidence or 0.0),
        ),
    )[0]


def _highest_confidence_missing_failure(
    failures: Sequence[ConstraintFailure],
) -> ConstraintFailure | None:
    missing = [
        item
        for item in failures
        if item.type == "missing_object"
        and (item.confidence is None or item.confidence >= 0.75)
    ]
    if not missing:
        return None
    return sorted(
        missing,
        key=lambda item: (
            0 if _target_has_material_or_color(item.target) else 1,
            -(item.confidence or 0.0),
        ),
    )[0]


def _target_has_material_or_color(target: str) -> bool:
    target_norm = _norm(target)
    if not target_norm:
        return False
    return any(word in target_norm.split() for word in _material_words()) or any(
        word in target_norm.split()
        for word in {
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "silver",
            "orange",
            "purple",
            "pink",
            "cyan",
            "teal",
            "magenta",
            "turquoise",
            "crimson",
            "indigo",
            "gold",
            "gray",
            "grey",
        }
    )


def _matching_interaction(
    target: str,
    constraints: PromptConstraints,
) -> Mapping[str, Any] | None:
    if constraints.intent_spec is None:
        return None
    for relation in constraints.intent_spec.interaction_relations:
        if _is_display_relation(relation):
            continue
        if _name_matches(target, relation.get("object")):
            return relation
    if constraints.intent_spec.interaction_relations:
        return constraints.intent_spec.interaction_relations[0]
    return None


def _layout_patch_for_spatial_failure(
    failure: ConstraintFailure,
    constraints: PromptConstraints,
) -> str:
    if constraints.intent_spec is None:
        return ""
    target = _norm(failure.target)
    for relation in constraints.intent_spec.relations:
        subject = str(relation.get("subject") or "").strip()
        phrase = str(relation.get("phrase") or "").strip()
        obj = str(relation.get("object") or "").strip()
        if not subject or not phrase or not obj:
            continue
        relation_text = f"{subject} {phrase} {obj}"
        if target and not _name_matches(target, relation_text):
            continue
        subject_colored = _colored_object_phrase(subject, constraints)
        object_colored = _colored_object_phrase(obj, constraints)
        normalized = phrase.replace("_", " ").lower()
        if normalized in {"right of", "right"}:
            return (
                f"strict 2D layout: place {subject_colored} on the right side of "
                f"{object_colored}, with clear horizontal separation and no overlap; "
                "keep object shapes unambiguous"
            )
        if normalized in {"left of", "left"}:
            return (
                f"strict 2D layout: place {subject_colored} on the left side of "
                f"{object_colored}, with clear horizontal separation and no overlap; "
                "keep object shapes unambiguous"
            )
        if normalized in {"above", "over"}:
            return (
                f"strict 2D layout: place {subject_colored} above {object_colored}, "
                "with clear vertical separation and no overlap; keep object shapes unambiguous"
            )
        if normalized in {"under", "below"}:
            return (
                f"strict 2D layout: place {subject_colored} below {object_colored}, "
                "with clear vertical separation and no overlap; keep object shapes unambiguous"
            )
        if normalized in {"behind"}:
            return (
                f"strict layout: place {subject_colored} visibly behind {object_colored}; "
                "keep both objects visible and separated"
            )
        if normalized in {"in front of", "front of"}:
            return (
                f"strict layout: place {subject_colored} visibly in front of {object_colored}; "
                "keep both objects visible and separated"
            )
    return ""


def _colored_object_phrase(object_name: str, constraints: PromptConstraints) -> str:
    color = _color_for_target(object_name, constraints)
    return f"{color} {object_name}".strip()


def _typed_object_phrase(object_name: str, constraints: PromptConstraints) -> str:
    color = _color_for_target(object_name, constraints)
    material = _expected_material_for_target(object_name, constraints)
    parts: list[str] = []
    for part in (color, material, object_name):
        text = str(part or "").strip()
        if not text:
            continue
        existing = " ".join(parts).lower().split()
        words = text.split()
        lowered_words = [word.lower() for word in words]
        while lowered_words and lowered_words[0] in existing:
            lowered_words.pop(0)
            words.pop(0)
        if not words or all(word.lower() in existing for word in words):
            continue
        parts.append(" ".join(words))
    return " ".join(parts).strip()


def _expected_material_for_target(
    target: str,
    constraints: PromptConstraints,
) -> str:
    if constraints.intent_spec is None:
        return ""
    attributes = getattr(constraints.intent_spec, "attributes", {})
    if not isinstance(attributes, Mapping):
        return ""
    for object_name, values in attributes.items():
        if not _name_matches(target, object_name):
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            value_text = str(value or "").strip().lower()
            if value_text in _material_words():
                return value_text
    return ""


def _generated_prompt_relation_conflicts(
    generated_prompt: str,
    constraints: PromptConstraints,
) -> list[ConstraintFailure]:
    if not generated_prompt or constraints.intent_spec is None:
        return []
    failures: list[ConstraintFailure] = []
    for relation in constraints.intent_spec.interaction_relations:
        if _is_display_relation(relation):
            continue
        subject = str(relation.get("subject") or "")
        action = str(relation.get("action") or "")
        target = str(relation.get("object") or "")
        expected = f"{subject} {action} {target}".strip()
        for wrong_target in constraints.subjects:
            if _is_support_subject(wrong_target):
                continue
            if _name_matches(wrong_target, subject) or _name_matches(wrong_target, target):
                continue
            if not _contains_action_target(generated_prompt, action, wrong_target):
                continue
            observed = f"{_surface_action(action)} the {wrong_target}"
            failures.append(
                ConstraintFailure(
                    type="wrong_relation",
                    target=target,
                    attribute="interaction",
                    expected=expected,
                    observed=observed,
                    evidence=(
                        "Generated prompt drift contradicts the original "
                        f"interaction: observed '{observed}', expected '{expected}'."
                    ),
                    confidence=1.0,
                )
            )
    return failures


def _selected_action_for_failure(failure: ConstraintFailure) -> str:
    if failure.type == "wrong_attribute":
        return "attribute_repair"
    if failure.type == "wrong_material":
        return "material_repair_or_regenerate"
    if failure.type == "wrong_object_type":
        return "object_type_repair_or_regenerate"
    if failure.type == "missing_object":
        return "object_insertion_or_regenerate"
    if failure.type == "wrong_count":
        return "count_repair_or_regenerate"
    if failure.type == "wrong_relation":
        return "relation_repair_or_object_insertion"
    if failure.type == "wrong_symbol_text":
        return "symbol_text_repair_or_regenerate"
    if failure.type == "wrong_spatial_relation":
        return "layout_guided_regeneration"
    return "regenerate"


def _safe_forbidden_suffix(
    forbidden_phrases: Sequence[str],
    relation: Mapping[str, Any],
    constraints: PromptConstraints,
) -> str:
    if not forbidden_phrases:
        return ""
    clauses: list[str] = []
    wrong_objects = _wrong_objects_from_forbidden(forbidden_phrases, constraints, relation)
    for wrong_object in wrong_objects:
        color = constraints.colors.get(wrong_object, "").strip()
        colored_wrong = f"{color} {wrong_object}".strip()
        clauses.append(f"physically separate from the {colored_wrong}")
    subject = str(relation.get("subject") or "subject").strip() or "subject"
    target = str(relation.get("object") or "target object").strip() or "target object"
    if wrong_objects:
        clauses.append(f"the {subject}'s {_relation_contact_part(subject)} stay on the separate {target}")
    if not clauses:
        clauses.append("clearly separated from any wrong object")
    return ", " + ", ".join(dict.fromkeys(clauses))


def _wrong_objects_from_forbidden(
    forbidden_phrases: Sequence[str],
    constraints: PromptConstraints,
    relation: Mapping[str, Any],
) -> list[str]:
    haystack = " ".join(str(item) for item in forbidden_phrases).lower()
    result: list[str] = []
    expected_subject = str(relation.get("subject") or "")
    expected_target = str(relation.get("object") or "")
    for subject in constraints.subjects:
        if _is_support_subject(subject):
            continue
        if expected_subject and _name_matches(subject, expected_subject):
            continue
        if expected_target and _name_matches(subject, expected_target):
            continue
        if _subject_mentioned(haystack, subject):
            result.append(subject)
    return _dedupe_text(result)


def _candidate_wrong_objects(
    text: str,
    expected_target: str,
    constraints: PromptConstraints,
    *,
    subject: str = "",
) -> list[str]:
    result: list[str] = []
    expected_subject = subject
    for candidate in constraints.subjects:
        if _is_support_subject(candidate):
            continue
        if _name_matches(candidate, expected_target):
            continue
        if expected_subject and _name_matches(candidate, expected_subject):
            continue
        if _subject_mentioned(text, candidate):
            result.append(candidate)
    return _dedupe_text(result)


def _color_for_target(target: str, constraints: PromptConstraints) -> str:
    target = str(target or "").strip()
    if not target:
        return ""
    if target in constraints.colors:
        return constraints.colors[target]
    for object_name, color in constraints.colors.items():
        if _name_matches(object_name, target) or _name_matches(target, object_name):
            return color
    return ""


def _subject_mentioned(text: str, subject: str) -> bool:
    normalized_text = f"_{_norm(text)}_"
    normalized_terms = sorted(_terms(subject), key=len, reverse=True)
    return any(
        f"_{term}_" in normalized_text
        for term in normalized_terms
        if term and len(term) > 2
    )


def _looks_like_part_confusion(text: str, wrong_object: str, expected_target: str) -> bool:
    expected_head = _target_head(expected_target)
    if not expected_head:
        return False
    lowered = text.lower()
    return wrong_object.lower() in lowered and expected_head.lower() in lowered


def _target_head(target: str) -> str:
    parts = str(target or "").strip().split()
    return parts[-1] if parts else ""


def _relation_contact_part(subject: Any) -> str:
    text = str(subject or "").lower()
    if any(token in text for token in ("bird", "dragon", "griffin")):
        return "claws"
    if any(token in text for token in ("cat", "dog", "bear", "lion", "tiger", "fox")):
        return "front paws"
    if any(token in text for token in ("robot", "person", "woman", "man", "wizard", "child")):
        return "hands"
    return "contact points"


def _relation_patch_phrase(
    subject: Any,
    action: Any,
    colored_target: str,
    suffix: str,
) -> str:
    subject_text = str(subject or "subject").strip() or "subject"
    target_text = str(colored_target or "target object").strip() or "target object"
    action_norm = _action_norm(action)
    if action_norm == "wear":
        return f"{subject_text} is visibly wearing a separate {target_text}{suffix}"
    if action_norm == "ride":
        return f"{subject_text} is visibly riding a separate {target_text}{suffix}"
    if action_norm == "attach":
        return f"{subject_text} is visibly attached to a separate {target_text}{suffix}"
    subject_part = _relation_contact_part(subject_text)
    return (
        f"{subject_text}'s {subject_part} are visibly {_action_gerund(action)} "
        f"a separate {target_text}{suffix}"
    )


def _action_target_phrases(action: str, target: str) -> list[str]:
    target = str(target or "").strip()
    if not target:
        return []
    action_norm = _action_norm(action)
    if action_norm == "hold":
        return [f"holds the {target}", f"holding the {target}"]
    if action_norm == "grip":
        return [f"grips the {target}", f"gripping the {target}"]
    if action_norm == "carry":
        return [f"carries the {target}", f"carrying the {target}"]
    if action_norm == "touch":
        return [f"touches the {target}", f"touching the {target}"]
    if action_norm == "wear":
        return [f"wears the {target}", f"wearing the {target}"]
    if action_norm == "ride":
        return [f"rides the {target}", f"riding the {target}"]
    if action_norm == "attach":
        return [f"attaches to the {target}", f"attached to the {target}"]
    return [f"{_surface_action(action)} the {target}"]


def _dedupe_text(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip(" ,.;")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _dominant_failure_family(error_type: str) -> str:
    if error_type == "missing_object":
        return "subject_existence"
    if error_type == "wrong_count":
        return "count_quantity"
    if error_type == "wrong_relation":
        return "interaction_relation"
    if error_type == "wrong_symbol_text":
        return "symbol_text_visibility"
    if error_type == "wrong_material":
        return "material_binding"
    if error_type == "wrong_object_type":
        return "object_type_binding"
    if error_type == "wrong_attribute":
        return "attribute_binding"
    if error_type == "wrong_spatial_relation":
        return "spatial_layout"
    return error_type or "unknown"


def _protected_constraints(constraints: PromptConstraints) -> dict[str, Any]:
    return {
        "subjects": list(constraints.subjects),
        "colors": dict(constraints.colors),
        "relations": list(constraints.relations),
        "actions": list(constraints.actions),
    }


def _is_support_subject(subject: str) -> bool:
    return _norm(subject) in {"floor", "table", "background", "scene", "objects"}


def _is_display_relation(relation: Mapping[str, Any]) -> bool:
    action = _action_norm(relation.get("action"))
    target = _norm(relation.get("object"))
    phrase = _norm(relation.get("phrase"))
    return (
        action in {"show", "display"}
        or any(token in target for token in ("text", "symbol", "logo", "mark", "letter", "word", "sign"))
        or any(token in phrase for token in ("shows", "display", "symbol", "text", "cover"))
    )


def _matching_symbol_text(
    target: str,
    constraints: PromptConstraints,
) -> Mapping[str, Any] | None:
    if constraints.intent_spec is None:
        return None
    display_relations = [
        relation
        for relation in constraints.intent_spec.interaction_relations
        if _is_display_relation(relation)
    ]
    for relation in display_relations:
        if _name_matches(target, relation.get("object")):
            return relation
    return display_relations[0] if display_relations else None


def _terms(value: Any) -> set[str]:
    text = _norm(value)
    if not text:
        return set()
    terms = {text}
    parts = text.split("_")
    if parts:
        terms.add(parts[-1])
    if text.endswith("s") and len(text) > 3:
        terms.add(text[:-1])
    else:
        terms.add(f"{text}s")
    return terms


def _name_matches(expected: Any, observed: Any) -> bool:
    """Match object names without confusing different compound parts.

    This allows a base object to match a colored compound object while keeping
    different objects with the same part name distinct.
    """

    expected_norm = _norm(expected)
    observed_norm = _norm(observed)
    if not expected_norm or not observed_norm:
        return False
    if expected_norm == observed_norm:
        return True
    expected_parts = expected_norm.split("_")
    observed_parts = observed_norm.split("_")
    if len(expected_parts) == 1:
        return expected_norm in observed_parts
    if len(observed_parts) == 1:
        return observed_norm in expected_parts
    observed_suffixes = {
        "_".join(observed_parts[index:])
        for index in range(len(observed_parts))
    }
    expected_suffixes = {
        "_".join(expected_parts[index:])
        for index in range(len(expected_parts))
    }
    return expected_norm in observed_suffixes or observed_norm in expected_suffixes


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _relation_norm(value: Any) -> str:
    text = _norm(value)
    return text.replace("left_of", "left_of").replace("right_of", "right_of")


def _action_norm(value: Any) -> str:
    text = _norm(value)
    if text in {"hold", "holds", "holding"}:
        return "hold"
    if text in {"grip", "grips", "gripping", "grasp", "grasps", "grasping"}:
        return "grip"
    if text in {"carry", "carries", "carrying"}:
        return "carry"
    if text in {"touch", "touches", "touching"}:
        return "touch"
    if text in {"wear", "wears", "wearing"}:
        return "wear"
    if text in {"show", "shows", "showing"}:
        return "show"
    if text in {"display", "displays", "displaying"}:
        return "display"
    if text in {"ride", "rides", "riding"}:
        return "ride"
    if text in {"attach", "attaches", "attached", "attached_to", "attaching"}:
        return "attach"
    return text


def _action_gerund(value: Any) -> str:
    action = _action_norm(value)
    if action == "hold":
        return "holding"
    if action == "grip":
        return "gripping"
    if action == "carry":
        return "carrying"
    if action == "touch":
        return "touching"
    if action == "wear":
        return "wearing"
    if action == "show":
        return "showing"
    if action == "display":
        return "displaying"
    if action == "ride":
        return "riding"
    if action == "attach":
        return "attached to"
    if not action:
        return "holding"
    if action.endswith("e"):
        return f"{action[:-1]}ing"
    if action.endswith("ing"):
        return action
    return f"{action}ing"


def _surface_action(value: Any) -> str:
    action = _action_norm(value)
    if action == "hold":
        return "holds"
    if action == "grip":
        return "grips"
    if action == "carry":
        return "carries"
    if action == "touch":
        return "touches"
    if action == "wear":
        return "wears"
    if action == "show":
        return "shows"
    if action == "display":
        return "displays"
    if action == "ride":
        return "rides"
    if action == "attach":
        return "attaches to"
    return action or "holds"


def _contains_action_target(text: str, action: str, target: str) -> bool:
    text_norm = _norm(text)
    target_norm = _norm(target)
    if not text_norm or not target_norm:
        return False
    action_norm = _action_norm(action)
    variants = {
        "hold": {"hold", "holds", "holding"},
        "grip": {"grip", "grips", "gripping", "grasp", "grasps", "grasping"},
        "carry": {"carry", "carries", "carrying"},
        "touch": {"touch", "touches", "touching"},
        "wear": {"wear", "wears", "wearing"},
        "ride": {"ride", "rides", "riding"},
        "attach": {"attach", "attaches_to", "attached_to", "attaching_to"},
    }.get(action_norm, {action_norm})
    determiners = ("", "a_", "an_", "the_", "this_", "that_")
    return any(
        f"{variant}_{determiner}{target_norm}" in text_norm
        for variant in variants
        for determiner in determiners
        if variant
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _negative_constraint_failure_type(
    constraint_text: str,
    item: Mapping[str, Any],
) -> str:
    text = " ".join(
        [
            str(constraint_text or ""),
            str(item.get("evidence") or ""),
            str(item.get("observed") or ""),
        ]
    ).lower()
    if re.search(
        r"\b(attach|attached|touch|touching|hold|holding|grip|gripping|carry|carrying|wear|wearing|ride|riding)\b",
        text,
    ):
        return "wrong_relation"
    return "wrong_attribute"


def _ensure_constraints(
    value: PromptConstraints | Mapping[str, Any] | str,
) -> PromptConstraints:
    if isinstance(value, PromptConstraints):
        return value
    return extract_constraints(value if isinstance(value, str) else str(value.get("original_prompt", "")))
