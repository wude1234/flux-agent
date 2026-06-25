"""Prompt-space mitigation for common SDXL attribute-binding failures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .prompt_constraints import (
    COLOR_WORDS,
    DEFAULT_CLIP_TOKEN_BUDGET,
    PromptConstraints,
    lock_prompt_to_user_constraints,
)


BINDING_FAILURE_THRESHOLD = 0.85
RETRY_CLIP_TOKEN_BUDGET = 72


def build_binding_retry_prompt(
    prompt: str,
    constraints: PromptConstraints | Mapping[str, Any],
    critique: Mapping[str, Any] | None = None,
    *,
    token_budget: int = DEFAULT_CLIP_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Return a revised prompt that foregrounds user-grounded bindings."""

    constraints = _ensure_constraints(constraints)
    critique = deepcopy(dict(critique or {}))
    binding_segments = _binding_segments(constraints, critique)
    if not binding_segments:
        locked = lock_prompt_to_user_constraints(
            prompt,
            constraints,
            token_budget=token_budget,
        )
        return {
            "prompt": locked["prompt"],
            "applied": locked["applied"],
            "negative_prompt": build_negative_prompt(constraints, critique),
            "reasons": [],
        }

    combined = ", ".join([*binding_segments, prompt])
    locked = lock_prompt_to_user_constraints(
        combined,
        constraints,
        token_budget=min(token_budget, RETRY_CLIP_TOKEN_BUDGET),
    )
    return {
        "prompt": locked["prompt"],
        "applied": locked["applied"],
        "negative_prompt": build_negative_prompt(constraints, critique),
        "reasons": _binding_failure_reasons(critique),
    }


def build_negative_prompt(
    constraints: PromptConstraints | Mapping[str, Any],
    critique: Mapping[str, Any] | None = None,
    *,
    extra_negative_prompt: str | None = None,
) -> str:
    """Build a compact negative prompt for known color/relation confusions."""

    constraints = _ensure_constraints(constraints)
    critique = deepcopy(dict(critique or {}))
    parts: list[str] = []
    requested_colors = set(constraints.colors.values())
    for object_name, color in constraints.colors.items():
        object_terms = _object_terms(object_name)
        conflict_colors = sorted(requested_colors - {color})
        for conflict_color in conflict_colors:
            for term in object_terms[:2]:
                if _negative_term_conflicts_with_required_object(
                    conflict_color,
                    term,
                    constraints,
                ):
                    continue
                parts.append(f"{conflict_color} {term}")
        for conflict_color in _explicit_conflict_colors(object_name, color, critique):
            for term in object_terms[:2]:
                if _negative_term_conflicts_with_required_object(
                    conflict_color,
                    term,
                    constraints,
                ):
                    continue
                parts.append(f"{conflict_color} {term}")
        parts.append(f"wrong {object_terms[-1]} color")

    relation = _interaction_relation(constraints)
    if relation:
        target = str(relation.get("object") or "").strip()
        action = str(relation.get("action") or "").strip()
        parts.extend(_relation_negative_terms(target, action=action))

    for reason in _binding_failure_reasons(critique):
        parts.extend(_explicit_wrong_color_object_terms(reason, constraints))

    if extra_negative_prompt:
        parts.extend(_split_negative(extra_negative_prompt))
    return ", ".join(_dedupe_preserve_order(parts))


def has_binding_failure(
    critique: Mapping[str, Any],
    constraints: PromptConstraints | Mapping[str, Any],
) -> bool:
    """Return true when feedback points to a user-grounded binding failure."""

    constraints = _ensure_constraints(constraints)
    if not constraints.colors and not constraints.actions and not constraints.relations:
        return False
    if critique.get("constraint_check"):
        constraint_check = critique["constraint_check"]
        if isinstance(constraint_check, Mapping) and not bool(
            constraint_check.get("passed", True)
        ):
            return True

    score = _coerce_score(critique.get("score"), default=1.0)
    if score < BINDING_FAILURE_THRESHOLD and bool(critique.get("user_grounded", True)):
        return True

    haystack = " ".join(_binding_failure_reasons(critique)).lower()
    needles = [
        "wrong color",
        "asked for",
        "user asked",
        "does not match",
        "not visible",
        "not held",
        "not gripping",
        "handle",
        *constraints.protected_phrases,
        *constraints.actions,
        *constraints.relations,
    ]
    return any(needle and needle.lower() in haystack for needle in needles)


def merge_negative_prompts(*values: str | None) -> str | None:
    """Combine optional negative prompts without repeated comma segments."""

    parts: list[str] = []
    for value in values:
        if value:
            parts.extend(_split_negative(value))
    merged = ", ".join(_dedupe_preserve_order(parts))
    return merged or None


def _binding_segments(
    constraints: PromptConstraints,
    critique: Mapping[str, Any],
) -> list[str]:
    segments: list[str] = []
    for object_name, color in constraints.colors.items():
        segments.append(f"{color} {object_name}")

    relation = _interaction_relation(constraints)
    if relation:
        subject = str(relation.get("subject") or "subject").strip() or "subject"
        action = str(relation.get("action") or "holding").strip() or "holding"
        object_hint = str(relation.get("object") or "object").strip() or "object"
        action_phrase = _progressive_action(action)
        segments.extend(
            [
                f"{subject} visibly {action_phrase} the {object_hint}",
                _relation_support_segment(subject, action, object_hint),
            ]
        )

    return _dedupe_preserve_order(segments)


def _binding_failure_reasons(critique: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for item in critique.get("errors", []) or []:
        if isinstance(item, Mapping):
            reasons.extend(
                str(item.get(key, "")).strip()
                for key in ("evidence", "description", "reason", "prompt_span")
                if str(item.get(key, "")).strip()
            )
        else:
            text = str(item).strip()
            if text:
                reasons.append(text)
    hint = str(critique.get("revision_hint", "")).strip()
    if hint:
        reasons.append(hint)
    constraint_check = critique.get("constraint_check")
    if isinstance(constraint_check, Mapping):
        reasons.extend(_binding_failure_reasons(constraint_check))
    return reasons


def _explicit_conflict_colors(
    object_name: str,
    color: str,
    critique: Mapping[str, Any],
) -> list[str]:
    reasons = " ".join(_binding_failure_reasons(critique)).lower()
    if object_name.lower() not in reasons:
        return []
    return [
        candidate
        for candidate in sorted(COLOR_WORDS)
        if candidate != color and candidate in reasons
    ]


def _explicit_wrong_color_object_terms(
    reason: str,
    constraints: PromptConstraints,
) -> list[str]:
    lowered = str(reason or "").lower()
    terms: list[str] = []
    for object_name, expected_color in constraints.colors.items():
        if object_name.lower() not in lowered:
            continue
        for color in sorted(COLOR_WORDS):
            if color != expected_color and color in lowered:
                for term in _object_terms(object_name)[:2]:
                    terms.append(f"{color} {term}")
    return terms


def _relation_negative_terms(target: str, *, action: str = "") -> list[str]:
    target = str(target or "object").strip()
    head = target.split()[-1] if target.split() else target
    action_norm = _action_norm(action)
    parts = [
        "hidden contact point",
        f"floating {head}",
        f"{head} not visible",
    ]
    if action_norm == "hold":
        parts.extend(["no visible hold", f"{target} not held"])
    elif action_norm == "grip":
        parts.extend(["no visible grip", f"{target} not gripped"])
    elif action_norm == "carry":
        parts.extend(["no visible carrying contact", f"{target} not carried"])
    elif action_norm == "touch":
        parts.extend(["no visible touch", f"{target} not touched"])
    elif action_norm == "wear":
        parts.extend(["not visibly worn", f"{target} not worn"])
    elif action_norm == "ride":
        parts.extend(["not visibly ridden", f"{target} not ridden"])
    elif action_norm == "attach":
        parts.extend(["not visibly attached", f"{target} not attached"])
    else:
        parts.extend(["wrong interaction relation", f"{target} relation missing"])
    return parts


def _object_hint(constraints: PromptConstraints, *, fallback: str) -> str:
    if constraints.colors:
        return next(reversed(constraints.colors.keys())).split()[-1]
    if constraints.subjects:
        return constraints.subjects[-1].split()[-1]
    return fallback


def _interaction_relation(
    constraints: PromptConstraints,
) -> Mapping[str, str] | None:
    """Prefer the explicit user relation over last-object heuristics."""

    intent = constraints.intent_spec
    if intent is not None:
        for relation in intent.interaction_relations:
            action = str(relation.get("action") or "").lower()
            target = str(relation.get("object") or "").strip()
            if target and _is_physical_action(action):
                return relation
    return None


def _relation_support_segment(subject: str, action: str, target: str) -> str:
    action_norm = _action_norm(action)
    if action_norm in {"hold", "grip", "carry", "touch"}:
        return f"clear physical contact between {subject} and the {target}"
    if action_norm == "wear":
        return f"the {target} is visibly worn on {subject}"
    if action_norm == "ride":
        return f"{subject} is clearly positioned on the {target}"
    if action_norm == "attach":
        return f"{subject} is visibly attached to the {target}"
    return f"clear visible relation between {subject} and the {target}"


def _progressive_action(action: str) -> str:
    action_norm = _action_norm(action)
    if action_norm == "hold":
        return "holding"
    if action_norm == "grip":
        return "gripping"
    if action_norm == "carry":
        return "carrying"
    if action_norm == "touch":
        return "touching"
    if action_norm == "wear":
        return "wearing"
    if action_norm == "ride":
        return "riding"
    if action_norm == "attach":
        return "attached to"
    return action_norm or "interacting with"


def _action_norm(value: str) -> str:
    text = "_".join(str(value or "").strip().lower().split())
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
    if text in {"ride", "rides", "riding"}:
        return "ride"
    if text in {"attach", "attaches", "attached", "attached_to", "attaching"}:
        return "attach"
    return text


def _is_physical_action(action: str) -> bool:
    return _action_norm(action) in {
        "hold",
        "grip",
        "carry",
        "touch",
        "wear",
        "ride",
        "attach",
    }


def _object_terms(object_name: str) -> list[str]:
    terms = [object_name]
    head = object_name.split()[-1] if object_name.split() else object_name
    if head and head != object_name:
        terms.append(head)
    return _dedupe_preserve_order(terms)


def _negative_term_conflicts_with_required_object(
    color: str,
    term: str,
    constraints: PromptConstraints,
) -> bool:
    phrase = f"{color} {term}".strip().lower()
    for object_name, required_color in constraints.colors.items():
        required_phrase = f"{required_color} {object_name}".strip().lower()
        object_text = object_name.lower()
        if phrase == object_text or phrase == required_phrase:
            return True
        if object_text and object_text.split()[0] in COLOR_WORDS and object_text in phrase:
            return True
        if object_text.startswith(f"{color} ") and term == object_text.split()[-1]:
            return True
    return False


def _split_negative(value: str) -> list[str]:
    return [part.strip(" ,;") for part in value.split(",") if part.strip(" ,;")]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip(" ,;")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _coerce_score(value: Any, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score > 1.0:
        score /= 10.0
    return max(0.0, min(1.0, score))


def _ensure_constraints(
    value: PromptConstraints | Mapping[str, Any],
) -> PromptConstraints:
    if isinstance(value, PromptConstraints):
        return value
    return PromptConstraints(
        original_prompt=str(value.get("original_prompt", "")),
        colors={str(key): str(item) for key, item in value.get("colors", {}).items()},
        subjects=[str(item) for item in value.get("subjects", [])],
        actions=[str(item) for item in value.get("actions", [])],
        relations=[str(item) for item in value.get("relations", [])],
        protected_phrases=[str(item) for item in value.get("protected_phrases", [])],
    )
