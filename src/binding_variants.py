"""Prompt variants for color, relation, and spatial binding failures."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .prompt_constraints import (
    COLOR_WORDS,
    PromptConstraints,
    approx_clip_token_count,
    lock_prompt_to_user_constraints,
)


def should_use_binding_variants(
    constraints: PromptConstraints,
    *,
    enabled: bool = True,
) -> bool:
    """Return true when prompt variants can help common cross-object leakage."""

    if not enabled:
        return False
    if _has_spatial_relations(constraints):
        return True
    if len(constraints.colors) < 2:
        return False
    return _has_physical_interaction(constraints)


def build_binding_variants(
    prompt: str,
    constraints: PromptConstraints,
    *,
    max_variants: int = 3,
    token_budget: int = 70,
) -> list[dict[str, Any]]:
    """Build short SDXL prompt variants that separate object-color bindings."""

    if max_variants < 1:
        return []
    base = _lock(prompt, constraints, token_budget)
    variants: list[dict[str, Any]] = [
        {"prompt": base, "strategy": "base", "reason": "Original locked prompt."}
    ]
    if not should_use_binding_variants(constraints):
        return variants[:max_variants]
    if _has_spatial_relations(constraints):
        variants.extend(_spatial_variants(base, constraints, token_budget))
        return _dedupe_variants(variants)[:max_variants]

    scene = _scene_phrase(constraints)
    color_segments = _color_segments(constraints)
    relation = _relation_segment(constraints)
    non_leaking = _remove_conflicting_color_adjectives(base, constraints)
    object_names = list(constraints.colors.keys())
    target_object = object_names[-1] if object_names else "object"
    subject_object = object_names[0] if object_names else "subject"

    variants.extend(
        [
            {
                "strategy": "color_first",
                "reason": "Put requested object colors before any composition details.",
                "prompt": _lock(
                    ", ".join([scene, *color_segments, relation, non_leaking]),
                    constraints,
                    token_budget,
                ),
            },
            {
                "strategy": "object_separation",
                "reason": "Describe the colored object separately to reduce color leakage.",
                "prompt": _lock(
                    ", ".join(
                        [
                            scene,
                            f"single {subject_object}",
                            f"separate vivid {constraints.colors[target_object]} {target_object}",
                            f"the {target_object} is not the same color as the {subject_object}",
                            relation,
                        ]
                    ),
                    constraints,
                    token_budget,
                ),
            },
            {
                "strategy": "minimal_binding",
                "reason": "Use a short literal prompt focused on the user binding.",
                "prompt": _lock(
                    ", ".join([scene, *color_segments, relation, "simple centered composition"]),
                    constraints,
                    token_budget,
                ),
            },
        ]
    )
    return _dedupe_variants(variants)[:max_variants]


def _spatial_variants(
    base: str,
    constraints: PromptConstraints,
    token_budget: int,
) -> list[dict[str, Any]]:
    spatial = _spatial_relation_specs(constraints)
    if not spatial:
        return []
    object_segments = _color_segments(constraints) or list(constraints.subjects)
    relation_segments = [_spatial_relation_phrase(item) for item in spatial]
    vertical = [
        phrase
        for phrase in relation_segments
        if any(term in phrase for term in (" under ", " above ", " below ", " behind ", " in front of "))
    ]
    horizontal = [
        phrase
        for phrase in relation_segments
        if any(term in phrase for term in (" left of ", " right of ", " next to "))
    ]
    variants: list[dict[str, Any]] = [
        {
            "strategy": "spatial_literal",
            "reason": "Short prompt with explicit user spatial relation clauses.",
            "prompt": _lock(
                ", ".join(
                    [
                        *object_segments,
                        *relation_segments,
                        "all objects clearly separated and fully visible",
                    ]
                ),
                constraints,
                token_budget,
            ),
        }
    ]
    if vertical or horizontal:
        variants.append(
            {
                "strategy": "spatial_axis_order",
                "reason": "Spell out vertical and horizontal ordering separately.",
                "prompt": _lock(
                    _join_nonempty(
                        *object_segments,
                        "vertical order: " + "; ".join(vertical) if vertical else "",
                        "horizontal order: " + "; ".join(horizontal) if horizontal else "",
                        "simple clean composition with no duplicate objects",
                        base,
                    ),
                    constraints,
                    token_budget,
                ),
            }
        )
    return variants


def _lock(prompt: str, constraints: PromptConstraints, token_budget: int) -> str:
    return lock_prompt_to_user_constraints(
        prompt,
        constraints,
        token_budget=token_budget,
    )["prompt"]


def _scene_phrase(constraints: PromptConstraints) -> str:
    for phrase in constraints.protected_phrases:
        lowered = phrase.lower()
        if sum(1 for term in ("cinematic", "photo", "rainy", "street") if term in lowered) >= 2:
            return phrase
    return constraints.original_prompt


def _color_segments(constraints: PromptConstraints) -> list[str]:
    segments: list[str] = []
    for object_name, color in constraints.colors.items():
        segments.append(f"{color} {object_name}")
    return _dedupe(segments)


def _relation_segment(constraints: PromptConstraints) -> str:
    relation = _first_interaction_relation(constraints)
    if relation:
        subject = relation.get("subject") or "subject"
        action = _progressive_action(str(relation.get("action") or "holding"))
        target = relation.get("object") or "object"
        return f"{subject} visibly {action} the {target}, clear physical contact"
    if any("grip" in action or "hold" in action for action in constraints.actions):
        target = _last_object_name(constraints, fallback="object")
        action = "gripping" if any("grip" in action for action in constraints.actions) else "holding"
        return f"subject visibly {action} the {target}, clear physical contact"
    return ", ".join(constraints.actions + constraints.relations)


def _has_physical_interaction(constraints: PromptConstraints) -> bool:
    if _first_interaction_relation(constraints):
        return True
    return any(_action_norm(action) in _PHYSICAL_ACTIONS for action in constraints.actions)


def _has_spatial_relations(constraints: PromptConstraints) -> bool:
    return bool(_spatial_relation_specs(constraints))


def _spatial_relation_specs(constraints: PromptConstraints) -> list[Mapping[str, str]]:
    if constraints.intent_spec is None:
        return []
    result: list[Mapping[str, str]] = []
    for relation in constraints.intent_spec.relations:
        if relation.get("subject") and relation.get("object"):
            result.append(relation)
    return result


def _spatial_relation_phrase(relation: Mapping[str, str]) -> str:
    subject = str(relation.get("subject") or "subject").strip()
    phrase = str(relation.get("relation") or relation.get("phrase") or "").strip()
    obj = str(relation.get("object") or "object").strip()
    return _normalize_spaces(f"{subject} {phrase} {obj}")


def _first_interaction_relation(constraints: PromptConstraints) -> Mapping[str, str] | None:
    if constraints.intent_spec is None:
        return None
    for relation in constraints.intent_spec.interaction_relations:
        action = str(relation.get("action") or "")
        if relation.get("object") and _action_norm(action) in _PHYSICAL_ACTIONS:
            return relation
    return None


def _last_object_name(constraints: PromptConstraints, *, fallback: str) -> str:
    if constraints.colors:
        return next(reversed(constraints.colors.keys()))
    if constraints.subjects:
        return constraints.subjects[-1]
    return fallback


def _progressive_action(action: str) -> str:
    action = _action_norm(action)
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
    if action == "ride":
        return "riding"
    if action == "attach":
        return "attached to"
    return action or "holding"


_PHYSICAL_ACTIONS = {"hold", "grip", "carry", "touch", "wear", "ride", "attach"}


def _action_norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
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


def _remove_conflicting_color_adjectives(
    prompt: str,
    constraints: PromptConstraints,
) -> str:
    result = prompt
    color_by_head = {
        object_name.split()[-1]: color
        for object_name, color in constraints.colors.items()
        if object_name.split()
    }
    for head, expected_color in color_by_head.items():
        for color in COLOR_WORDS - {expected_color}:
            result = re.sub(
                rf"\b{re.escape(color)}\s+{re.escape(head)}\b",
                f"{expected_color} {head}",
                result,
                flags=re.IGNORECASE,
            )
    result = re.sub(r"\bcrimson metallic\b", "clean red", result, flags=re.IGNORECASE)
    result = re.sub(r"\bglossy crimson\b", "red", result, flags=re.IGNORECASE)
    return _normalize_spaces(result)


def _dedupe_variants(variants: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant in variants:
        prompt = _normalize_spaces(str(variant.get("prompt", "")))
        key = prompt.lower()
        if not prompt or key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "prompt": prompt,
                "strategy": str(variant.get("strategy", "variant")),
                "reason": str(variant.get("reason", "")),
                "token_count": approx_clip_token_count(prompt),
            }
        )
    return result


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_spaces(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _join_nonempty(*values: str) -> str:
    return ", ".join(_normalize_spaces(value) for value in values if _normalize_spaces(value))


def _normalize_spaces(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,;.])", r"\1", value)
    value = re.sub(r"([,;])\s*", r"\1 ", value)
    return value.strip(" ,;")
