"""Typed action backend policy helpers.

This module keeps P5.5 route-to-action prompt strategy out of the main
orchestrator loop.  The orchestrator decides when to call an action backend;
this module decides whether a route is action-backed and how to sample a small
set of route-specific candidates.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .prompt_constraints import PromptConstraints


TYPED_ACTION_BACKEND_ROUTES = {
    "count_aware_regeneration",
    "comparative_count_rerank",
    "layout_guided_regeneration",
    "multi_constraint_decompose",
    "comparative_attribute_binding",
    "role_action_binding_regeneration",
    "lexical_grounding_regeneration",
    "material_guided_regeneration",
    "object_type_guided_regeneration",
    "relation_focused_regeneration",
}


def typed_action_backend_route(route: str) -> bool:
    """Return true when a repair route should use candidate regeneration."""

    return str(route or "").strip() in TYPED_ACTION_BACKEND_ROUTES


def build_typed_action_prompt_variants(
    *,
    route: str,
    user_prompt: str,
    selected_prompt: str,
    repair_plan: Mapping[str, Any],
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
    count: int,
) -> list[dict[str, Any]]:
    """Build route-specific prompt variants for low-count reranking.

    These are intentionally generic typed variants, not prompt-specific fixes.
    The original constraints still get re-locked by the orchestrator before
    generation and hard-checked by the VLM after generation.
    """

    del constraints
    base = str(user_prompt or selected_prompt or "").strip()
    target = str(repair_plan.get("target_object") or "").strip()
    reason = str(repair_plan.get("reason") or critique.get("revision_hint") or "").strip()
    variants: list[dict[str, Any]] = []

    def add(prompt: str, strategy: str) -> None:
        prompt = re.sub(r"\s+", " ", prompt).strip(" ,")
        if not prompt:
            return
        if prompt in {item["prompt"] for item in variants}:
            return
        variants.append({"prompt": prompt, "strategy": strategy, "route": route})

    if route == "count_aware_regeneration":
        expected = repair_plan.get("expected_count")
        phrase = f"exactly {expected} visible {target}" if expected and target else target
        add(f"{base}, {phrase}, no extra duplicate {target}, all required objects clearly visible", "count_explicit")
        add(f"{base}, simple uncluttered composition, {phrase}, avoid extra copies", "count_uncluttered")
        add(f"{base}, front view, separated objects, {phrase}", "count_separated")
    elif route == "comparative_count_rerank":
        add(f"{base}, make the relative quantity obvious: {target}, clearly more of the requested larger group than the smaller group", "comparative_count_explicit")
        add(f"{base}, arrange the two compared groups side by side so the viewer can count them", "comparative_count_side_by_side")
        add(f"{base}, avoid ambiguous or hidden items; make the count comparison visually unmistakable", "comparative_count_unambiguous")
    elif route in {"layout_guided_regeneration", "relation_focused_regeneration"}:
        add(f"{base}, layout-guided composition, keep every object separated and fully visible, satisfy: {target}", "layout_explicit")
        add(f"{base}, clean diagram-like spatial arrangement, clear left/right/above/below relations, no overlapping objects", "layout_diagram")
        add(f"{base}, camera straight-on, simple background, spatial relation is the main visual priority", "layout_straight_on")
    elif route == "multi_constraint_decompose":
        add(f"{base}, first satisfy exact object counts and spatial layout, then preserve colors/materials/actions; simple uncluttered scene", "multi_count_layout_first")
        add(f"{base}, each required object visible once as requested, clear separation, no extra forbidden objects", "multi_separated")
        add(f"{base}, hard constraints prioritized over style: count, position, color/material, action, and negative constraints", "multi_hard_constraints")
    elif route == "comparative_attribute_binding":
        add(f"{base}, make each compared role visually distinct; the larger and smaller subjects must have different required attributes", "comparative_attribute_roles")
        add(f"{base}, separate the compared subjects, label the visual roles through size/pose/color differences without text", "comparative_attribute_separated")
        add(f"{base}, avoid giving the two compared subjects the same color or attribute", "comparative_attribute_negative")
    elif route == "role_action_binding_regeneration":
        add(f"{base}, separate the two role-specific subjects; each role performs only its assigned action", "role_action_separated")
        add(f"{base}, clear role cues, visible action pose, do not swap the actions between the two subjects", "role_action_no_swap")
        add(f"{base}, simple scene, one subject with the specified attribute doing its action and the other without that attribute doing the other action", "role_action_explicit")
    elif route == "lexical_grounding_regeneration":
        literal = target or base
        add(f"{base}, literal interpretation of '{literal}', centered subject, clear visual depiction, no unrelated objects", "lexical_literal")
        normalized = str(repair_plan.get("normalized_prompt") or "").strip()
        if not normalized:
            normalized = normalize_rare_word_prompt(base)
        add(f"{normalized}, clear centered subject, no unrelated objects", "lexical_normalized")
        add(f"{base}, if the term is fictional, depict a single coherent invented object matching the full word, not a random object", "lexical_fictional")
    else:
        add(f"{base}, fix this failure: {reason}, preserve all original constraints", "typed_route_reason")
        add(f"{base}, hard constraints first, avoid previous failure: {reason}", "typed_route_hard_constraints")
        add(f"{base}, simple composition, all requested subjects visible and correctly bound", "typed_route_simple")
    return variants[: max(1, int(count))]


def normalize_rare_word_prompt(prompt: str) -> str:
    replacements = {
        "tcennis": "tennis",
        "rpacket": "racket",
        "racket packet": "racket",
    }
    result = str(prompt)
    for source, target in replacements.items():
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.I)
    return result


def best_typed_action_candidate_index(candidate_checks: Sequence[Mapping[str, Any]]) -> int:
    if not candidate_checks:
        return 0

    def key(item: Mapping[str, Any]) -> tuple[int, float, int]:
        passed = int(bool(item.get("passed")) and not item.get("failed"))
        score = _coerce_float(item.get("score"), default=0.0) or 0.0
        error_count = len(item.get("errors", []) or [])
        return passed, score, -error_count

    best = max(range(len(candidate_checks)), key=lambda index: key(candidate_checks[index]))
    return int(best)


def typed_action_rejected_reasons(
    candidate_checks: Sequence[Mapping[str, Any]],
    selected_index: int,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for index, check in enumerate(candidate_checks):
        if index == selected_index:
            continue
        errors = check.get("errors", []) or []
        reasons.append(
            {
                "candidate_index": index,
                "score": check.get("score"),
                "passed": check.get("passed"),
                "error_count": len(errors),
                "first_error": _compact(errors[0]) if errors else None,
            }
        )
    return reasons


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact(value: Any, *, max_text: int = 300) -> Any:
    if isinstance(value, str):
        return value[:max_text]
    if isinstance(value, Mapping):
        return {str(k): _compact(v, max_text=max_text) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact(item, max_text=max_text) for item in value[:10]]
    return deepcopy(value)
