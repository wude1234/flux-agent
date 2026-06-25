"""Flexible repair planning for user-grounded T2I failures."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
from typing import Any, Mapping, Protocol, Sequence

from .clients import VLMClient
from .constraint_questions import generate_constraint_questions
from .prompt_constraints import PromptConstraints


REPAIR_ACTIONS = {
    "none",
    "recolor",
    "object_insertion",
    "relation_repair",
    "regenerate",
}

MATERIAL_WORDS = {
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

_NUMBER_TO_WORD = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
_COUNT_WORDS = {word: number for number, word in _NUMBER_TO_WORD.items()}


class RepairPlanner(Protocol):
    """Plan which repair tool should be used for the current generated image."""

    def plan(
        self,
        *,
        user_prompt: str,
        prompt: str,
        image_path: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        enabled_tools: Mapping[str, bool],
    ) -> dict[str, Any]:
        ...


class RuleBasedRepairPlanner:
    """Tool router that can optionally ask a VLM before falling back to rules."""

    def __init__(self, vlm: VLMClient | None = None) -> None:
        self.vlm = vlm

    def plan(
        self,
        *,
        user_prompt: str,
        prompt: str,
        image_path: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        enabled_tools: Mapping[str, bool],
    ) -> dict[str, Any]:
        request = build_repair_planning_request(
            user_prompt=user_prompt,
            prompt=prompt,
            image_path=image_path,
            critique=critique,
            constraints=constraints,
            enabled_tools=enabled_tools,
        )
        preflight = _prompt_preflight_route_plan(
            constraints=constraints,
            enabled_tools=enabled_tools,
        )
        if self.vlm is not None:
            try:
                raw_response = self.vlm.vision(request, [image_path])
                parsed = parse_repair_plan_response(
                    raw_response,
                    constraints=constraints,
                    enabled_tools=enabled_tools,
                )
                if preflight and _plan_can_inherit_typed_failure_route(parsed, preflight):
                    parsed = _merge_preflight_route(parsed, preflight)
                parsed = _apply_plan_safety_overrides(
                    parsed,
                    critique=critique,
                    constraints=constraints,
                    enabled_tools=enabled_tools,
                )
                parsed["source"] = "vlm_repair_planner"
                parsed["request"] = request
                parsed["raw_response"] = raw_response
                if parsed["primary_action"] != "none" or parsed.get("repairable") is False:
                    return parsed
            except Exception as exc:
                fallback = heuristic_repair_plan(
                    critique,
                    constraints=constraints,
                    enabled_tools=enabled_tools,
                )
                fallback["source"] = "heuristic_after_vlm_failure"
                fallback["vlm_error"] = str(exc)
                fallback["request"] = request
                return fallback

        if preflight:
            fallback = deepcopy(dict(preflight))
            fallback["source"] = "prompt_preflight_repair_planner"
            fallback["request"] = request
            return _apply_tool_availability(fallback, enabled_tools)

        fallback = heuristic_repair_plan(
            critique,
            constraints=constraints,
            enabled_tools=enabled_tools,
        )
        fallback["source"] = "heuristic_repair_planner"
        fallback["request"] = request
        return fallback


def build_repair_planning_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> str:
    """Build a VLM prompt for selecting the correct repair strategy."""

    feedback = _compact_feedback_for_request(critique)
    feedback_json = _truncate_text(
        json.dumps(feedback, ensure_ascii=False, sort_keys=True),
        18000,
    )
    schema = {
        "primary_action": "recolor",
        "tool_sequence": ["recolor"],
        "repairable": True,
        "target_object": "umbrella",
        "target_attribute": "color",
        "reason": "short reason grounded in the visible image",
        "preconditions": {
            "target_object_visible": True,
            "subject_visible": True,
            "handle_visible": True,
            "relation_locally_repairable": False,
        },
    }
    return "\n".join(
        [
            "You are the repair planner for a text-to-image agent.",
            "Your job is to choose the next tool that best helps satisfy the original user prompt.",
            "Analyze the already generated image and the feedback. Do not blindly choose a fixed tool.",
            "Use recolor only when the target object is visible but has the wrong color.",
            "Use object_insertion when a required subject/object is missing but there is a plausible empty/local region to add it.",
            "Use relation_repair only when both related objects and the relevant hand/handle/contact parts are visible and only a small local connection is wrong.",
            "Use regenerate when the structure is too wrong for local editing, such as missing handle, hidden hand, severe occlusion, wrong main subject, or no clear local edit target.",
            "Return exactly one JSON object.",
            f"Allowed actions: {sorted(REPAIR_ACTIONS)}",
            f"Enabled tools: {json.dumps(dict(enabled_tools), ensure_ascii=False, sort_keys=True)}",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Original user prompt: {user_prompt}",
            f"Current generation prompt: {prompt}",
            f"Extracted user constraints: {json.dumps(constraints.to_dict(), ensure_ascii=False, sort_keys=True)}",
            f"Feedback for the CURRENT selected image only: {feedback_json}",
            f"Image path: {image_path}",
        ]
    )


def parse_repair_plan_response(
    response: str,
    *,
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any]:
    """Parse a VLM repair-plan JSON object with conservative normalization."""

    data = _extract_json_object(str(response)) or {}
    action = _normalize_action(data.get("primary_action") or data.get("action"))
    sequence = _normalize_tool_sequence(data.get("tool_sequence"), primary_action=action)
    repairable = _to_bool(data.get("repairable", action not in {"none", "regenerate"}))
    target_object = _clean_optional(data.get("target_object") or data.get("object"))
    target_attribute = _clean_optional(data.get("target_attribute") or data.get("attribute"))
    if not target_object:
        target_object = _default_target_for_action(action, constraints)
    normalized = {
        "primary_action": action,
        "tool_sequence": sequence,
        "repairable": repairable,
        "target_object": target_object,
        "target_attribute": target_attribute,
        "reason": _clean_optional(data.get("reason") or data.get("rationale")),
        "preconditions": deepcopy(dict(data.get("preconditions", {})))
        if isinstance(data.get("preconditions"), Mapping)
        else {},
    }
    return _apply_tool_availability(normalized, enabled_tools)


def heuristic_repair_plan(
    critique: Mapping[str, Any],
    *,
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any]:
    """Choose a repair action from structured errors when no VLM plan is available."""

    present_targets = _targets_marked_present(critique, constraints)
    errors = _filter_stale_missing_errors(
        _collect_errors(critique),
        present_targets=present_targets,
        constraints=constraints,
    )
    haystack = " ".join(_error_text(item) for item in errors).lower()
    typed_plan = _typed_failure_route_plan(
        errors,
        constraints=constraints,
        present_targets=present_targets,
        enabled_tools=enabled_tools,
    )
    if typed_plan:
        return _apply_tool_availability(typed_plan, enabled_tools)
    occlusion_plan = _occlusion_repair_plan(
        errors,
        constraints,
        present_targets=present_targets,
    )
    if occlusion_plan:
        return _apply_tool_availability(occlusion_plan, enabled_tools)
    object_type_target = _find_object_type_or_material_failure(errors, constraints)
    if object_type_target:
        target_name, expected, failure_kind = object_type_target
        typed_route = (
            "material_guided_regeneration"
            if failure_kind == "wrong_material"
            else "object_type_guided_regeneration"
        )
        return _apply_tool_availability(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "typed_route": typed_route,
                "target_object": target_name,
                "target_attribute": expected or failure_kind,
                "reason": (
                    f"{target_name} has a {failure_kind.replace('_', ' ')} "
                    f"failure for '{expected}'. Regenerate with explicit "
                    "object-type/material wording before color repair."
                ),
                "preconditions": {
                    "object_type_or_material_failure": True,
                    "local_recolor_insufficient": True,
                },
            },
            enabled_tools,
        )
    recolor_target = _find_recolor_target_from_errors(errors, constraints)
    if recolor_target:
        object_name, target_color, source_color = recolor_target
        tool_sequence = ["recolor"]
        reason_suffix = ""
        if _has_relation_failure(errors, haystack) and enabled_tools.get("relation_repair", False):
            tool_sequence.append("relation_repair")
            reason_suffix = " Queue relation repair after color repair if the contact failure remains."
        return _apply_tool_availability(
            {
                "primary_action": "recolor",
                "tool_sequence": tool_sequence,
                "repairable": True,
                "target_object": object_name,
                "target_attribute": "color",
                "target_color": target_color,
                "source_color": source_color,
                "reason": f"{object_name} is visible but has the wrong color.{reason_suffix}",
                "preconditions": {"target_object_visible": True},
            },
            enabled_tools,
        )

    missing_target = _find_missing_target_from_errors(
        errors,
        constraints,
        present_targets=present_targets,
    )
    if missing_target:
        return _apply_tool_availability(
            {
                "primary_action": "object_insertion",
                "tool_sequence": ["object_insertion"],
                "repairable": True,
                "typed_route": "missing_required_object",
                "target_object": missing_target,
                "target_attribute": "presence",
                "reason": f"Required object appears missing: {missing_target}",
                "preconditions": {"target_object_visible": False},
            },
            enabled_tools,
        )

    count_target = _find_count_underflow_target(errors, constraints)
    if count_target:
        target_name, expected, observed = count_target
        if _count_insertion_is_high_risk(target_name, constraints):
            return _apply_tool_availability(
                {
                    "primary_action": "regenerate",
                    "tool_sequence": ["regenerate"],
                    "repairable": False,
                    "fallback_from": "object_insertion",
                    "target_object": target_name,
                    "target_attribute": "count",
                    "expected_count": expected,
                    "observed_count": observed,
                    "missing_count": max(1, expected - observed),
                    "reason": (
                        f"Required count is short for {target_name}, but the missing "
                        "instances participate in a hard action/spatial relation. "
                        "Regenerate with stronger layout constraints instead of "
                        "risking a small local insertion."
                    ),
                    "preconditions": {
                        "target_object_visible": observed > 0,
                        "local_insertion_high_risk": True,
                    },
                },
                enabled_tools,
            )
        return _apply_tool_availability(
            {
                "primary_action": "object_insertion",
                "tool_sequence": ["object_insertion"],
                "repairable": True,
                "target_object": target_name,
                "target_attribute": "count",
                "expected_count": expected,
                "observed_count": observed,
                "missing_count": max(1, expected - observed),
                "reason": (
                    f"Required count is short for {target_name}: "
                    f"expected {expected}, observed {observed}."
                ),
                "preconditions": {"target_object_visible": observed > 0},
            },
            enabled_tools,
        )

    count_mismatch_target = _find_any_count_mismatch_target(errors, constraints)
    if count_mismatch_target:
        target_name, expected, observed = count_mismatch_target
        return _apply_tool_availability(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "typed_route": "count_aware_regeneration",
                "target_object": target_name,
                "target_attribute": "count",
                "expected_count": expected,
                "observed_count": observed,
                "reason": (
                    f"Count mismatch for {target_name}: expected {expected}, "
                    f"observed {observed}. Use count-aware regeneration or "
                    "candidate reranking instead of local insertion."
                ),
                "preconditions": {"count_mismatch": True},
            },
            enabled_tools,
        )

    spatial_target = _find_spatial_failure_target(errors, constraints)
    if spatial_target:
        return _apply_tool_availability(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "typed_route": "layout_guided_regeneration",
                "target_object": spatial_target,
                "target_attribute": "spatial_relation",
                "reason": (
                    f"Spatial relation failed for {spatial_target}. Regenerate "
                    "with explicit layout guidance and keep all required "
                    "objects visible and separated."
                ),
                "preconditions": {"layout_guidance_required": True},
            },
            enabled_tools,
        )

    if _has_relation_failure(errors, haystack):
        if _relation_looks_unrepairable(haystack):
            return {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "target_object": _relation_target(constraints),
                "target_attribute": "relation",
                "reason": "Relation failure lacks visible local parts for a small edit.",
                "preconditions": {"relation_locally_repairable": False},
                "source": "heuristic_repair_planner",
            }
        return _apply_tool_availability(
            {
                "primary_action": "relation_repair",
                "tool_sequence": ["relation_repair"],
                "repairable": True,
                "target_object": _relation_target(constraints),
                "target_attribute": "relation",
                "reason": "Visible relation/contact appears locally repairable.",
                "preconditions": {"relation_locally_repairable": True},
            },
            enabled_tools,
        )

    return {
        "primary_action": "none",
        "tool_sequence": [],
        "repairable": False,
        "target_object": "",
        "target_attribute": "",
        "reason": "No enabled local repair is indicated by current feedback.",
        "preconditions": {},
    }


def _apply_tool_availability(
    plan: Mapping[str, Any],
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any]:
    result = deepcopy(dict(plan))
    action = _normalize_action(result.get("primary_action"))
    if action == "object_insertion" and not enabled_tools.get("object_insertion", False):
        result.update(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "fallback_from": action,
                "reason": f"{result.get('reason', '')} Object insertion tool is not enabled.",
            }
        )
        return result
    if action == "recolor" and not enabled_tools.get("recolor", False):
        result.update(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "fallback_from": action,
                "reason": f"{result.get('reason', '')} Recolor tool is not enabled.",
            }
        )
        return result
    if action == "relation_repair" and not enabled_tools.get("relation_repair", False):
        result.update(
            {
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "fallback_from": action,
                "reason": f"{result.get('reason', '')} Relation repair tool is not enabled.",
            }
        )
        return result
    result["primary_action"] = action
    result["tool_sequence"] = [
        item
        for item in _normalize_tool_sequence(result.get("tool_sequence"), primary_action=action)
        if item == "regenerate" or enabled_tools.get(item, item in {"none"})
    ]
    if action not in result["tool_sequence"] and action not in {"none"}:
        result["tool_sequence"].insert(0, action)
    return result


def _typed_failure_route_plan(
    errors: Sequence[Mapping[str, Any]],
    *,
    constraints: PromptConstraints,
    present_targets: set[str],
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any] | None:
    preflight = _prompt_preflight_route_plan(
        constraints=constraints,
        enabled_tools=enabled_tools,
    )
    if preflight and (
        str(preflight.get("typed_route")) == "unverifiable_rare_word_or_clarify"
        or _errors_support_prompt_preflight(errors)
    ):
        return preflight
    if not errors:
        return None
    lexical_target = _find_lexical_grounding_failure(errors, constraints)
    if lexical_target:
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "lexical_grounding_regeneration",
            "target_object": lexical_target,
            "target_attribute": "lexical_grounding",
            "reason": (
                f"The requested term '{lexical_target}' is rare, misspelled, "
                "or visually underspecified. Try literal-preserving and "
                "typo-normalized prompt variants; do not use local editing."
            ),
            "preconditions": {"lexical_grounding_uncertain": True},
        }
    comparative_count = _find_comparative_count_failure(errors, constraints)
    if comparative_count:
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "comparative_count_rerank",
            "target_object": comparative_count,
            "target_attribute": "comparative_count",
            "reason": (
                "Relative count/comparison failed; generate candidates that make "
                "the comparison visually explicit and rerank with comparison VQA."
            ),
            "preconditions": {"comparative_count_failure": True},
        }
    comparative_attribute = _find_comparative_attribute_failure(errors, constraints)
    if comparative_attribute:
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "comparative_attribute_binding",
            "target_object": comparative_attribute,
            "target_attribute": "comparative_attribute",
            "reason": (
                "Role-specific comparative attribute binding failed; regenerate "
                "or rerank role/object-specific variants before local editing."
            ),
            "preconditions": {"comparative_attribute_failure": True},
        }
    role_action = _find_role_action_binding_failure(errors, constraints)
    if role_action:
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "role_action_binding_regeneration",
            "target_object": role_action,
            "target_attribute": "role_action_binding",
            "reason": (
                "Role/action binding failed; regenerate with explicit role "
                "separation and rerank by action VQA instead of removal/edit."
            ),
            "preconditions": {"role_action_binding_failure": True},
        }
    if _broad_multi_failure(errors, constraints):
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "multi_constraint_decompose",
            "target_object": _dominant_target_from_errors(errors, constraints),
            "target_attribute": "multi_constraint",
            "reason": "Multiple hard constraint families failed; decompose failures before choosing local edit.",
            "preconditions": {"broad_multi_failure": True},
        }
    occlusion_plan = _occlusion_repair_plan(
        errors,
        constraints,
        present_targets=present_targets,
    )
    if occlusion_plan:
        return occlusion_plan
    forbidden_symbol = _find_forbidden_symbol_failure(errors, constraints)
    if forbidden_symbol:
        target, symbol = forbidden_symbol
        return _editable_route_plan(
            typed_route="forbidden_symbol_removal",
            route_action="object_insertion",
            target_object=target,
            target_attribute="forbidden_symbol",
            reason=(
                f"Forbidden symbol/text is present on {target}; remove or cover "
                f"{symbol or 'the forbidden mark'} while preserving the object."
            ),
            enabled_tools=enabled_tools,
            preconditions={"forbidden_symbol_present": True},
        )
    forbidden_object = _find_forbidden_object_failure(errors, constraints)
    if forbidden_object:
        return _editable_route_plan(
            typed_route="forbidden_object_removal",
            route_action="object_insertion",
            target_object=forbidden_object,
            target_attribute="forbidden_object",
            reason=f"Forbidden object is visible: remove or erase {forbidden_object}.",
            enabled_tools=enabled_tools,
            preconditions={"forbidden_object_present": True},
        )
    exact_text = _find_exact_text_failure(errors, constraints)
    if exact_text:
        target, expected_text = exact_text
        return _editable_route_plan(
            typed_route="exact_text_overlay",
            route_action="object_insertion",
            target_object=target or "text",
            target_attribute="exact_text",
            reason=f"Exact text is wrong; overlay the required text {expected_text}.",
            enabled_tools=enabled_tools,
            preconditions={"wrong_exact_text": True},
            extra={"exact_text": expected_text.strip("'\"")},
        )
    object_type_target = _find_object_type_or_material_failure(errors, constraints)
    if object_type_target:
        target_name, expected, failure_kind = object_type_target
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": (
                "material_guided_regeneration"
                if failure_kind == "wrong_material"
                else "object_type_guided_regeneration"
            ),
            "target_object": target_name,
            "target_attribute": expected or failure_kind,
            "reason": (
                f"{target_name} has a {failure_kind.replace('_', ' ')} "
                f"failure for '{expected}'. Regenerate with explicit "
                "object-type/material wording before local editing."
            ),
            "preconditions": {
                "object_type_or_material_failure": True,
                "local_recolor_insufficient": True,
            },
        }
    count_target = _find_any_count_mismatch_target(errors, constraints)
    spatial_target = _find_spatial_failure_target(errors, constraints)
    if count_target and spatial_target:
        target_name, expected, observed = count_target
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "layout_guided_regeneration",
            "target_object": spatial_target or target_name,
            "target_attribute": "count_and_spatial_relation",
            "expected_count": expected,
            "observed_count": observed,
            "reason": (
                "Count and spatial/layout constraints both failed; use "
                "layout-guided regeneration instead of local editing."
            ),
            "preconditions": {"count_mismatch": True, "layout_guidance_required": True},
        }
    if count_target:
        target_name, expected, observed = count_target
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "count_aware_regeneration",
            "target_object": target_name,
            "target_attribute": "count",
            "expected_count": expected,
            "observed_count": observed,
            "reason": (
                f"Count mismatch for {target_name}: expected {expected}, "
                f"observed {observed}. Prefer count-aware regeneration/rerank."
            ),
            "preconditions": {"count_mismatch": True},
        }
    if spatial_target:
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "layout_guided_regeneration",
            "target_object": spatial_target,
            "target_attribute": "spatial_relation",
            "reason": "Spatial relation failed; use layout-guided regeneration.",
            "preconditions": {"layout_guidance_required": True},
        }
    attribute_patch = _find_single_attribute_patch_target(errors, constraints)
    if attribute_patch:
        target, attribute = attribute_patch
        tool_sequence = ["recolor"]
        if _has_relation_failure(errors, " ".join(_error_text(item) for item in errors).lower()):
            tool_sequence.append("relation_repair")
        plan = {
            "primary_action": "recolor",
            "tool_sequence": tool_sequence,
            "repairable": True,
            "typed_route": "single_attribute_patch",
            "target_object": target,
            "target_attribute": attribute,
            "reason": f"Single visible object has wrong {attribute}; use localized attribute patch or rerank.",
            "preconditions": {"single_attribute_failure": True},
        }
        if attribute == "color":
            target_color = constraints.colors.get(target, "")
            if target_color:
                plan["target_color"] = target_color
                plan["source_color"] = _mentioned_wrong_color(
                    " ".join(_error_text(item).lower() for item in errors),
                    target_color,
                ) or ""
        return plan
    relation_target = _find_interaction_failure_target(errors, constraints)
    if relation_target:
        if _relation_has_local_contact_evidence(errors):
            return {
                "primary_action": "relation_repair",
                "tool_sequence": ["relation_repair"],
                "repairable": True,
                "typed_route": "relation_contact_repair",
                "target_object": relation_target,
                "target_attribute": "interaction_relation",
                "reason": "Interaction/contact relation failed and local contact evidence is present.",
                "preconditions": {"relation_locally_repairable": True},
            }
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "relation_focused_regeneration",
            "target_object": relation_target,
            "target_attribute": "interaction_relation",
            "reason": "Interaction/contact relation failed without local contact evidence; regenerate with relation emphasis.",
            "preconditions": {"relation_locally_repairable": False},
        }
    return None


def _editable_route_plan(
    *,
    typed_route: str,
    route_action: str,
    target_object: str,
    target_attribute: str,
    reason: str,
    enabled_tools: Mapping[str, bool],
    preconditions: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action = route_action if enabled_tools.get("object_insertion", False) else "regenerate"
    plan = {
        "primary_action": action,
        "tool_sequence": [action],
        "repairable": action != "regenerate",
        "typed_route": typed_route,
        "target_object": target_object,
        "target_attribute": target_attribute,
        "reason": reason if action != "regenerate" else f"{reason} Local edit tool is not enabled.",
        "preconditions": dict(preconditions),
    }
    if action == "regenerate":
        plan["fallback_from"] = route_action
    if extra:
        plan.update(dict(extra))
    return plan


def _plan_can_inherit_typed_failure_route(
    plan: Mapping[str, Any],
    typed_plan: Mapping[str, Any],
) -> bool:
    typed_route = str(typed_plan.get("typed_route") or "")
    if not typed_route:
        return False
    action = _normalize_action(plan.get("primary_action"))
    if action == "object_insertion" and typed_route == "relation_contact_repair":
        return False
    if action in {"none", "regenerate"}:
        return True
    if typed_route in {
        "forbidden_object_removal",
        "forbidden_symbol_removal",
        "exact_text_overlay",
        "count_aware_regeneration",
        "comparative_count_rerank",
        "comparative_attribute_binding",
        "layout_guided_regeneration",
        "multi_constraint_decompose",
        "relation_focused_regeneration",
        "role_action_binding_regeneration",
        "lexical_grounding_regeneration",
        "unverifiable_rare_word_or_clarify",
    }:
        return True
    if typed_route == "relation_contact_repair":
        return action in {"relation_repair", "object_insertion"}
    if typed_route == "single_attribute_patch":
        return action in {"recolor", "object_insertion", "relation_repair"}
    return False


def _broad_multi_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> bool:
    families: set[str] = set()
    for item in errors:
        family = _failure_family(item)
        if family:
            families.add(family)
    if len(families) < 3:
        return False
    text = " ".join(_error_text(item) for item in errors).lower()
    return len(_mentioned_constraint_objects(text, constraints)) >= 1 or len(errors) >= 3


def _dominant_target_from_errors(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str:
    counts: Counter[str] = Counter()
    for item in errors:
        text = _error_text(item).lower()
        target = _object_name_mentioned_in_text(text, constraints)
        if not target:
            target = _match_constraint_object(
                str(item.get("target") or item.get("prompt_span") or ""),
                constraints,
            ) or str(item.get("target") or item.get("prompt_span") or "").strip()
        if target:
            counts[target] += 1
    if counts:
        return counts.most_common(1)[0][0]
    names = _constraint_object_names(constraints)
    return names[0] if names else "multi-constraint scene"


def _failure_family(item: Mapping[str, Any]) -> str:
    text = _error_text(item).lower()
    category = str(item.get("category") or item.get("type") or "").lower()
    question_id = str(item.get("question_id") or "").lower()
    if _comparative_count_language(text):
        return "comparative_count"
    if _role_action_language(text):
        return "role_action"
    if _comparative_attribute_language(text):
        return "comparative_attribute"
    if "count" in category or question_id.startswith("count:") or _count_failure_text(text):
        return "count"
    if _spatial_language_in_text(text) or category in {"spatial_relation", "wrong_spatial_relation"}:
        return "spatial"
    if _text_or_symbol_failure_text(text):
        return "text_symbol"
    if _forbidden_presence_text(text):
        return "forbidden"
    if _interaction_language_in_text(text) or category in {"action_relation", "wrong_relation"}:
        return "interaction"
    if category in {"wrong_attribute", "color_binding", "material_binding"}:
        return "attribute"
    return ""


def _find_lexical_grounding_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    prompt = str(constraints.original_prompt or "")
    if not _prompt_looks_lexically_underspecified(prompt):
        return None
    text = " ".join(_error_text(item) for item in errors).lower()
    if not any(
        marker in text
        for marker in (
            "missing",
            "unrelated",
            "does not match",
            "incorrect",
            "wrong main subject",
            "entirely unrelated",
            "not satisfy",
        )
    ):
        return None
    names = _constraint_object_names(constraints)
    return names[0] if names else _lexical_target_from_prompt(prompt)


VISUAL_LEXICAL_VOCABULARY = {
    "airplane",
    "alien",
    "apple",
    "astronaut",
    "baseball",
    "bicycle",
    "bird",
    "boat",
    "building",
    "car",
    "cat",
    "cow",
    "dining",
    "dog",
    "flower",
    "glove",
    "horse",
    "laptop",
    "meter",
    "oven",
    "parking",
    "pizza",
    "racket",
    "refrigerator",
    "scarecrow",
    "spoon",
    "storefront",
    "surfboard",
    "table",
    "teddy",
    "tennis",
    "train",
    "umbrella",
    "wine",
}


def _prompt_preflight_route_plan(
    *,
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any] | None:
    prompt = str(constraints.original_prompt or "").strip()
    if not _prompt_looks_lexically_underspecified(prompt):
        return None
    normalized = _normalize_visual_misspelling_prompt(prompt)
    target = _lexical_target_from_prompt(prompt)
    if normalized and normalized.lower() != prompt.lower():
        return {
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "typed_route": "lexical_grounding_regeneration",
            "target_object": target,
            "target_attribute": "lexical_grounding",
            "normalized_prompt": normalized,
            "reason": (
                f"The short prompt appears misspelled or lexically unstable. "
                f"Use the normalized visual wording '{normalized}' as a candidate "
                "and verify against the original prompt."
            ),
            "preconditions": {
                "prompt_preflight": True,
                "lexical_grounding_uncertain": True,
                "normalized_prompt": normalized,
            },
        }
    if _prompt_is_probably_unverifiable_rare_word(prompt):
        return {
            "primary_action": "none",
            "tool_sequence": [],
            "repairable": False,
            "typed_route": "unverifiable_rare_word_or_clarify",
            "target_object": target,
            "target_attribute": "lexical_grounding",
            "reason": (
                f"The prompt term '{target}' is rare or visually underspecified "
                "without a definition. Mark for clarification instead of spending "
                "additional FLUX repair rounds."
            ),
            "preconditions": {
                "prompt_preflight": True,
                "lexical_grounding_uncertain": True,
                "needs_clarification": True,
            },
        }
    return None


def prompt_needs_lexical_preflight(constraints: PromptConstraints) -> bool:
    """Return whether the original prompt needs lexical/rare-word routing."""

    return _prompt_preflight_route_plan(
        constraints=constraints,
        enabled_tools={"regenerate": True},
    ) is not None


def _merge_preflight_route(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    result = {**deepcopy(dict(plan)), **deepcopy(dict(preflight))}
    result["fallback_from"] = plan.get("primary_action") or "vlm_plan"
    result["source_plan"] = {
        "primary_action": plan.get("primary_action"),
        "target_object": plan.get("target_object"),
        "target_attribute": plan.get("target_attribute"),
        "reason": plan.get("reason"),
    }
    result["preconditions"] = {
        **_dict_value(plan.get("preconditions")),
        **_dict_value(preflight.get("preconditions")),
        "planner_override": "prompt_preflight_route",
    }
    return result


def _errors_support_prompt_preflight(errors: Sequence[Mapping[str, Any]]) -> bool:
    text = " ".join(_error_text(item) for item in errors).lower()
    if not text.strip():
        return True
    return any(
        marker in text
        for marker in (
            "missing",
            "unrelated",
            "does not match",
            "incorrect",
            "wrong main subject",
            "entirely unrelated",
            "not satisfy",
            "unrecognizable",
            "unclear",
            "rare",
            "misspelled",
        )
    )


def _prompt_looks_lexically_underspecified(prompt: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z-]*", prompt)
    if len(words) > 4:
        return False
    return any(_rare_or_nonsense_word(word) or _near_visual_word(word) for word in words)


def _rare_or_nonsense_word(word: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z]", "", word)
    if len(cleaned) < 7:
        return False
    lower = cleaned.lower()
    common = {
        "tennis",
        "packet",
        "bicycle",
        "umbrella",
        "hamburger",
        "sandwich",
        "building",
        "computer",
        "keyboard",
        "triangle",
    }
    if lower in common:
        return False
    vowels = sum(1 for char in lower if char in "aeiou")
    consonants = len(lower) - vowels
    if vowels == 0 or consonants / max(1, vowels) >= 4:
        return True
    if cleaned[:1].isupper() and cleaned[1:].islower() and lower not in VISUAL_LEXICAL_VOCABULARY:
        return True
    unusual_pairs = sum(1 for pair in ("ck", "rp", "tc", "nm", "zr", "q", "xq", "bf", "pf") if pair in lower)
    return unusual_pairs >= 1 and len(lower) >= 7


def _near_visual_word(word: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z]", "", word).lower()
    if len(cleaned) < 4:
        return False
    return any(_visual_word_similarity(cleaned, vocab) >= 0.82 for vocab in VISUAL_LEXICAL_VOCABULARY)


def _normalize_visual_misspelling_prompt(prompt: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]*|[^A-Za-z]+", prompt)
    normalized: list[str] = []
    changed = False
    for token in words:
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", token):
            normalized.append(token)
            continue
        replacement = _closest_visual_word(token)
        if replacement and replacement.lower() != token.lower():
            normalized.append(_match_case(token, replacement))
            changed = True
        else:
            normalized.append(token)
    result = "".join(normalized).strip()
    result = re.sub(r"\btennis\s+racket\b", "tennis racket", result, flags=re.I)
    return result if changed else prompt


def _closest_visual_word(word: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z]", "", word).lower()
    if len(cleaned) < 4:
        return None
    best = ""
    best_score = 0.0
    for candidate in VISUAL_LEXICAL_VOCABULARY:
        score = _visual_word_similarity(cleaned, candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best if best_score >= 0.82 else None


def _visual_word_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _prompt_is_probably_unverifiable_rare_word(prompt: str) -> bool:
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z-]*", prompt) if word.strip()]
    if not words or len(words) > 3:
        return False
    normalized = _normalize_visual_misspelling_prompt(prompt)
    if normalized.lower() != prompt.lower():
        return False
    return any(_rare_or_nonsense_word(word) for word in words)


def _lexical_target_from_prompt(prompt: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]*", prompt)
    return " ".join(words[:3]) if words else "rare word prompt"


def _find_comparative_count_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    text = " ".join(_error_text(item) for item in errors).lower()
    prompt = str(constraints.original_prompt or "").lower()
    if not (_comparative_count_language(text) or _comparative_count_language(prompt)):
        return None
    if not any(marker in text for marker in ("violat", "wrong", "more", "fewer", "less", "than", "instead")):
        return None
    target = _object_name_mentioned_in_text(text, constraints)
    return target or _comparative_phrase(prompt) or "comparative count"


def _comparative_count_language(text: str) -> bool:
    return bool(
        re.search(r"\b(?:more|fewer|less)\s+[a-z0-9 -]{1,40}\s+than\s+[a-z0-9 -]{1,40}\b", text)
        or re.search(r"\b(?:larger|smaller|taller|shorter)\s+number\s+of\b", text)
    )


def _comparative_phrase(text: str) -> str:
    match = re.search(r"\b(?:more|fewer|less)\s+[a-z0-9 -]{1,60}\s+than\s+[a-z0-9 -]{1,60}\b", text)
    return match.group(0).strip() if match else ""


def _find_comparative_attribute_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    text = " ".join(_error_text(item) for item in errors).lower()
    prompt = str(constraints.original_prompt or "").lower()
    if not (_comparative_attribute_language(text) or _comparative_attribute_language(prompt)):
        return None
    if not any(marker in text for marker in ("same", "different", "violat", "wrong", "instead", "not")):
        return None
    target = _object_name_mentioned_in_text(text, constraints)
    return target or "comparative attribute"


def _comparative_attribute_language(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "different color",
            "same color",
            "different colour",
            "same colour",
            "larger person",
            "smaller person",
            "larger object",
            "smaller object",
        )
    )


def _find_role_action_binding_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    text = " ".join(_error_text(item) for item in errors).lower()
    prompt = str(constraints.original_prompt or "").lower()
    if not (_role_action_language(text) or _role_action_language(prompt)):
        return None
    if not any(marker in text for marker in ("wrong", "rather than", "not", "instead", "violat", "depicted as")):
        return None
    target = _object_name_mentioned_in_text(text, constraints)
    return target or "role action binding"


def _role_action_language(text: str) -> bool:
    return (
        bool(re.search(r"\bwith\s+[a-z0-9 -]{1,40}\b", text))
        and bool(re.search(r"\bwithout\s+[a-z0-9 -]{1,40}\b", text))
        and any(action in text for action in ("sing", "singing", "draw", "drawing", "hold", "holding", "wear", "wearing"))
    )


def _mentioned_constraint_objects(text: str, constraints: PromptConstraints) -> set[str]:
    result: set[str] = set()
    for object_name in _constraint_object_names(constraints):
        if any(term and re.search(rf"\b{re.escape(term)}\b", text) for term in _object_terms(object_name)):
            result.add(object_name)
    return result


def _find_forbidden_symbol_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, str] | None:
    for item in errors:
        text = _error_text(item).lower()
        if not _forbidden_presence_text(text):
            continue
        if not _symbol_text_in_text(text):
            continue
        explicit_target = str(item.get("target") or item.get("prompt_span") or "").strip()
        target = explicit_target or _object_name_mentioned_in_text(text, constraints)
        return target or "object", _symbol_word_in_text(text)
    return None


def _find_forbidden_object_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    for item in errors:
        text = _error_text(item).lower()
        if not _forbidden_presence_text(text):
            continue
        if _symbol_text_in_text(text):
            continue
        target = _forbidden_target_from_text(text, constraints)
        if target:
            return target
        raw_target = str(item.get("target") or item.get("prompt_span") or "").strip()
        if raw_target:
            return raw_target
    return None


def _forbidden_presence_text(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    if re.search(r"\bno\s+(?:visible\s+)?[a-z0-9-]+\b", text) and not any(
        marker in text
        for marker in (
            "violat",
            "extra",
            "forbidden",
            "should not",
            "must not",
            "plain",
            "blank",
            "without",
            "despite",
        )
    ):
        return False
    negative = any(
        marker in text
        for marker in (
            " no ",
            "no ",
            "without ",
            "forbidden",
            "should not",
            "must not",
            "not supposed",
            "plain",
            "blank",
        )
    )
    present = any(
        marker in text
        for marker in (
            "contains",
            "contain",
            "has ",
            "have ",
            "visible",
            "present",
            "appears",
            "shown",
            "depicted",
            "violat",
            "contradict",
            "extra",
        )
    )
    return negative and present


def _forbidden_target_from_text(text: str, constraints: PromptConstraints) -> str:
    candidates = _constraint_object_names(constraints)
    intent = constraints.intent_spec
    if intent is not None:
        candidates.extend(str(item) for item in getattr(intent, "negative_constraints", []) or [])
    for candidate in candidates:
        for term in _object_terms(str(candidate)):
            if not term or len(term) <= 2:
                continue
            if re.search(rf"\bno\s+{re.escape(term)}\b", text) or re.search(
                rf"\b{re.escape(term)}\b.{0,80}\b(?:violat|forbidden|should not|must not|present|visible|contains?)\b",
                text,
            ):
                return str(candidate)
    return ""


def _find_exact_text_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, str] | None:
    for item in errors:
        text = _error_text(item)
        lowered = text.lower()
        if not _exact_text_failure_text(lowered):
            continue
        expected = _quoted_text_from_text(text) or _expected_text_from_constraints(constraints)
        explicit_target = str(item.get("target") or item.get("prompt_span") or "").strip()
        target = explicit_target or _object_name_mentioned_in_text(lowered, constraints)
        return target, expected or "requested text"
    return None


def _exact_text_failure_text(text: str) -> bool:
    return any(marker in text for marker in ("exact text", "should read", "reads", "text")) and any(
        marker in text for marker in ("wrong", "incorrect", "instead", "not ", "violat")
    )


def _quoted_text_from_text(text: str) -> str:
    matches = re.findall(r"['\"]([^'\"]{1,12})['\"]", str(text or ""))
    return matches[-1] if matches else ""


def _expected_text_from_constraints(constraints: PromptConstraints) -> str:
    intent = constraints.intent_spec
    if intent is None:
        return ""
    raw = getattr(intent, "text", None)
    if isinstance(raw, str):
        return raw
    for value in getattr(intent, "style", []) or []:
        value_text = str(value)
        quoted = _quoted_text_from_text(value_text)
        if quoted:
            return quoted
    return ""


def _find_single_attribute_patch_target(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, str] | None:
    families = {_failure_family(item) for item in errors}
    families.discard("")
    if families - {"attribute", "interaction"}:
        return None
    attribute_errors: list[tuple[str, str]] = []
    for item in errors:
        category = str(item.get("category") or item.get("type") or "").lower()
        text = _error_text(item).lower()
        if category not in {"wrong_attribute", "color_binding"}:
            continue
        if _forbidden_presence_text(text) or _text_or_symbol_failure_text(text):
            continue
        target = _object_name_mentioned_in_text(text, constraints) or str(item.get("target") or "").strip()
        if not target:
            continue
        attribute = "material" if _material_in_text(text) else "color" if _color_failure_text(text) else "attribute"
        attribute_errors.append((target, attribute))
    unique_targets = {target for target, _ in attribute_errors}
    if len(attribute_errors) == 1 and len(unique_targets) == 1:
        return attribute_errors[0]
    return None


def _find_interaction_failure_target(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    for item in errors:
        text = _error_text(item).lower()
        category = str(item.get("category") or item.get("type") or "").lower()
        if category not in {"wrong_relation", "action_relation", "relation_error", "contact_error"} and not _interaction_language_in_text(text):
            continue
        if _spatial_language_in_text(text):
            continue
        return _relation_target(constraints) or str(item.get("target") or item.get("prompt_span") or "").strip()
    return None


def _relation_has_local_contact_evidence(errors: Sequence[Mapping[str, Any]]) -> bool:
    text = " ".join(_error_text(item) for item in errors).lower()
    if any(
        marker in text
        for marker in (
            "no visible",
            "not visible",
            "missing",
            "absent",
            "lacks",
            "not present",
            "hidden",
            "occluded",
        )
    ):
        return False
    return any(marker in text for marker in ("hand", "paw", "finger", "handle", "loop", "rim", "side", "top", "attached", "touch"))


def _interaction_language_in_text(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "hold",
            "holding",
            "grip",
            "gripping",
            "touch",
            "touching",
            "attached",
            "connect",
            "contact",
            "perch",
            "perches",
            "loop",
            "hook",
            "hug",
        )
    )


def _symbol_text_in_text(text: str) -> bool:
    return any(marker in text for marker in ("symbol", "text", "logo", "star", "moon", "triangle", "plus", "circle", "square", "letter"))


def _symbol_word_in_text(text: str) -> str:
    for marker in ("star", "moon", "triangle", "plus", "circle", "square", "symbol", "text", "logo"):
        if marker in text:
            return marker
    return ""


def _text_or_symbol_failure_text(text: str) -> bool:
    return _symbol_text_in_text(text) and any(
        marker in text for marker in ("wrong", "incorrect", "violat", "contradict", "extra", "instead", "not ")
    )


def _count_failure_text(text: str) -> bool:
    if any(marker in text for marker in ("too many", "extra", "only", "exactly", "count", "duplicate")):
        return True
    number_words = "|".join(re.escape(word) for word in _COUNT_WORDS)
    return bool(
        re.search(
            rf"\b(?:\d+|{number_words})\b.{0,40}\b(?:visible|objects?|items?|expected|observed|present)\b",
            text,
        )
    )


def _color_failure_text(text: str) -> bool:
    if any(marker in text for marker in ("wrong color", "instead of", "rather than", "color")):
        return True
    color_words = ("teal", "turquoise", "silver", "gold", "bronze")
    return any(re.search(rf"\bnot\s+{re.escape(color)}\b", text) for color in color_words)


def _collect_errors(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for key in ("errors",):
        value = record.get(key, [])
        errors.extend(_normalize_error_records(value))
    for nested_key in ("constraint_check", "evaluation", "relation_repair_verification"):
        nested = record.get(nested_key)
        if isinstance(nested, Mapping):
            errors.extend(_normalize_error_records(nested.get("errors", [])))
            for check in nested.get("checks", []) or []:
                if isinstance(check, Mapping) and check.get("passed") is False:
                    errors.append(deepcopy(dict(check)))
    return errors


def _normalize_error_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, str):
        return [{"type": "wrong_attribute", "evidence": value}]
    if not isinstance(value, Sequence):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(deepcopy(dict(item)))
        elif isinstance(item, str):
            result.append({"type": "wrong_attribute", "evidence": item})
    return result


def _error_text(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key, ""))
        for key in (
            "type",
            "target",
            "expected",
            "observed",
            "evidence",
            "description",
            "prompt_span",
            "message",
        )
    )


def _find_missing_target(text: str, constraints: PromptConstraints) -> str | None:
    object_names = _constraint_object_names(constraints)
    for object_name in object_names:
        terms = _object_terms(object_name)
        if any(_missing_object_mentioned(text, term) for term in terms):
            if not _is_minor_part(object_name):
                return object_name
    return None


def _find_missing_target_from_errors(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
    *,
    present_targets: set[str],
) -> str | None:
    object_names = _constraint_object_names(constraints)
    for item in errors:
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        if category not in {"missing_object", "entity_existence", "subject"}:
            continue
        text = _error_text(item).lower()
        for object_name in object_names:
            if _is_minor_part(object_name):
                continue
            if _match_constraint_object(object_name, constraints) in present_targets:
                continue
            terms = _object_terms(object_name)
            if any(_missing_object_mentioned(text, term) for term in terms):
                return object_name
    return None


def _filter_stale_missing_errors(
    errors: Sequence[Mapping[str, Any]],
    *,
    present_targets: set[str],
    constraints: PromptConstraints,
) -> list[dict[str, Any]]:
    if not present_targets:
        return [deepcopy(dict(item)) for item in errors]
    filtered: list[dict[str, Any]] = []
    for item in errors:
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        text = _error_text(item).lower()
        stale = False
        if category in {"missing_object", "entity_existence", "subject"}:
            for target in present_targets:
                terms = _object_terms(target)
                if any(_missing_object_mentioned(text, term) for term in terms):
                    stale = True
                    break
        if stale:
            continue
        filtered.append(deepcopy(dict(item)))
    return filtered


def _missing_object_mentioned(text: str, term: str) -> bool:
    term = re.escape(term.lower())
    patterns = [
        rf"\bmissing\s+(?:[a-z0-9-]+\s+){{0,3}}{term}\b",
        rf"\bno\s+(?:[a-z0-9-]+\s+){{0,3}}{term}\b",
        rf"\b{term}\s+(?:is|are|appears|looks)?\s*(?:not\s+present|not\s+visible|not\s+shown|absent|missing)\b",
        rf"\bdoes\s+not\s+show\s+(?:[a-z0-9-]+\s+){{0,3}}{term}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _find_recolor_target(
    text: str,
    constraints: PromptConstraints,
) -> tuple[str, str, str] | None:
    for object_name, target_color in constraints.colors.items():
        terms = _object_terms(object_name)
        if not any(term in text for term in terms):
            continue
        if any(pattern in text for pattern in ("missing", "not present", "no ")):
            continue
        source_color = _mentioned_wrong_color(text, target_color) or ""
        if (
            f"not {target_color}" in text
            or "wrong color" in text
            or ("instead of" in text and source_color)
        ):
            return object_name, target_color, source_color
        for color in _known_colors(constraints):
            if color != target_color and any(
                re.search(rf"\b{re.escape(color)}\b.{0,50}\b{re.escape(term)}\b", text)
                or re.search(rf"\b{re.escape(term)}\b.{0,50}\b{re.escape(color)}\b", text)
                for term in terms
            ):
                return object_name, target_color, color
    return None


def _find_recolor_target_from_errors(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, str, str] | None:
    texts = [_error_text(item).lower() for item in errors]
    for text in texts:
        target = _find_recolor_target(text, constraints)
        if target:
            return target
    return _find_recolor_target(" ".join(texts), constraints)


def _find_object_type_or_material_failure(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, str, str] | None:
    for item in errors:
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        text = _error_text(item).lower()
        if category in {"missing_object", "entity_existence", "subject", "count", "wrong_count"}:
            continue
        target = _match_constraint_object(
            str(item.get("target") or item.get("prompt_span") or ""),
            constraints,
        )
        expected = str(item.get("expected") or item.get("attribute") or "").strip().lower()
        if not target:
            target = _object_name_mentioned_in_text(text, constraints)
        if category in {"wrong_material", "material", "material_binding"}:
            return target or _default_target_for_action("regenerate", constraints), expected or _material_in_text(text), "wrong_material"
        if category in {"wrong_object_type", "object_type", "object_type_binding"}:
            return target or _default_target_for_action("regenerate", constraints), expected or "object type", "wrong_object_type"
        expected_material = _expected_material_for_target(target or "", constraints)
        if expected_material and _material_failure_text(text, expected_material):
            return target or _object_name_mentioned_in_text(text, constraints), expected_material, "wrong_material"
        if any(token in text for token in ("wrong object", "object type", "substitute", "visually similar")):
            if target:
                return target, expected or "object type", "wrong_object_type"
    return None


def _object_name_mentioned_in_text(
    text: str,
    constraints: PromptConstraints,
) -> str:
    for object_name in _constraint_object_names(constraints):
        if any(term and term in text for term in _object_terms(object_name)):
            return object_name
    return ""


def _expected_material_for_target(
    target: str,
    constraints: PromptConstraints,
) -> str:
    intent = constraints.intent_spec
    raw = getattr(intent, "attributes", {}) if intent is not None else {}
    if not isinstance(raw, Mapping):
        return ""
    matched = _match_constraint_object(target, constraints) or target
    for object_name, values in raw.items():
        if _match_constraint_object(str(object_name), constraints) != _match_constraint_object(matched, constraints):
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            value_text = str(value or "").strip().lower()
            if value_text in MATERIAL_WORDS:
                return value_text
    return ""


def _material_in_text(text: str) -> str:
    for material in MATERIAL_WORDS:
        if re.search(rf"\b{re.escape(material)}\b", text):
            return material
    return ""


def _material_failure_text(text: str, expected_material: str) -> bool:
    if not expected_material:
        return False
    if any(
        phrase in text
        for phrase in (
            "not glass",
            "not paper",
            "not fabric",
            "not metal",
            "not wooden",
            "not wood",
            "not made of",
            "wrong material",
            "material does not",
            "looks like plastic",
            "looks plastic",
            "looks like leather",
            "leather",
            "looks metallic",
            "electric fan",
        )
    ):
        return True
    if expected_material in text and any(marker in text for marker in ("instead of", "rather than", "not ")):
        return True
    if any(material in text for material in MATERIAL_WORDS) and any(
        marker in text for marker in ("instead of", "rather than", "wrong material")
    ):
        return True
    return bool(
        re.search(
            rf"\b{re.escape(expected_material)}\b.{0,60}\b(?:not|missing|unclear|failed)\b",
            text,
        )
    )


def _find_count_underflow_target(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, int, int] | None:
    for item in errors:
        question_id = str(item.get("question_id") or "").strip().lower()
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        if not (
            question_id.startswith("count:")
            or category == "count"
            or category == "wrong_count"
        ):
            continue
        expected = _parse_count_value(item.get("expected"))
        observed = _parse_count_value(item.get("observed"))
        if observed is None:
            observed = _parse_count_from_text(_error_text(item))
        if expected is None or observed is None or observed >= expected:
            continue
        target = str(item.get("target") or "").strip()
        if not target and question_id.startswith("count:"):
            target = question_id.split(":", 1)[1]
        target = _match_constraint_object(target, constraints) or target
        if target and not _is_minor_part(target):
            return target, expected, observed
    return None


def _find_any_count_mismatch_target(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> tuple[str, int, int] | None:
    for item in errors:
        question_id = str(item.get("question_id") or "").strip().lower()
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        if not (
            question_id.startswith("count:")
            or category == "count"
            or category == "wrong_count"
        ):
            continue
        expected = _parse_count_value(item.get("expected"))
        observed = _parse_count_value(item.get("observed"))
        text = _error_text(item)
        if expected is None:
            expected = _expected_count_from_constraints(item, constraints)
        if observed is None:
            observed = _parse_count_from_text(text)
        if expected is None or observed is None or observed == expected:
            continue
        target = str(item.get("target") or "").strip()
        if not target and question_id.startswith("count:"):
            target = question_id.split(":", 1)[1]
        target = _match_constraint_object(target, constraints) or target
        if target and not _is_minor_part(target):
            return target, expected, observed
    return None


def _expected_count_from_constraints(
    item: Mapping[str, Any],
    constraints: PromptConstraints,
) -> int | None:
    target = str(item.get("target") or item.get("prompt_span") or "").strip()
    question_id = str(item.get("question_id") or "").strip().lower()
    if not target and question_id.startswith("count:"):
        target = question_id.split(":", 1)[1]
    matched = _match_constraint_object(target, constraints) or target
    if constraints.intent_spec is None:
        return None
    for object_name, count in constraints.intent_spec.counts.items():
        if _match_constraint_object(matched, constraints) == _match_constraint_object(object_name, constraints):
            return int(count)
    return None


def _find_spatial_failure_target(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> str | None:
    for item in errors:
        question_id = str(item.get("question_id") or "").strip().lower()
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        text = _error_text(item).lower()
        if not (
            question_id.startswith("relation:")
            or category in {"spatial_relation", "wrong_spatial_relation"}
            or _spatial_language_in_text(text)
        ):
            continue
        relation = _matching_spatial_relation_text(text, constraints)
        if relation:
            return relation
        target = str(item.get("target") or item.get("prompt_span") or "").strip()
        return target or "spatial relation"
    return None


def _spatial_language_in_text(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "left of",
            "right of",
            "above",
            "below",
            "under",
            "behind",
            "in front of",
            "on top of",
            "spatial",
            "position",
            "positioned",
            "layout",
        )
    )


def _matching_spatial_relation_text(
    text: str,
    constraints: PromptConstraints,
) -> str:
    if constraints.intent_spec is None:
        return ""
    for relation in constraints.intent_spec.relations:
        subject = str(relation.get("subject") or "")
        phrase = str(relation.get("phrase") or "")
        obj = str(relation.get("object") or "")
        if not phrase:
            continue
        if phrase.lower() not in text and not _spatial_phrase_alias_in_text(phrase, text):
            continue
        if subject and subject.lower() not in text and obj and obj.lower() not in text:
            continue
        return f"{subject} {phrase} {obj}".strip()
    return ""


def _spatial_phrase_alias_in_text(phrase: str, text: str) -> bool:
    phrase = phrase.lower().replace("_", " ").strip()
    aliases = {
        "right of": ("right", "to the right"),
        "left of": ("left", "to the left"),
        "under": ("under", "below", "beneath"),
        "above": ("above", "over"),
        "behind": ("behind",),
        "in front of": ("in front", "front of"),
        "next to": ("next to", "beside"),
        "on top of": ("on top", "top of"),
    }
    return any(alias in text for alias in aliases.get(phrase, (phrase,)))


def _parse_count_value(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return _COUNT_WORDS.get(text)


def _parse_count_from_text(text: str) -> int | None:
    lowered = text.lower()
    for number in range(0, 11):
        word = _NUMBER_TO_WORD.get(number, str(number))
        if re.search(rf"\b(?:only\s+)?(?:{number}|{word})\b", lowered):
            return number
    return None


def _match_constraint_object(target: str, constraints: PromptConstraints) -> str | None:
    target = target.strip().lower()
    if not target:
        return None
    for object_name in _constraint_object_names(constraints):
        terms = _object_terms(object_name)
        if target in terms or any(term in target or target in term for term in terms):
            return object_name
    return None


def _has_relation_failure(errors: Sequence[Mapping[str, Any]], text: str) -> bool:
    relation_types = {"wrong_relation", "relation_error", "action_error", "contact_error"}
    if any(str(item.get("type", "")).lower() in relation_types for item in errors):
        return True
    return any(
        word in text
        for word in (
            "grip",
            "gripping",
            "hold",
            "holding",
            "handle",
            "contact",
            "connected",
            "detached",
        )
    )


def _relation_looks_unrepairable(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "no robot",
            "robot is not visible",
            "robot is not present",
            "no handle",
            "handle is not visible",
            "handle not visible",
            "missing handle",
            "hidden hand",
            "hand is hidden",
            "wrong main subject",
            "standalone mechanical arm",
            "not being held by",
            "not clearly visible or distinct",
        )
    )


def _relation_target(constraints: PromptConstraints) -> str:
    if constraints.actions:
        return constraints.actions[0]
    if len(constraints.subjects) >= 2:
        return " ".join(constraints.subjects[:2])
    return ""


def _count_insertion_is_high_risk(target_name: str, constraints: PromptConstraints) -> bool:
    target_terms = set(_object_terms(target_name))
    for question in generate_constraint_questions(constraints):
        if question.category not in {"action_relation", "spatial_relation"}:
            continue
        source = question.source_constraint
        subject = str(source.get("subject") or "").strip().lower()
        obj = str(source.get("object") or "").strip().lower()
        if not (
            any(term and term in _object_terms(subject) for term in target_terms)
            or any(term and term in _object_terms(obj) for term in target_terms)
        ):
            continue
        relation = str(source.get("relation") or source.get("action") or "").strip().lower()
        other = obj if any(term and term in _object_terms(subject) for term in target_terms) else subject
        if relation in {"near", "beside", "next_to"}:
            continue
        if relation == "on" and other in {"table", "wooden table", "ground", "floor", "grass"}:
            continue
        return True
    return False


def _apply_plan_safety_overrides(
    plan: Mapping[str, Any],
    *,
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any]:
    result = deepcopy(dict(plan))
    action = _normalize_action(result.get("primary_action"))
    present_targets = _targets_marked_present(critique, constraints)
    errors = _filter_stale_missing_errors(
        _collect_errors(critique),
        present_targets=present_targets,
        constraints=constraints,
    )
    haystack = " ".join(_error_text(item) for item in errors).lower()
    typed_plan = _typed_failure_route_plan(
        errors,
        constraints=constraints,
        present_targets=present_targets,
        enabled_tools=enabled_tools,
    )
    if typed_plan and _plan_can_inherit_typed_failure_route(result, typed_plan):
        planner_override = (
            "typed_occlusion_object_insertion"
            if typed_plan.get("typed_route") == "occlusion_object_insertion"
            else "typed_failure_route"
        )
        override = {
            **result,
            **typed_plan,
            "fallback_from": result.get("primary_action") or "regenerate",
            "source_plan": {
                "primary_action": result.get("primary_action"),
                "target_object": result.get("target_object"),
                "target_attribute": result.get("target_attribute"),
                "reason": result.get("reason"),
            },
            "reason": (
                f"{result.get('reason', '')} Safety override: current feedback "
                f"matches typed route {typed_plan.get('typed_route')}."
            ).strip(),
            "preconditions": {
                **_dict_value(result.get("preconditions")),
                **_dict_value(typed_plan.get("preconditions")),
                "planner_override": planner_override,
            },
        }
        return _apply_tool_availability(override, enabled_tools)
    object_type_target = _find_object_type_or_material_failure(errors, constraints)
    if object_type_target and action in {"recolor", "relation_repair", "object_insertion", "none"}:
        target_name, expected, failure_kind = object_type_target
        override = {
            **result,
            "primary_action": "regenerate",
            "tool_sequence": ["regenerate"],
            "repairable": False,
            "fallback_from": action,
            "typed_route": (
                "material_guided_regeneration"
                if failure_kind == "wrong_material"
                else "object_type_guided_regeneration"
            ),
            "target_object": target_name,
            "target_attribute": expected or failure_kind,
            "reason": (
                f"{result.get('reason', '')} Safety override: current evidence "
                f"indicates {failure_kind.replace('_', ' ')} for {target_name}, "
                "so color/local repair is insufficient."
            ).strip(),
            "preconditions": {
                **_dict_value(result.get("preconditions")),
                "object_type_or_material_failure": True,
                "planner_override": "typed_object_material_failure",
            },
        }
        return _apply_tool_availability(override, enabled_tools)
    occlusion_plan = _occlusion_repair_plan(
        errors,
        constraints,
        present_targets=present_targets,
    )
    if occlusion_plan and _plan_can_inherit_occlusion_typing(result, occlusion_plan, constraints):
        override = {
            **result,
            **occlusion_plan,
            "fallback_from": result.get("primary_action") or "regenerate",
            "source_plan": {
                "primary_action": result.get("primary_action"),
                "target_object": result.get("target_object"),
                "target_attribute": result.get("target_attribute"),
                "reason": result.get("reason"),
            },
            "reason": (
                f"{result.get('reason', '')} Safety override: typed occlusion "
                "failure should trigger localized occluder editing before "
                "another generic regeneration."
            ).strip(),
            "preconditions": {
                **_dict_value(result.get("preconditions")),
                **_dict_value(occlusion_plan.get("preconditions")),
                "planner_override": "typed_occlusion_object_insertion",
            },
        }
        return _apply_tool_availability(override, enabled_tools)
    recolor_target = _find_recolor_target_from_errors(errors, constraints)
    if action == "relation_repair" and recolor_target:
        object_name, target_color, source_color = recolor_target
        override = {
            **result,
            "primary_action": "recolor",
            "tool_sequence": ["recolor", "relation_repair"],
            "repairable": True,
            "fallback_from": "relation_repair",
            "target_object": object_name,
            "target_attribute": "color",
            "target_color": target_color,
            "source_color": source_color,
            "reason": (
                f"{result.get('reason', '')} Safety override: a user color "
                f"binding for {object_name} is still wrong, so recolor must run "
                "before relation repair."
            ).strip(),
        }
        return _apply_tool_availability(override, enabled_tools)
    if action != "object_insertion":
        return result
    target = _match_constraint_object(str(result.get("target_object") or ""), constraints)
    if not target:
        target = str(result.get("target_object") or "").strip()
    if not target:
        return result
    target_attribute = str(result.get("target_attribute") or "").strip().lower()
    if target_attribute not in {"count", "wrong_count"} and _target_marked_present(
        target,
        critique,
        constraints,
    ):
        override = _plan_for_present_target_insertion_override(
            result,
            target=target,
            critique=critique,
            constraints=constraints,
            enabled_tools=enabled_tools,
        )
        return _apply_tool_availability(override, enabled_tools)
    expected_count = _expected_count_for_target(target, constraints)
    if expected_count <= 1:
        return result
    if not _count_insertion_is_high_risk(target, constraints):
        return result
    if not _critique_mentions_target_missing_or_count(target, critique):
        return result
    preconditions = (
        dict(result.get("preconditions", {}))
        if isinstance(result.get("preconditions"), Mapping)
        else {}
    )
    override = {
        **result,
        "primary_action": "regenerate",
        "tool_sequence": ["regenerate"],
        "repairable": False,
        "fallback_from": "object_insertion",
        "target_object": target,
        "target_attribute": result.get("target_attribute") or "count_or_presence",
        "reason": (
            f"{result.get('reason', '')} Safety override: {target} has a "
            "plural/count requirement tied to a hard relation/action, so small "
            "object insertion is high risk. Regenerate with stronger layout."
        ).strip(),
        "preconditions": {
            **preconditions,
            "local_insertion_high_risk": True,
            "planner_override": "relation_bound_plural_insertion",
        },
    }
    return _apply_tool_availability(override, enabled_tools)


def _vlm_plan_targets_occluder_or_generic(
    plan: Mapping[str, Any],
    occlusion_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> bool:
    """Return whether a VLM object-insertion plan should inherit occlusion typing."""

    target = str(plan.get("target_object") or "").strip()
    if not target:
        return True
    occluder = str(occlusion_plan.get("target_object") or "").strip()
    if not occluder:
        return False
    target_match = _match_constraint_object(target, constraints) or target
    occluder_match = _match_constraint_object(occluder, constraints) or occluder
    if target_match == occluder_match:
        return True
    attr = str(plan.get("target_attribute") or "").strip().lower()
    reason = str(plan.get("reason") or "").strip().lower()
    return attr in {"occlusion", "presence", "missing_object"} and any(
        term in reason for term in _object_terms(occluder)
    )


def _plan_can_inherit_occlusion_typing(
    plan: Mapping[str, Any],
    occlusion_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> bool:
    action = _normalize_action(plan.get("primary_action"))
    if action == "regenerate":
        return True
    if action == "object_insertion":
        return _vlm_plan_targets_occluder_or_generic(plan, occlusion_plan, constraints)
    if action in {"none", "relation_repair"}:
        return _vlm_plan_targets_occluder_or_generic(plan, occlusion_plan, constraints)
    return False


def _occlusion_repair_plan(
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
    *,
    present_targets: set[str],
) -> dict[str, Any] | None:
    relation = _first_occlusion_relation(constraints)
    if not relation:
        return None
    occluder = str(relation.get("subject") or "").strip()
    target = str(relation.get("object") or "").strip()
    hidden_part = str(relation.get("hidden_part") or "").strip()
    visible_part = str(relation.get("visible_part") or "").strip()
    if not occluder or not target:
        return None
    haystack = " ".join(_error_text(item) for item in errors).lower()
    occlusion_failed = any(
        token in haystack
        for token in (
            "occlusion",
            "occluded",
            "hide",
            "hides",
            "hidden",
            "cover",
            "covers",
            "screen",
            hidden_part.lower(),
        )
        if token
    )
    occluder_missing = any(
        _missing_object_mentioned(haystack, term)
        for term in _object_terms(occluder)
    ) or any(term in haystack and "absent" in haystack for term in _object_terms(occluder))
    target_present = (
        (_match_constraint_object(target, constraints) or target) in present_targets
        or any(term in haystack and "visible" in haystack for term in _object_terms(target))
    )
    if not occlusion_failed and not occluder_missing:
        return None
    return {
        "primary_action": "object_insertion",
        "tool_sequence": ["object_insertion"],
        "repairable": True,
        "typed_route": "occlusion_object_insertion",
        "edit_timing": "early_edit",
        "evidence_policy": "historical_prompt_stats_and_typed_occlusion_failure",
        "target_object": occluder,
        "target_attribute": "occlusion",
        "target_region": _occlusion_target_region(hidden_part),
        "occlusion_spec": {
            "occluder": occluder,
            "target": target,
            "hidden_part": hidden_part,
            "visible_part": visible_part,
            "target_present": target_present,
        },
        "reason": (
            f"Typed occlusion failed: add/restore the {occluder} over the "
            f"{hidden_part or 'requested part'} of the {target} while preserving "
            f"{visible_part or 'the rest of the target'}."
        ),
        "preconditions": {
            "local_occlusion_repairable": True,
            "target_object_visible": False,
            "occlusion_target_visible": target_present,
            "planner_override_allowed": "occlusion_occluder_missing",
        },
    }


def _first_occlusion_relation(constraints: PromptConstraints) -> Mapping[str, str] | None:
    intent = constraints.intent_spec
    if intent is None:
        return None
    for relation in intent.interaction_relations:
        if str(relation.get("type") or "").strip().lower() == "occlusion":
            return relation
    return None


def _occlusion_target_region(hidden_part: str) -> str:
    text = hidden_part.lower()
    if any(token in text for token in ("lower", "bottom")):
        return "lower_half"
    if any(token in text for token in ("upper", "top")):
        return "upper_half"
    if "left" in text:
        return "left_half"
    if "right" in text:
        return "right_half"
    return "center"


def _plan_for_present_target_insertion_override(
    plan: Mapping[str, Any],
    *,
    target: str,
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
    enabled_tools: Mapping[str, bool],
) -> dict[str, Any]:
    errors = _collect_errors(critique)
    haystack = " ".join(_error_text(item) for item in errors).lower()
    recolor_target = _find_recolor_target_from_errors(errors, constraints)
    if recolor_target and enabled_tools.get("recolor", False):
        object_name, target_color, source_color = recolor_target
        sequence = ["recolor"]
        if _has_relation_failure(errors, haystack) and enabled_tools.get("relation_repair", False):
            sequence.append("relation_repair")
        return {
            **dict(plan),
            "primary_action": "recolor",
            "tool_sequence": sequence,
            "repairable": True,
            "fallback_from": "object_insertion",
            "target_object": object_name,
            "target_attribute": "color",
            "target_color": target_color,
            "source_color": source_color,
            "reason": (
                f"Safety override: {target} is already verified as visible, so "
                f"do not insert another {target}. Repair the visible color issue first."
            ),
            "preconditions": {
                **_dict_value(plan.get("preconditions")),
                "target_object_visible": True,
                "planner_override": "target_presence_verified",
            },
        }
    if _has_relation_failure(errors, haystack):
        if _relation_looks_unrepairable(haystack) or not enabled_tools.get(
            "relation_repair",
            False,
        ):
            return {
                **dict(plan),
                "primary_action": "regenerate",
                "tool_sequence": ["regenerate"],
                "repairable": False,
                "fallback_from": "object_insertion",
                "target_object": _relation_target(constraints),
                "target_attribute": "relation",
                "reason": (
                    f"Safety override: {target} is already visible, so object "
                    "insertion is wrong; regenerate because the relation lacks "
                    "clear local repair evidence."
                ),
                "preconditions": {
                    **_dict_value(plan.get("preconditions")),
                    "target_object_visible": True,
                    "planner_override": "target_presence_verified",
                    "relation_locally_repairable": False,
                },
            }
        return {
            **dict(plan),
            "primary_action": "relation_repair",
            "tool_sequence": ["relation_repair"],
            "repairable": True,
            "fallback_from": "object_insertion",
            "target_object": _relation_target(constraints),
            "target_attribute": "relation",
            "reason": (
                f"Safety override: {target} is already visible, so do not "
                "insert another object; repair the visible relation/contact."
            ),
            "preconditions": {
                **_dict_value(plan.get("preconditions")),
                "target_object_visible": True,
                "planner_override": "target_presence_verified",
                "relation_locally_repairable": True,
            },
        }
    return {
        **dict(plan),
        "primary_action": "none",
        "tool_sequence": [],
        "repairable": False,
        "fallback_from": "object_insertion",
        "reason": (
            f"Safety override: {target} is already verified as visible, so "
            "object insertion is not allowed without a count deficit."
        ),
        "preconditions": {
            **_dict_value(plan.get("preconditions")),
            "target_object_visible": True,
            "planner_override": "target_presence_verified",
        },
    }


def _expected_count_for_target(target: str, constraints: PromptConstraints) -> int:
    for question in generate_constraint_questions(constraints):
        if question.category != "count":
            continue
        source = question.source_constraint
        object_name = str(source.get("object") or "")
        if _match_constraint_object(target, constraints) == _match_constraint_object(object_name, constraints):
            count = _parse_count_value(source.get("count"))
            return count or 1
    return 1


def _targets_marked_present(
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
) -> set[str]:
    present: set[str] = set()
    check = critique.get("constraint_check")
    if isinstance(check, Mapping):
        summary = check.get("question_summary")
        if isinstance(summary, Mapping):
            for item in summary.get("passed_constraints", []) or []:
                text = str(item).strip().lower()
                if text.startswith(("existence:", "count:")):
                    target = _match_constraint_object(text.split(":", 1)[1], constraints)
                    if target:
                        present.add(target)
        for item in check.get("checks", []) or []:
            if not isinstance(item, Mapping) or item.get("passed") is not True:
                continue
            question_id = str(item.get("question_id") or "").strip().lower()
            category = str(item.get("category") or item.get("type") or "").strip().lower()
            target = _match_constraint_object(str(item.get("target") or ""), constraints)
            if not target and ":" in question_id:
                target = _match_constraint_object(question_id.split(":", 1)[1], constraints)
            if not target:
                continue
            if question_id.startswith("existence:") or category in {
                "entity_existence",
                "subject",
            }:
                present.add(target)
            elif question_id.startswith("count:") or category in {"count", "wrong_count"}:
                observed = _parse_count_value(item.get("observed"))
                if observed is None:
                    observed = _parse_count_from_text(_error_text(item))
                if observed is None or observed > 0:
                    present.add(target)
    return present


def _target_marked_present(
    target: str,
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
) -> bool:
    normalized = _match_constraint_object(target, constraints) or target
    return normalized in _targets_marked_present(critique, constraints)


def _critique_mentions_target_missing_or_count(
    target: str,
    critique: Mapping[str, Any],
) -> bool:
    target_terms = set(_object_terms(target))
    for error in _collect_errors(critique):
        text = _error_text(error).lower()
        if not any(term and term in text for term in target_terms):
            continue
        if any(token in text for token in ("missing", "not visible", "no ", "absent")):
            return True
        if str(error.get("question_id") or "").startswith("count:"):
            return True
        if str(error.get("category") or error.get("type") or "").lower() in {"count", "wrong_count"}:
            return True
    return False


def _constraint_object_names(constraints: PromptConstraints) -> list[str]:
    names = list(constraints.colors.keys())
    for subject in constraints.subjects:
        if subject not in names and not _is_modifier(subject):
            names.append(subject)
    return names


def _object_terms(object_name: str) -> list[str]:
    lowered = object_name.lower().strip()
    terms = [lowered]
    pieces = [part for part in re.split(r"[^a-z0-9-]+", lowered) if part]
    if pieces:
        terms.append(pieces[-1])
    for term in list(terms):
        if term.endswith("ies") and len(term) > 3:
            terms.append(f"{term[:-3]}y")
        elif term.endswith("s") and len(term) > 3:
            terms.append(term[:-1])
        elif len(term) > 2:
            terms.append(f"{term}s")
    return sorted(set(terms), key=len, reverse=True)


def _is_modifier(value: str) -> bool:
    return value.lower() in {"small", "large", "clear", "clearly"}


def _is_minor_part(value: str) -> bool:
    return value.lower() in {"handle", "hand", "claw", "stem"}


def _known_colors(constraints: PromptConstraints) -> list[str]:
    return sorted(
        set(constraints.colors.values())
        | {"red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange"}
    )


def _mentioned_wrong_color(text: str, target_color: str) -> str | None:
    for color in ("red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange"):
        if color != target_color and re.search(rf"\b{re.escape(color)}\b", text):
            return color
    return None


def _normalize_action(value: Any) -> str:
    action = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "insert_object": "object_insertion",
        "add_object": "object_insertion",
        "insertion": "object_insertion",
        "local_repair": "recolor",
        "color_repair": "recolor",
        "color": "recolor",
        "relation": "relation_repair",
        "action_repair": "relation_repair",
        "rerun": "regenerate",
        "retry_generation": "regenerate",
    }
    action = aliases.get(action, action)
    return action if action in REPAIR_ACTIONS else "none"


def _normalize_tool_sequence(value: Any, *, primary_action: str) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        items = []
    sequence: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            item = item.get("action") or item.get("tool")
        action = _normalize_action(item)
        if action != "none" and action not in sequence:
            sequence.append(action)
    if not sequence and primary_action not in {"none"}:
        sequence = [primary_action]
    return sequence


def _default_target_for_action(action: str, constraints: PromptConstraints) -> str:
    if action == "recolor" and constraints.colors:
        return next(iter(constraints.colors.keys()))
    if action == "object_insertion":
        names = _constraint_object_names(constraints)
        return names[0] if names else ""
    if action == "relation_repair":
        return _relation_target(constraints)
    return ""


def _clean_optional(value: Any) -> str:
    return str(value or "").strip()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "repairable", "pass"}
    return bool(value)


def _compact_feedback_for_request(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only current-image evidence needed by the VLM repair planner."""

    compact: dict[str, Any] = {
        "score": record.get("score"),
        "user_grounded": record.get("user_grounded"),
        "errors": _compact_errors(record.get("errors", [])),
        "revision_hint": _truncate_text(str(record.get("revision_hint") or ""), 1000),
    }
    for key in ("constraint_check", "evaluation", "relation_repair_verification"):
        value = record.get(key)
        if isinstance(value, Mapping):
            compact[key] = _compact_check(value)
    repair_selection = record.get("repairability_selection")
    if isinstance(repair_selection, Mapping):
        selected_check = repair_selection.get("selected_constraint_check")
        compact["repairability_selection"] = {
            "selected_index": repair_selection.get("selected_index"),
            "tier": repair_selection.get("tier"),
            "blocked": repair_selection.get("blocked"),
        }
        if isinstance(selected_check, Mapping):
            compact["repairability_selection"][
                "selected_constraint_check"
            ] = _compact_check(selected_check)
    return compact


def _compact_check(record: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for item in record.get("checks", []) or []:
        if not isinstance(item, Mapping):
            continue
        keep = item.get("passed") is False
        question_id = str(item.get("question_id") or "")
        if question_id.startswith(("existence:", "count:", "color:")):
            keep = True
        if not keep and len(checks) >= 12:
            continue
        checks.append(
            {
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "type": item.get("type"),
                "target": item.get("target"),
                "expected": item.get("expected"),
                "observed": item.get("observed"),
                "passed": item.get("passed"),
                "description": _truncate_text(
                    str(item.get("description") or item.get("evidence") or ""),
                    500,
                ),
            }
        )
    return {
        "passed": record.get("passed"),
        "failed": record.get("failed"),
        "score": record.get("score"),
        "source": record.get("source"),
        "question_summary": deepcopy(dict(record.get("question_summary", {})))
        if isinstance(record.get("question_summary"), Mapping)
        else {},
        "errors": _compact_errors(record.get("errors", [])),
        "checks": checks[:24],
    }


def _compact_errors(value: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in _normalize_error_records(value)[:12]:
        compact.append(
            {
                "type": item.get("type"),
                "category": item.get("category"),
                "question_id": item.get("question_id"),
                "target": item.get("target"),
                "expected": item.get("expected"),
                "observed": item.get("observed"),
                "prompt_span": item.get("prompt_span"),
                "evidence": _truncate_text(
                    str(item.get("evidence") or item.get("description") or ""),
                    700,
                ),
            }
        )
    return compact


def _dict_value(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


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


def _strip_runtime(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    for key in ("request", "raw_response"):
        result.pop(key, None)
    return result
