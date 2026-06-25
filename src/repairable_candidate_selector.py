"""Repairability-aware candidate selection for local edit tools."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .candidate_arbitration import (
    add_human_rule_evidence,
    constraint_items_from_constraints,
    summarize_constraint_check,
)
from .prompt_constraints import COLOR_WORDS, PromptConstraints


REPAIRABILITY_MIN_SCORE = {
    "recolor": 7.0,
    "object_insertion": 5.0,
    "relation_repair": 6.0,
}

REPAIRABILITY_SWITCH_MARGIN = 1.5


def select_repairable_candidate(
    *,
    arbitration: Mapping[str, Any] | None,
    current_index: int,
    critique: Mapping[str, Any],
    repair_plan: Mapping[str, Any] | None,
    constraints: PromptConstraints,
) -> dict[str, Any] | None:
    """Choose a better base image for the planned repair when one exists.

    Constraint arbitration answers "which candidate looks best now". Local repair
    needs a second decision: "which candidate is easiest and safest to repair".
    This function only uses already logged candidate constraint checks, so it
    does not add extra VLM/API calls.
    """

    if not isinstance(arbitration, Mapping) or not isinstance(repair_plan, Mapping):
        return None
    action = _normalize_action(repair_plan.get("primary_action"))
    if action not in REPAIRABILITY_MIN_SCORE:
        return None
    candidate_checks = arbitration.get("candidate_checks", [])
    if not isinstance(candidate_checks, Sequence):
        return None

    effective_repair_plan = _prefer_recolor_plan_when_available(
        repair_plan,
        constraints,
        candidate_checks,
    )
    effective_action = _normalize_action(effective_repair_plan.get("primary_action"))
    if effective_action not in REPAIRABILITY_MIN_SCORE:
        return None

    candidates: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(candidate_checks):
        if not isinstance(item, Mapping):
            continue
        candidate_item = item
        index = _coerce_int(item.get("index"), default=fallback_index)
        if index is not None and int(index) == int(current_index):
            candidate_item = _with_current_feedback(item, critique)
        profile = build_repairability_profile(
            candidate_item,
            fallback_index=fallback_index,
            repair_plan=effective_repair_plan,
            constraints=constraints,
        )
        candidates.append(profile)
    if not candidates:
        return None

    current = _find_profile(candidates, current_index)
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(_tier_rank(item)),
            float(item["repairability"]["score"]),
            float(item["constraint_summary"].get("score", 0.0) or 0.0),
            float(item["constraint_summary"].get("passed_checks", 0) or 0),
            -float(item["index"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    best_score = float(best["repairability"]["score"])
    current_score = (
        float(current["repairability"]["score"]) if current is not None else float("-inf")
    )
    if int(best["index"]) == int(current_index):
        return None
    if current is not None and _tier_rank(best) < _tier_rank(current):
        return {
            "type": "repairability_switch_blocked",
            "primary_action": effective_action,
            "target_object": _target_object(effective_repair_plan, constraints),
            "target_attribute": _target_attribute(effective_repair_plan, effective_action),
            "previous_index": int(current_index),
            "previous_image": current.get("image_path"),
            "selected_index": int(current_index),
            "selected_image": current.get("image_path"),
            "selected_prompt": current.get("prompt"),
            "blocked": True,
            "reason": (
                "blocked repair-base switch because the best repairability candidate "
                "has a worse human-rule constraint tier than the current selection"
            ),
            "ranking": [
                {
                    "index": int(item["index"]),
                    "image_path": item["image_path"],
                    "constraint_summary": deepcopy(dict(item["constraint_summary"])),
                    "repairability": deepcopy(dict(item["repairability"])),
                    "tier_rank": _tier_rank(item),
                }
                for item in ranked
            ],
        }
    if best_score < REPAIRABILITY_MIN_SCORE[effective_action]:
        return None
    if current is not None and best_score < current_score + REPAIRABILITY_SWITCH_MARGIN:
        return None

    return {
        "type": "repairability_aware_candidate_selection",
        "primary_action": effective_action,
        "target_object": _target_object(effective_repair_plan, constraints),
        "target_attribute": _target_attribute(effective_repair_plan, effective_action),
        "repair_plan_override": (
            deepcopy(dict(effective_repair_plan)) if effective_action != action else None
        ),
        "previous_index": int(current_index),
        "previous_image": current.get("image_path") if current else arbitration.get("selected_image"),
        "selected_index": int(best["index"]),
        "selected_image": best["image_path"],
        "selected_prompt": best["prompt"],
        "reason": _selection_reason(best, current, effective_action),
        "selected_constraint_check": deepcopy(best.get("constraint_check", {})),
        "ranking": [
            {
                "index": int(item["index"]),
                "image_path": item["image_path"],
                "constraint_summary": deepcopy(dict(item["constraint_summary"])),
                "repairability": deepcopy(dict(item["repairability"])),
            }
            for item in ranked
        ],
    }


def _prefer_recolor_plan_when_available(
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
    candidate_checks: Sequence[Any],
) -> Mapping[str, Any]:
    action = _normalize_action(repair_plan.get("primary_action"))
    if action != "object_insertion":
        return repair_plan
    if not candidate_checks or not constraints.colors:
        return repair_plan
    target_object = _target_object(repair_plan, constraints)
    if not target_object or target_object not in constraints.colors:
        return repair_plan
    recolor_plan = {
        **dict(repair_plan),
        "primary_action": "recolor",
        "tool_sequence": ["recolor"],
        "target_object": target_object,
        "target_attribute": "color",
        "target_color": constraints.colors[target_object],
        "override_reason": (
            "prefer recolor because at least one candidate shows the target object "
            "with a color mismatch, which is safer than object insertion"
        ),
    }
    for fallback_index, item in enumerate(candidate_checks):
        if not isinstance(item, Mapping):
            continue
        profile = build_repairability_profile(
            item,
            fallback_index=fallback_index,
            repair_plan=recolor_plan,
            constraints=constraints,
        )
        if profile["repairability"]["score"] >= REPAIRABILITY_MIN_SCORE["recolor"]:
            return recolor_plan
    return repair_plan


def build_repairability_profile(
    candidate: Mapping[str, Any],
    *,
    fallback_index: int,
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    """Summarize whether one candidate is a good base for the planned tool."""

    index = _coerce_int(candidate.get("index"), default=fallback_index)
    check = candidate.get("constraint_check", candidate)
    if not isinstance(check, Mapping):
        check = {}
    check = deepcopy(dict(check))
    prompt = str(candidate.get("prompt") or check.get("prompt") or "")
    image_path = str(candidate.get("image_path") or check.get("image_path") or "")
    summary = add_human_rule_evidence(
        summarize_constraint_check(check),
        check,
        constraint_items_from_constraints(constraints),
    )
    facts = _candidate_facts(check, constraints)
    action = _normalize_action(repair_plan.get("primary_action"))
    repairability = _score_repairability(
        facts,
        summary,
        action=action,
        repair_plan=repair_plan,
        constraints=constraints,
    )
    return {
        "index": int(index),
        "image_path": image_path,
        "prompt": prompt,
        "constraint_check": check,
        "constraint_summary": summary,
        "facts": facts,
        "repairability": repairability,
    }


def _with_current_feedback(
    candidate: Mapping[str, Any],
    critique: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(critique, Mapping):
        return deepcopy(dict(candidate))
    result = deepcopy(dict(candidate))
    check = result.get("constraint_check", result)
    if not isinstance(check, Mapping):
        check = {}
    merged = deepcopy(dict(check))
    checks = _list_records(merged.get("checks"))
    errors = _list_records(merged.get("errors"))
    for record in _feedback_records(critique, key="checks"):
        checks.append(record)
    for record in _feedback_records(critique, key="errors"):
        errors.append(record)
    if checks:
        merged["checks"] = checks
    if errors:
        merged["errors"] = errors
        merged["passed"] = False
    if any(item.get("passed") is False for item in checks):
        merged["passed"] = False
    result["constraint_check"] = merged
    return result


def _feedback_records(feedback: Mapping[str, Any], *, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_list_records(feedback.get(key)))
    for nested_key in ("constraint_check", "evaluation", "relation_repair_verification"):
        nested = feedback.get(nested_key)
        if isinstance(nested, Mapping):
            records.extend(_list_records(nested.get(key)))
    if key == "errors":
        for item in records:
            item.setdefault("passed", False)
    return records


def _list_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _tier_rank(candidate: Mapping[str, Any]) -> int:
    summary = candidate.get("constraint_summary", {})
    if not isinstance(summary, Mapping):
        return 0
    tier = str(summary.get("human_rule_tier") or "").strip()
    ranks = {
        "satisfies_user_constraints": 6,
        "repairable_action_or_relation_gap": 5,
        "uncertain_action_or_relation_evidence": 4,
        "minor_or_style_gap": 4,
        "uncertain_user_attribute_binding": 3,
        "reject_wrong_user_attribute": 2,
        "uncertain_required_object_evidence": 1,
        "reject_missing_or_wrong_count": 0,
    }
    return ranks.get(tier, 0)


def _score_repairability(
    facts: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    action: str,
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    target_object = _target_object(repair_plan, constraints)
    target_attribute = _target_attribute(repair_plan, action)
    reasons: list[str] = []
    penalties: list[str] = []
    score = 0.0

    if action == "recolor":
        target_color_failed = bool(facts["target_color_failures"].get(target_object))
        target_present = _target_present(facts, target_object)
        relation_clean = bool(facts["relation_action_passed"]) and not bool(
            facts["relation_action_failed"]
        )
        other_colors_ok = _other_user_colors_ok(facts, constraints, target_object)
        if target_color_failed:
            score += 6.0
            reasons.append(f"{target_object} has a visible color mismatch")
        else:
            score -= 8.0
            penalties.append(f"{target_object} has no explicit color mismatch to recolor")
        if target_present:
            score += 3.0
            reasons.append(f"{target_object} appears present")
        else:
            score -= 8.0
            penalties.append(f"{target_object} appears missing")
        if relation_clean:
            score += 3.0
            reasons.append("action/relation checks are already stable")
        elif facts["relation_action_failed"]:
            score -= 4.0
            penalties.append("action/relation also fails, so recolor alone is risky")
        elif _has_relation_action_requirement(constraints):
            score -= 2.5
            penalties.append("required action/relation has no explicit pass evidence")
        if other_colors_ok:
            score += 1.0
            reasons.append("other user color bindings look preserved")
        else:
            score -= 3.0
            penalties.append("another user color binding failed")
        if facts["major_missing_objects"]:
            score -= 5.0 * len(facts["major_missing_objects"])
            penalties.append("required object is missing")

    elif action == "object_insertion":
        missing = target_object in facts["missing_objects"]
        other_visible = _other_required_objects_visible(facts, constraints, target_object)
        if missing:
            score += 5.0
            reasons.append(f"{target_object} is the missing local target")
        else:
            penalties.append(f"{target_object} is not clearly missing")
        if other_visible:
            score += 3.0
            reasons.append("other required objects are visible anchors")
        else:
            score -= 4.0
            penalties.append("too many required objects are missing")
        if facts["relation_action_failed"]:
            score += 1.0
            reasons.append("relation failure is explained by the missing object")

    elif action == "relation_repair":
        target_present = _relation_objects_present(facts, constraints)
        colors_ok = _all_user_colors_ok(facts, constraints)
        relation_failed = bool(facts["relation_action_failed"])
        if relation_failed:
            score += 5.0
            reasons.append("relation/action is the main visible failure")
        else:
            penalties.append("relation/action is not clearly the failure")
        if target_present:
            score += 3.0
            reasons.append("related objects and parts appear present")
        else:
            score -= 6.0
            penalties.append("related objects are missing, so local contact repair is risky")
        if colors_ok:
            score += 2.0
            reasons.append("user color bindings are already stable")
        else:
            score -= 3.0
            penalties.append("color binding also fails")

    score += 1.5 * float(summary.get("score", 0.0) or 0.0)
    score += 0.15 * float(summary.get("passed_checks", 0) or 0)
    score -= 0.35 * float(summary.get("failed_checks", 0) or 0)
    score = round(score, 4)
    return {
        "action": action,
        "target_object": target_object,
        "target_attribute": target_attribute,
        "score": score,
        "repairable": score >= REPAIRABILITY_MIN_SCORE.get(action, 999.0),
        "reasons": reasons,
        "penalties": penalties,
        "target_color_failures": deepcopy(dict(facts["target_color_failures"])),
        "missing_objects": list(facts["missing_objects"]),
        "major_missing_objects": list(facts["major_missing_objects"]),
        "relation_action_passed": bool(facts["relation_action_passed"]),
        "relation_action_failed": bool(facts["relation_action_failed"]),
    }


def _candidate_facts(
    check: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    records = _constraint_records(check)
    object_names = _constraint_object_names(constraints)
    target_color_failures: dict[str, bool] = {name: False for name in constraints.colors}
    color_failures: dict[str, bool] = {name: False for name in constraints.colors}
    color_passes: dict[str, bool] = {name: False for name in constraints.colors}
    object_passes: dict[str, bool] = {name: False for name in object_names}
    missing_objects: set[str] = set()
    major_missing_objects: set[str] = set()
    relation_action_passed = False
    relation_action_failed = False

    for record in records:
        text = _record_text(record)
        passed = record.get("passed")
        for object_name, expected_color in constraints.colors.items():
            if _mentions_object(text, object_name):
                if _is_color_record(record, text) and passed is True:
                    color_passes[object_name] = True
                    object_passes[object_name] = True
                if _is_target_color_failure(record, text, object_name, expected_color):
                    color_failures[object_name] = True
                    target_color_failures[object_name] = True
                    object_passes[object_name] = True
        for object_name in object_names:
            if _mentions_object(text, object_name):
                if passed is True:
                    object_passes[object_name] = True
                if _is_missing_object_failure(record, text, object_name):
                    missing_objects.add(object_name)
                    if not _is_minor_part(object_name):
                        major_missing_objects.add(object_name)
        if _is_relation_action_record(record, text):
            if passed is True:
                relation_action_passed = True
            elif passed is False or _is_relation_action_failure_text(text):
                relation_action_failed = True
        elif _is_relation_action_failure_text(text):
            relation_action_failed = True

    return {
        "target_color_failures": target_color_failures,
        "color_failures": color_failures,
        "color_passes": color_passes,
        "object_passes": object_passes,
        "missing_objects": sorted(missing_objects),
        "major_missing_objects": sorted(major_missing_objects),
        "relation_action_passed": relation_action_passed,
        "relation_action_failed": relation_action_failed,
    }


def _constraint_records(check: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    checks = check.get("checks", [])
    if isinstance(checks, Mapping):
        checks = [checks]
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        records.extend(deepcopy(dict(item)) for item in checks if isinstance(item, Mapping))
    errors = check.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for item in errors:
            if isinstance(item, Mapping):
                record = deepcopy(dict(item))
                record.setdefault("passed", False)
                records.append(record)
    return records


def _target_object(
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> str:
    target = _clean_name(repair_plan.get("target_object"))
    if target:
        return target
    if constraints.colors:
        return next(iter(constraints.colors))
    if constraints.subjects:
        return _clean_name(constraints.subjects[0])
    return ""


def _target_attribute(repair_plan: Mapping[str, Any], action: str) -> str:
    attribute = _clean_name(repair_plan.get("target_attribute"))
    if attribute:
        return attribute
    if action == "recolor":
        return "color"
    if action == "object_insertion":
        return "presence"
    if action == "relation_repair":
        return "relation"
    return ""


def _target_present(facts: Mapping[str, Any], target_object: str) -> bool:
    if not target_object:
        return False
    if target_object in facts["missing_objects"]:
        return False
    if facts["target_color_failures"].get(target_object):
        return True
    return bool(facts["object_passes"].get(target_object))


def _relation_objects_present(
    facts: Mapping[str, Any],
    constraints: PromptConstraints,
) -> bool:
    required = [
        name
        for name in _constraint_object_names(constraints)
        if not _is_minor_part(name)
    ]
    if not required:
        return True
    return all(name not in facts["missing_objects"] for name in required)


def _has_relation_action_requirement(constraints: PromptConstraints) -> bool:
    return bool(constraints.actions or constraints.relations)


def _other_required_objects_visible(
    facts: Mapping[str, Any],
    constraints: PromptConstraints,
    target_object: str,
) -> bool:
    required = [
        name
        for name in _constraint_object_names(constraints)
        if name != target_object and not _is_minor_part(name)
    ]
    if not required:
        return True
    return any(facts["object_passes"].get(name) for name in required) and not any(
        name in facts["major_missing_objects"] for name in required
    )


def _other_user_colors_ok(
    facts: Mapping[str, Any],
    constraints: PromptConstraints,
    target_object: str,
) -> bool:
    for object_name in constraints.colors:
        if object_name == target_object:
            continue
        if facts["color_failures"].get(object_name):
            return False
    return True


def _all_user_colors_ok(
    facts: Mapping[str, Any],
    constraints: PromptConstraints,
) -> bool:
    return not any(facts["color_failures"].values())


def _find_profile(candidates: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any] | None:
    for item in candidates:
        if int(item.get("index", -1)) == int(index):
            return item
    return None


def _selection_reason(
    best: Mapping[str, Any],
    current: Mapping[str, Any] | None,
    action: str,
) -> str:
    reasons = best["repairability"].get("reasons", [])
    reason = "; ".join(str(item) for item in reasons[:3]) or "higher repairability score"
    if current is None:
        return f"selected candidate {best['index']} for {action}: {reason}"
    return (
        f"selected candidate {best['index']} for {action}: {reason}; "
        f"repairability {best['repairability']['score']} > "
        f"{current['repairability']['score']} for current candidate {current['index']}"
    )


def _constraint_object_names(constraints: PromptConstraints) -> list[str]:
    names = [_clean_name(name) for name in constraints.colors.keys()]
    for subject in constraints.subjects:
        name = _clean_name(subject)
        if name and name not in names and not _is_modifier(name):
            names.append(name)
    return names


def _object_terms(object_name: str) -> list[str]:
    lowered = _clean_name(object_name)
    terms = [lowered]
    pieces = [part for part in re.split(r"[^a-z0-9]+", lowered) if part]
    if pieces:
        terms.append(pieces[-1])
    return sorted(set(term for term in terms if term), key=len, reverse=True)


def _mentions_object(text: str, object_name: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in _object_terms(object_name))


def _is_color_record(record: Mapping[str, Any], text: str) -> bool:
    check_type = _normalize_token(record.get("type"))
    return "color" in check_type or "attribute" in check_type or any(
        re.search(rf"\b{re.escape(color)}\b", text) for color in COLOR_WORDS
    )


def _is_target_color_failure(
    record: Mapping[str, Any],
    text: str,
    object_name: str,
    expected_color: str,
) -> bool:
    if record.get("passed") is True:
        return False
    if not _mentions_object(text, object_name):
        return False
    if _is_missing_object_failure(record, text, object_name):
        return False
    if expected_color and (
        f"instead of {expected_color}" in text
        or f"rather than {expected_color}" in text
        or f"not {expected_color}" in text
        or f"should be {expected_color}" in text
        or f"expected {expected_color}" in text
    ):
        return True
    if _is_color_record(record, text):
        return True
    other_colors = COLOR_WORDS - {expected_color}
    return any(
        re.search(rf"\b{re.escape(color)}\b.{0,80}\b{re.escape(term)}\b", text)
        or re.search(rf"\b{re.escape(term)}\b.{0,80}\b{re.escape(color)}\b", text)
        for color in other_colors
        for term in _object_terms(object_name)
    )


def _is_missing_object_failure(
    record: Mapping[str, Any],
    text: str,
    object_name: str,
) -> bool:
    if record.get("passed") is True:
        return False
    if not _mentions_object(text, object_name):
        return False
    error_type = _normalize_token(record.get("type"))
    if "missing_object" in error_type:
        return True
    for term in _object_terms(object_name):
        escaped = re.escape(term)
        patterns = (
            rf"\bmissing\s+(?:[a-z0-9-]+\s+){{0,3}}{escaped}\b",
            rf"\bno\s+(?:[a-z0-9-]+\s+){{0,3}}{escaped}\b",
            rf"\b{escaped}\s+(?:is|are|appears|looks)?\s*(?:not\s+present|not\s+visible|not\s+shown|absent|missing)\b",
            rf"\bdoes\s+not\s+show\s+(?:[a-z0-9-]+\s+){{0,3}}{escaped}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return True
    return False


def _is_relation_action_record(record: Mapping[str, Any], text: str) -> bool:
    check_type = _normalize_token(record.get("type"))
    target = _normalize_token(record.get("target"))
    if any(token in check_type for token in ("relation", "action", "contact")):
        return True
    if any(token in target for token in ("grip", "holding", "contact")):
        return True
    return _is_relation_action_failure_text(text)


def _is_relation_action_failure_text(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "not gripping",
            "not clearly gripping",
            "not shown gripping",
            "not holding",
            "not touching",
            "no relation",
            "detached",
            "not connected",
            "holding the umbrella with its arms",
            "rather than gripping",
        )
    )


def _record_text(record: Mapping[str, Any]) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in (
            "type",
            "target",
            "expected",
            "observed",
            "description",
            "evidence",
            "prompt_span",
            "reason",
            "message",
        )
    ).lower()


def _normalize_action(value: Any) -> str:
    action = _normalize_token(value)
    if action in {"local_repair", "color_repair"}:
        return "recolor"
    return action


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _clean_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_minor_part(value: str) -> bool:
    return _clean_name(value) in {"handle", "hand", "claw", "grip", "gripping"}


def _is_modifier(value: str) -> bool:
    return _clean_name(value) in {"small", "large", "clear", "clearly"}
