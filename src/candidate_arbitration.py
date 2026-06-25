"""Constraint-aware image candidate arbitration for M6.8/M6.21.1."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence


HARD_CONSTRAINT_TYPES = {
    "color",
    "subject",
    "action",
    "relation",
    "wrong_attribute",
    "wrong_relation",
    "missing_object",
    "wrong_count",
}


def arbitrate_image_candidates(
    *,
    image_paths: Sequence[str],
    prompts: Sequence[str],
    selection: Mapping[str, Any],
    reward_ranking: Mapping[str, Any] | None = None,
    candidate_checks: Sequence[Mapping[str, Any]] | None = None,
    constraints: Mapping[str, Any] | Any | None = None,
    reward_epsilon: float = 0.02,
) -> dict[str, Any]:
    """Select the best image with user constraints before reward preference.

    M6.21.1 treats the VLM/reward outputs as visual evidence, not as the final
    judge. The rule priority is:

    1. reject missing/count failures;
    2. reject wrong or unsupported user attribute bindings, especially color;
    3. prefer repairable action/relation gaps over wrong attributes;
    4. use reward/visual selection only as same-tier tie-breakers.
    """

    image_paths = [str(path) for path in image_paths]
    prompts = [str(prompt) for prompt in prompts]
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if len(prompts) != len(image_paths):
        raise ValueError("prompts and image_paths must have the same length")

    selection_index = _bounded_index(selection.get("selected_index", 0), len(image_paths))
    reward_scores = _reward_scores_by_index(reward_ranking)
    checks = _checks_by_index(candidate_checks)
    constraint_items = constraint_items_from_constraints(constraints)

    candidates: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        check = checks.get(index)
        summary = summarize_constraint_check(check)
        summary = add_human_rule_evidence(summary, check, constraint_items)
        reward_score = reward_scores.get(index)
        trace = _candidate_trace(summary, reward_score, visual_selected=index == selection_index)
        candidates.append(
            {
                "index": index,
                "image_path": image_path,
                "prompt": prompts[index],
                "constraint_summary": summary,
                "human_rule_trace": trace,
                "reward_score": reward_score,
                "visual_selected": index == selection_index,
                "sort_key": _candidate_sort_key(
                    summary,
                    reward_score,
                    visual_selected=index == selection_index,
                    original_index=index,
                    reward_epsilon=reward_epsilon,
                ),
            }
        )

    ranked = sorted(candidates, key=lambda item: item["sort_key"], reverse=True)
    selected = ranked[0]
    decision = {
        "type": "constraint_aware_arbitration",
        "selected_index": selected["index"],
        "selected_image": selected["image_path"],
        "selected_prompt": selected["prompt"],
        "visual_selected_index": selection_index,
        "reward_selected_index": _reward_selected_index(reward_ranking, len(image_paths)),
        "used_candidate_constraints": bool(candidate_checks),
        "reward_epsilon": float(reward_epsilon),
        "selection_policy": "m6211_rule_grounded_human_like_selector",
        "ranking": [
            {
                "index": item["index"],
                "image_path": item["image_path"],
                "constraint_summary": deepcopy(item["constraint_summary"]),
                "human_rule_trace": deepcopy(item["human_rule_trace"]),
                "reward_score": item["reward_score"],
                "visual_selected": item["visual_selected"],
            }
            for item in ranked
        ],
        "selection_trace": _selection_trace(ranked, selected),
    }
    if selected["index"] != selection_index:
        decision["overrode_visual_selection"] = True
    if (
        decision["reward_selected_index"] is not None
        and selected["index"] != decision["reward_selected_index"]
    ):
        decision["overrode_reward_selection"] = True
    return decision


def summarize_constraint_check(
    check: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convert a VLM constraint check into a small sortable summary."""

    if not isinstance(check, Mapping):
        return {
            "available": False,
            "passed": None,
            "score": 0.0,
            "passed_checks": 0,
            "failed_checks": 0,
            "hard_failures": 0,
            "soft_failures": 0,
            "failed_types": [],
            "failed_targets": [],
            "error_count": 0,
            "failed": False,
        }

    checks = check.get("checks", [])
    if isinstance(checks, Mapping):
        checks = [checks]
    if not isinstance(checks, list):
        checks = []

    errors = check.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    if not isinstance(errors, list):
        errors = []

    passed_checks = 0
    failed_checks = 0
    hard_failures = 0
    soft_failures = 0
    failed_types: list[str] = []
    failed_targets: list[str] = []
    fatal_failures = 0
    major_failures = 0
    repairable_failures = 0
    minor_failures = 0

    for item in checks:
        if not isinstance(item, Mapping):
            continue
        check_type = _normalize_text(item.get("type"))
        target = _normalize_text(item.get("target"))
        passed = bool(item.get("passed", False))
        if passed:
            passed_checks += 1
            continue
        failed_checks += 1
        if check_type:
            failed_types.append(check_type)
        if target:
            failed_targets.append(target)
        severity = _failure_severity(check_type, target)
        if severity == "fatal":
            fatal_failures += 1
        elif severity == "major":
            major_failures += 1
        elif severity == "repairable":
            repairable_failures += 1
        else:
            minor_failures += 1
        if _is_hard_constraint_failure(check_type, target):
            hard_failures += 1
        else:
            soft_failures += 1

    for item in errors:
        if not isinstance(item, Mapping):
            continue
        error_type = _normalize_text(item.get("type"))
        prompt_span = _normalize_text(item.get("prompt_span"))
        if error_type:
            failed_types.append(error_type)
        if prompt_span:
            failed_targets.append(prompt_span)
        severity = _failure_severity(error_type, prompt_span)
        if severity == "fatal":
            fatal_failures += 1
        elif severity == "major":
            major_failures += 1
        elif severity == "repairable":
            repairable_failures += 1
        else:
            minor_failures += 1
        if _is_hard_constraint_failure(error_type, prompt_span):
            hard_failures += 1
        else:
            soft_failures += 1

    score = _coerce_float(
        check.get("constraint_score", check.get("score")),
        default=0.0,
    )
    passed_value = check.get("passed")
    passed = bool(passed_value) if passed_value is not None else None
    failed = bool(check.get("failed", False))
    return {
        "available": True,
        "passed": passed,
        "score": score,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "fatal_failures": fatal_failures,
        "major_failures": major_failures,
        "repairable_failures": repairable_failures,
        "minor_failures": minor_failures,
        "failed_types": _dedupe(failed_types),
        "failed_targets": _dedupe(failed_targets),
        "error_count": len(errors),
        "failed": failed,
    }


def _candidate_sort_key(
    summary: Mapping[str, Any],
    reward_score: float | None,
    *,
    visual_selected: bool,
    original_index: int,
    reward_epsilon: float,
) -> tuple[float, ...]:
    passed_value = summary.get("passed")
    passed_rank = 1.0 if passed_value is True else 0.0
    if passed_value is None:
        passed_rank = 0.5
    reward_bucket = 0.0
    if reward_score is not None:
        epsilon = max(float(reward_epsilon), 1e-9)
        reward_bucket = int(float(reward_score) / epsilon)
    return (
        -float(summary.get("fatal_failures", 0) or 0),
        -float(summary.get("major_failures", 0) or 0),
        -float(summary.get("fatal_evidence_gaps", 0) or 0),
        -float(summary.get("major_evidence_gaps", 0) or 0),
        float(summary.get("coverage_ratio", 0.0) or 0.0),
        -float(summary.get("repairable_failures", 0) or 0),
        -float(summary.get("repairable_evidence_gaps", 0) or 0),
        -float(summary.get("minor_failures", 0) or 0),
        -float(summary.get("minor_evidence_gaps", 0) or 0),
        -float(summary.get("hard_failures", 0) or 0),
        -float(summary.get("failed_checks", 0) or 0),
        passed_rank,
        float(summary.get("passed_checks", 0) or 0),
        float(summary.get("score", 0.0) or 0.0),
        reward_bucket,
        float(reward_score if reward_score is not None else -1.0),
        1.0 if visual_selected else 0.0,
        -float(original_index),
    )


def _reward_scores_by_index(
    reward_ranking: Mapping[str, Any] | None,
) -> dict[int, float]:
    if not isinstance(reward_ranking, Mapping):
        return {}
    scores = reward_ranking.get("scores", [])
    if not isinstance(scores, list):
        return {}
    result: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        index = _coerce_int(item.get("index"), default=None)
        if index is None:
            continue
        result[index] = _coerce_float(item.get("score"), default=0.0)
    return result


def _reward_selected_index(
    reward_ranking: Mapping[str, Any] | None,
    num_images: int,
) -> int | None:
    if not isinstance(reward_ranking, Mapping):
        return None
    if "selected_index" not in reward_ranking:
        return None
    return _bounded_index(reward_ranking.get("selected_index", 0), num_images)


def _checks_by_index(
    candidate_checks: Sequence[Mapping[str, Any]] | None,
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    if not candidate_checks:
        return result
    for fallback_index, item in enumerate(candidate_checks):
        if not isinstance(item, Mapping):
            continue
        index = _coerce_int(item.get("index"), default=fallback_index)
        if index is None:
            continue
        check = item.get("constraint_check", item)
        if isinstance(check, Mapping):
            result[index] = check
    return result


def _is_hard_constraint_failure(check_type: str, target: str) -> bool:
    if check_type in HARD_CONSTRAINT_TYPES:
        return True
    text = f"{check_type} {target}".lower()
    return any(
        token in text
        for token in (
            "color",
            "attribute",
            "subject",
            "object",
            "relation",
            "action",
            "grip",
            "hold",
            "handle",
            "count",
        )
    )


def _failure_severity(check_type: str, target: str) -> str:
    text = f"{check_type} {target}".lower().replace("-", "_")
    if _is_part_like_target(target) and any(
        token in text for token in ("missing_object", "entity_existence", "subject")
    ):
        return "repairable"
    if any(
        token in text
        for token in (
            "missing_object",
            "entity_existence",
            "wrong_count",
            "count",
            "subject",
        )
    ):
        return "fatal"
    if any(
        token in text
        for token in (
            "wrong_attribute",
            "attribute_binding",
            "color_binding",
            "color",
            "attribute",
        )
    ):
        return "major"
    if any(
        token in text
        for token in (
            "wrong_relation",
            "relation",
            "action",
            "contact",
            "grip",
            "hold",
            "handle",
            "wearing",
            "sitting",
            "standing",
            "riding",
            "near",
            "left_of",
            "right_of",
        )
    ):
        return "repairable"
    return "minor"


def add_human_rule_evidence(
    summary: Mapping[str, Any],
    check: Mapping[str, Any] | None,
    constraint_items: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Attach human-rule coverage/tier evidence to a constraint summary."""

    enriched = deepcopy(dict(summary))
    items = [dict(item) for item in constraint_items if item.get("id")]
    covered: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for item in items:
        if _constraint_item_covered(item, check):
            covered.append(item)
        else:
            missing.append(item)

    gap_counts = {"fatal": 0, "major": 0, "repairable": 0, "minor": 0}
    for item in missing:
        gap_counts[item.get("severity", "minor")] = (
            gap_counts.get(item.get("severity", "minor"), 0) + 1
        )

    total = len(items)
    coverage_ratio = (len(covered) / total) if total else 0.0
    enriched.update(
        {
            "constraint_items": items,
            "covered_constraints": covered,
            "missing_evidence": missing,
            "coverage_ratio": round(coverage_ratio, 6),
            "fatal_evidence_gaps": gap_counts["fatal"],
            "major_evidence_gaps": gap_counts["major"],
            "repairable_evidence_gaps": gap_counts["repairable"],
            "minor_evidence_gaps": gap_counts["minor"],
            "human_rule_tier": _human_rule_tier(enriched, gap_counts),
        }
    )
    return enriched


def _human_rule_tier(summary: Mapping[str, Any], gap_counts: Mapping[str, int]) -> str:
    if int(summary.get("fatal_failures", 0) or 0) > 0:
        return "reject_missing_or_wrong_count"
    if int(summary.get("major_failures", 0) or 0) > 0:
        return "reject_wrong_user_attribute"
    if int(gap_counts.get("fatal", 0) or 0) > 0:
        return "uncertain_required_object_evidence"
    if int(gap_counts.get("major", 0) or 0) > 0:
        return "uncertain_user_attribute_binding"
    if int(summary.get("repairable_failures", 0) or 0) > 0:
        return "repairable_action_or_relation_gap"
    if int(gap_counts.get("repairable", 0) or 0) > 0:
        return "uncertain_action_or_relation_evidence"
    if int(summary.get("minor_failures", 0) or 0) > 0 or int(
        gap_counts.get("minor", 0) or 0
    ) > 0:
        return "minor_or_style_gap"
    return "satisfies_user_constraints"


def _candidate_trace(
    summary: Mapping[str, Any],
    reward_score: float | None,
    *,
    visual_selected: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    tier = str(summary.get("human_rule_tier") or "unknown")
    reasons.append(f"tier={tier}")
    for key in (
        "fatal_failures",
        "major_failures",
        "fatal_evidence_gaps",
        "major_evidence_gaps",
        "repairable_failures",
        "repairable_evidence_gaps",
    ):
        value = int(summary.get(key, 0) or 0)
        if value:
            reasons.append(f"{key}={value}")
    reasons.append(f"coverage={float(summary.get('coverage_ratio', 0.0) or 0.0):.2f}")
    if reward_score is not None:
        reasons.append(f"reward={float(reward_score):.3f}")
    if visual_selected:
        reasons.append("vlm_selected=true")
    return {
        "tier": tier,
        "coverage_ratio": summary.get("coverage_ratio", 0.0),
        "covered_constraints": deepcopy(summary.get("covered_constraints", [])),
        "missing_evidence": deepcopy(summary.get("missing_evidence", [])),
        "fatal_failures": summary.get("fatal_failures", 0),
        "major_failures": summary.get("major_failures", 0),
        "repairable_failures": summary.get("repairable_failures", 0),
        "minor_failures": summary.get("minor_failures", 0),
        "reward_score": reward_score,
        "visual_selected": visual_selected,
        "reason": "; ".join(reasons),
    }


def _selection_trace(
    ranked: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy": "VLM/reward are evidence providers; user-intent rules arbitrate.",
        "priority_order": [
            "required object/count failures",
            "user attribute/color binding failures or missing evidence",
            "repairable action/relation failures",
            "minor scene/style gaps",
            "constraint score",
            "reward score",
            "VLM visual selection",
        ],
        "selected_index": selected["index"],
        "selected_image": selected["image_path"],
        "selected_reason": selected["human_rule_trace"]["reason"],
        "candidates": [
            {
                "index": item["index"],
                "image_path": item["image_path"],
                **deepcopy(item["human_rule_trace"]),
            }
            for item in ranked
        ],
    }


def constraint_items_from_constraints(
    constraints: Mapping[str, Any] | Any | None,
) -> list[dict[str, str]]:
    """Build sortable user-intent constraint items from extracted constraints."""

    data = _constraint_data(constraints)
    items: list[dict[str, str]] = []
    for subject in _as_list(data.get("subjects")):
        target = _normalize_phrase(subject)
        if target:
            severity = "repairable" if _is_part_like_target(target) else "fatal"
            items.append(
                {
                    "id": f"subject:{target}",
                    "type": "subject",
                    "target": target,
                    "expected": "visible",
                    "severity": severity,
                }
            )
    for object_name, color in _as_mapping(data.get("colors")).items():
        target = _normalize_phrase(object_name)
        expected = _normalize_phrase(color)
        if target and expected:
            items.append(
                {
                    "id": f"color:{target}:{expected}",
                    "type": "color",
                    "target": target,
                    "expected": expected,
                    "severity": "major",
                }
            )
    for action in _as_list(data.get("actions")):
        phrase = _normalize_phrase(action)
        if phrase:
            items.append(
                {
                    "id": f"action:{phrase}",
                    "type": "action",
                    "target": phrase,
                    "expected": "yes",
                    "severity": "repairable",
                }
            )
    for relation in _as_list(data.get("relations")):
        phrase = _normalize_phrase(relation)
        if phrase:
            items.append(
                {
                    "id": f"relation:{phrase}",
                    "type": "relation",
                    "target": phrase,
                    "expected": "yes",
                    "severity": "repairable",
                }
            )
    for phrase in _as_list(data.get("protected_phrases")):
        target = _normalize_phrase(phrase)
        if target and not any(item["target"] == target for item in items):
            items.append(
                {
                    "id": f"protected:{target}",
                    "type": "protected_phrase",
                    "target": target,
                    "expected": "visible_or_styled",
                    "severity": "minor",
                }
            )
    return _dedupe_items(items)


def _constraint_data(constraints: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if constraints is None:
        return {}
    if isinstance(constraints, Mapping):
        return dict(constraints)
    to_dict = getattr(constraints, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return dict(data) if isinstance(data, Mapping) else {}
    result: dict[str, Any] = {}
    for key in ("subjects", "colors", "actions", "relations", "protected_phrases"):
        if hasattr(constraints, key):
            result[key] = getattr(constraints, key)
    return result


def _constraint_item_covered(
    item: Mapping[str, str],
    check: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(check, Mapping):
        return False
    if _structured_check_covers(item, check):
        return True
    for unit in _positive_evidence_units(check):
        if _evidence_unit_covers(item, unit):
            return True
    return False


def _structured_check_covers(item: Mapping[str, str], check: Mapping[str, Any]) -> bool:
    checks = check.get("checks", [])
    if isinstance(checks, Mapping):
        checks = [checks]
    if not isinstance(checks, list):
        return False
    for entry in checks:
        if not isinstance(entry, Mapping) or entry.get("passed") is not True:
            continue
        entry_type = _normalize_text(entry.get("type") or entry.get("category"))
        target = _normalize_phrase(entry.get("target") or entry.get("prompt_span"))
        expected = _normalize_phrase(entry.get("expected"))
        observed = _normalize_phrase(entry.get("observed"))
        qid = _normalize_phrase(entry.get("question_id"))
        if item.get("type") == "color":
            if _target_matches(item.get("target", ""), target) and (
                item.get("expected") in {expected, observed}
                or item.get("expected", "") in _normalize_phrase(entry)
            ):
                return True
            if (
                "color" in entry_type
                and _target_matches(item.get("target", ""), qid)
                and item.get("expected", "") in f"{expected} {observed}"
            ):
                return True
        elif item.get("type") == "subject":
            if _target_matches(item.get("target", ""), target) or _target_matches(
                item.get("target", ""), qid
            ):
                return True
        elif item.get("type") in {"action", "relation"}:
            text = _normalize_phrase(entry)
            if _phrase_terms_match(item.get("target", ""), text):
                return True
        elif item.get("type") == "protected_phrase":
            if _phrase_terms_match(item.get("target", ""), _normalize_phrase(entry)):
                return True
    return False


def _positive_evidence_units(check: Mapping[str, Any]) -> list[str]:
    units: list[str] = []
    strengths = check.get("strengths", [])
    if isinstance(strengths, (str, Mapping)):
        strengths = [strengths]
    if isinstance(strengths, list):
        for item in strengths:
            text = _evidence_text(item)
            if text:
                units.append(text)

    checks = check.get("checks", [])
    if isinstance(checks, Mapping):
        checks = [checks]
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, Mapping) and item.get("passed") is True:
                text = _evidence_text(item)
                if text:
                    units.append(text)
    return units


def _evidence_unit_covers(item: Mapping[str, str], unit: str) -> bool:
    text = _normalize_phrase(unit)
    item_type = item.get("type", "")
    target = item.get("target", "")
    expected = item.get("expected", "")
    if item_type == "color":
        return _target_matches(target, text) and bool(expected and expected in text)
    if item_type == "subject":
        return _target_matches(target, text)
    if item_type in {"action", "relation"}:
        return _phrase_terms_match(target, text)
    if item_type == "protected_phrase":
        return _phrase_terms_match(target, text)
    return False


def _evidence_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for key in (
            "description",
            "evidence",
            "reason",
            "message",
            "target",
            "expected",
            "observed",
            "question_id",
            "category",
            "type",
        ):
            item = value.get(key)
            if item is not None:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_evidence_text(item) for item in value)
    return str(value or "")


def _as_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item or "").strip()]
    return []


def _dedupe_items(items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for item in items:
        key = str(item.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _normalize_phrase(value: Any) -> str:
    if isinstance(value, Mapping):
        value = " ".join(str(item) for item in value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = " ".join(str(item) for item in value)
    text = str(value or "").lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _target_matches(target: str, text: str) -> bool:
    target = _normalize_phrase(target)
    text = _normalize_phrase(text)
    if not target or not text:
        return False
    terms = _object_terms(target)
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms if term)


def _phrase_terms_match(phrase: str, text: str) -> bool:
    phrase = _normalize_phrase(phrase)
    text = _normalize_phrase(text)
    if not phrase or not text:
        return False
    terms = [term for term in phrase.split() if len(term) > 2]
    if not terms:
        return phrase in text
    return all(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _object_terms(target: str) -> list[str]:
    target = _normalize_phrase(target)
    terms = [target]
    parts = [part for part in target.split() if part]
    if parts:
        terms.append(parts[-1])
    for term in list(terms):
        if term.endswith("ies") and len(term) > 3:
            terms.append(f"{term[:-3]}y")
        elif term.endswith("s") and len(term) > 3:
            terms.append(term[:-1])
        elif len(term) > 2:
            terms.append(f"{term}s")
    return _dedupe([item for item in terms if item])


def _is_part_like_target(target: str) -> bool:
    text = _normalize_phrase(target)
    return any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in (
            "handle",
            "hand",
            "hands",
            "arm",
            "arms",
            "claw",
            "claws",
            "paw",
            "paws",
            "finger",
            "fingers",
            "face",
            "eye",
            "eyes",
            "leg",
            "legs",
            "wheel",
            "wheels",
            "seat",
            "strap",
            "brim",
        )
    )


def _bounded_index(value: Any, size: int) -> int:
    index = _coerce_int(value, default=0)
    if index is None:
        index = 0
    return max(0, min(index, size - 1))


def _coerce_int(value: Any, *, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else 1.0
    return max(0.0, min(score, 1.0))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
