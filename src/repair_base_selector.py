"""Base-image selection before local repair tools run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Protocol, Sequence

from .candidate_arbitration import (
    add_human_rule_evidence,
    constraint_items_from_constraints,
    summarize_constraint_check,
)
from .prompt_constraints import PromptConstraints
from .repairable_candidate_selector import build_repairability_profile


class VLMRepairabilityJudge(Protocol):
    """Optional pairwise repairability judge backed by a VLM/LLM adapter."""

    def judge(self, request: str, image_paths: Sequence[str]) -> Mapping[str, Any] | str:
        ...


def select_repair_base(
    *,
    image_paths: Sequence[str],
    prompts: Sequence[str],
    candidate_checks: Sequence[Mapping[str, Any]] | None,
    constraints: PromptConstraints,
    repair_plan: Mapping[str, Any] | None = None,
    reward_ranking: Mapping[str, Any] | None = None,
    current_index: int | None = None,
    current_feedback: Mapping[str, Any] | None = None,
    vlm_judge: VLMRepairabilityJudge | None = None,
) -> dict[str, Any]:
    """Rank candidates by original constraints, repairability, and edit risk.

    This decision happens before local repair. It answers "which candidate is
    the best base to repair into the user's original prompt", not "which current
    image looks best".
    """

    images = [str(path) for path in image_paths]
    candidate_prompts = [str(prompt) for prompt in prompts]
    if not images:
        raise ValueError("image_paths must not be empty")
    if len(candidate_prompts) != len(images):
        raise ValueError("prompts and image_paths must have the same length")

    checks_by_index = _checks_by_index(candidate_checks)
    reward_scores = _reward_scores_by_index(reward_ranking)
    action_plan = _normalize_repair_plan(repair_plan, constraints)
    current_index = _bounded_index(current_index, len(images)) if current_index is not None else None

    candidates = [
        _build_candidate(
            index=index,
            image_path=image_path,
            prompt=candidate_prompts[index],
            check=checks_by_index.get(index),
            current_feedback=current_feedback if index == current_index else None,
            constraints=constraints,
            repair_plan=action_plan,
            reward_score=reward_scores.get(index),
            current_index=current_index,
        )
        for index, image_path in enumerate(images)
    ]

    vlm_preference: dict[str, Any] | None = None
    if vlm_judge is not None and len(candidates) > 1:
        vlm_preference = _maybe_vlm_preference(
            vlm_judge=vlm_judge,
            user_prompt=constraints.original_prompt,
            candidates=candidates,
        )
        preferred_index = _preferred_index(vlm_preference, len(candidates))
        if preferred_index is not None:
            for item in candidates:
                item["vlm_pairwise_preferred"] = int(item["index"]) == int(preferred_index)

    ranked = sorted(candidates, key=_sort_key, reverse=True)
    selected = ranked[0]
    rejected = [_rejected_candidate(item, selected) for item in ranked[1:]]
    result = {
        "type": "repair_base_selection",
        "selected_index": int(selected["index"]),
        "selected_image": selected["image_path"],
        "selected_prompt": selected["prompt"],
        "current_index": current_index,
        "current_image": images[current_index] if current_index is not None else None,
        "intended_action": selected["intended_action"],
        "repair_target": selected["repair_target"],
        "satisfied_constraints": list(selected["satisfied_constraints"]),
        "failed_constraints": list(selected["failed_constraints"]),
        "repairability_score": selected["repairability_score"],
        "edit_risk_score": selected["edit_risk_score"],
        "reward_score": selected["reward_score"],
        "decision_reasons": list(selected["decision_reasons"]),
        "rejected_candidates": rejected,
        "ranking": [_ranking_record(item) for item in ranked],
        "vlm_pairwise": deepcopy(vlm_preference),
    }
    if current_index is not None and int(selected["index"]) != int(current_index):
        result["overrode_current_selection"] = True
    return result


def _build_candidate(
    *,
    index: int,
    image_path: str,
    prompt: str,
    check: Mapping[str, Any] | None,
    current_feedback: Mapping[str, Any] | None,
    constraints: PromptConstraints,
    repair_plan: Mapping[str, Any],
    reward_score: float | None,
    current_index: int | None,
) -> dict[str, Any]:
    check = deepcopy(dict(check)) if isinstance(check, Mapping) else {}
    if isinstance(current_feedback, Mapping):
        check = _merge_current_feedback_check(check, current_feedback)
    summary = add_human_rule_evidence(
        summarize_constraint_check(check),
        check,
        constraint_items_from_constraints(constraints),
    )
    profile = build_repairability_profile(
        {
            "index": index,
            "image_path": image_path,
            "prompt": prompt,
            "constraint_check": check,
        },
        fallback_index=index,
        repair_plan=repair_plan,
        constraints=constraints,
    )
    satisfied, failed = _constraint_labels(check)
    action = str(repair_plan.get("primary_action") or profile["repairability"].get("action") or "none")
    hard_failures = _unique_hard_failure_count(failed, summary)
    repairability = _coerce_number(profile["repairability"].get("score"), default=0.0)
    edit_risk = _estimate_edit_risk(
        summary=summary,
        repairability=profile["repairability"],
        action=action,
        hard_failures=hard_failures,
    )
    failed_checks = int(summary.get("failed_checks", 0) or 0)
    passed_checks = int(summary.get("passed_checks", 0) or 0)
    reward = _coerce_float(reward_score, default=None)
    decision_reasons = _decision_reasons(
        summary=summary,
        repairability=profile["repairability"],
        action=action,
        satisfied=satisfied,
        failed=failed,
        hard_failures=hard_failures,
    )
    return {
        "index": int(index),
        "image_path": image_path,
        "prompt": prompt,
        "constraint_check": check,
        "constraint_summary": summary,
        "repairability_profile": profile,
        "intended_action": action,
        "repair_target": str(profile["repairability"].get("target_object") or ""),
        "satisfied_constraints": satisfied,
        "failed_constraints": failed,
        "hard_failures": hard_failures,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "constraint_score": _coerce_float(summary.get("score"), default=0.0) or 0.0,
        "repairability_score": repairability,
        "edit_risk_score": edit_risk,
        "reward_score": reward,
        "is_current": current_index is not None and int(index) == int(current_index),
        "vlm_pairwise_preferred": False,
        "decision_reasons": decision_reasons,
    }


def _sort_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    reward_score = candidate.get("reward_score")
    reward_value = float(reward_score) if reward_score is not None else -1.0
    return (
        float(_tier_rank(candidate)),
        -float(candidate.get("hard_failures", 0) or 0),
        float(candidate.get("repairability_score", 0.0) or 0.0),
        -float(candidate.get("edit_risk_score", 0.0) or 0.0),
        float(candidate.get("passed_checks", 0) or 0),
        float(candidate.get("constraint_score", 0.0) or 0.0),
        -float(candidate.get("failed_checks", 0) or 0),
        1.0 if candidate.get("vlm_pairwise_preferred") else 0.0,
        reward_value,
        1.0 if candidate.get("is_current") else 0.0,
        -float(candidate.get("index", 0) or 0),
    )


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


def _constraint_labels(check: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    satisfied: list[str] = []
    failed: list[str] = []
    checks = check.get("checks", [])
    if isinstance(checks, Mapping):
        checks = [checks]
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            label = _check_label(item)
            if not label:
                continue
            if item.get("passed") is True:
                satisfied.append(label)
            elif item.get("passed") is False:
                failed.append(label)

    errors = check.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for item in errors:
            if isinstance(item, Mapping):
                label = _error_label(item)
                if label:
                    failed.append(label)
    return _dedupe(satisfied), _dedupe(failed)


def _merge_current_feedback_check(
    check: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge fresher selected-image feedback into a candidate check.

    Candidate checks are often gathered before the final selected-image critique
    and evaluator. If that later feedback finds a hard failure, the repair-base
    selector should not keep trusting a stale "passed" candidate check.
    """

    result = deepcopy(dict(check))
    checks = _list_records(result.get("checks"))
    errors = _list_records(result.get("errors"))
    for record in _feedback_records(feedback, key="checks"):
        checks.append(record)
    for record in _feedback_records(feedback, key="errors"):
        errors.append(record)
    if checks:
        result["checks"] = checks
    if errors:
        result["errors"] = errors
    if any(item.get("passed") is False for item in checks) or errors:
        result["passed"] = False
    scores = [
        _coerce_float(value, default=None)
        for value in (
            result.get("score"),
            feedback.get("score"),
            _nested_value(feedback, "constraint_check", "score"),
            _nested_value(feedback, "evaluation", "score"),
        )
    ]
    valid_scores = [score for score in scores if score is not None]
    if valid_scores:
        result["score"] = min(valid_scores)
    result["merged_current_feedback"] = True
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


def _nested_value(mapping: Mapping[str, Any], nested_key: str, value_key: str) -> Any:
    nested = mapping.get(nested_key)
    if isinstance(nested, Mapping):
        return nested.get(value_key)
    return None


def _unique_hard_failure_count(
    failed_constraints: Sequence[str],
    summary: Mapping[str, Any],
) -> int:
    if not failed_constraints:
        return int(summary.get("hard_failures", 0) or 0)
    unique: set[str] = set()
    for label in failed_constraints:
        text = str(label).lower()
        if _looks_like_soft_failure(text):
            continue
        unique.add(_canonical_failure_key(text))
    if unique:
        return len(unique)
    return int(summary.get("hard_failures", 0) or 0)


def _looks_like_soft_failure(text: str) -> bool:
    return any(token in text for token in ("cinematic", "photo", "style", "lighting"))


def _canonical_failure_key(text: str) -> str:
    if "umbrella" in text and (
        "blue" in text or "color" in text or "brown" in text or "red" in text
    ):
        return "umbrella_color"
    if "robot" in text and ("red" in text or "color" in text):
        return "robot_color"
    if any(token in text for token in ("grip", "holding", "handle", "relation")):
        return "relation_action"
    if any(token in text for token in ("missing", "not present", "not visible", "absent")):
        return f"missing:{text[:80]}"
    return text[:120]


def _estimate_edit_risk(
    *,
    summary: Mapping[str, Any],
    repairability: Mapping[str, Any],
    action: str,
    hard_failures: int,
) -> float:
    risk = 0.2
    failed_checks = float(summary.get("failed_checks", 0) or 0)
    risk += 0.12 * float(hard_failures)
    risk += 0.04 * failed_checks
    if action == "object_insertion":
        risk += 0.28
    elif action == "relation_repair":
        risk += 0.20
    elif action == "recolor":
        risk += 0.10
    if repairability.get("missing_objects"):
        risk += 0.12 * len(repairability.get("missing_objects") or [])
    if repairability.get("major_missing_objects"):
        risk += 0.2 * len(repairability.get("major_missing_objects") or [])
    if repairability.get("relation_action_failed") and action != "relation_repair":
        risk += 0.14
    return round(max(0.0, min(risk, 1.0)), 4)


def _decision_reasons(
    *,
    summary: Mapping[str, Any],
    repairability: Mapping[str, Any],
    action: str,
    satisfied: Sequence[str],
    failed: Sequence[str],
    hard_failures: int,
) -> list[str]:
    reasons: list[str] = []
    if hard_failures == 0:
        reasons.append("fewest hard user-constraint failures")
    else:
        reasons.append(f"{hard_failures} hard user-constraint failures")
    if satisfied:
        reasons.append(f"preserves {len(satisfied)} checked user constraints")
    if failed:
        reasons.append(f"remaining failures: {', '.join(str(item) for item in failed[:3])}")
    repair_reasons = repairability.get("reasons", [])
    if isinstance(repair_reasons, Sequence) and not isinstance(repair_reasons, (str, bytes)):
        reasons.extend(str(item) for item in list(repair_reasons)[:2])
    if action:
        reasons.append(f"planned repair action: {action}")
    return _dedupe(reasons)


def _rejected_candidate(
    candidate: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if candidate.get("hard_failures", 0) > selected.get("hard_failures", 0):
        reasons.append("more hard user-constraint failures than selected base")
    if candidate.get("repairability_score", 0.0) < selected.get("repairability_score", 0.0):
        reasons.append("lower repairability score")
    if candidate.get("edit_risk_score", 0.0) > selected.get("edit_risk_score", 0.0):
        reasons.append("higher edit risk")
    if not reasons:
        reasons.append("lower overall base-selection rank")
    return {
        "index": int(candidate["index"]),
        "image_path": candidate["image_path"],
        "failed_constraints": list(candidate["failed_constraints"]),
        "repairability_score": candidate["repairability_score"],
        "edit_risk_score": candidate["edit_risk_score"],
        "reward_score": candidate["reward_score"],
        "reason": "; ".join(reasons),
    }


def _ranking_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": int(candidate["index"]),
        "image_path": candidate["image_path"],
        "constraint_summary": deepcopy(dict(candidate["constraint_summary"])),
        "hard_failures": candidate["hard_failures"],
        "failed_checks": candidate["failed_checks"],
        "passed_checks": candidate["passed_checks"],
        "satisfied_constraints": list(candidate["satisfied_constraints"]),
        "failed_constraints": list(candidate["failed_constraints"]),
        "intended_action": candidate["intended_action"],
        "repair_target": candidate["repair_target"],
        "repairability_score": candidate["repairability_score"],
        "edit_risk_score": candidate["edit_risk_score"],
        "reward_score": candidate["reward_score"],
        "decision_reasons": list(candidate["decision_reasons"]),
    }


def _maybe_vlm_preference(
    *,
    vlm_judge: VLMRepairabilityJudge,
    user_prompt: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    request = _pairwise_request(user_prompt, candidates)
    image_paths = [str(item["image_path"]) for item in candidates]
    try:
        raw = vlm_judge.judge(request, image_paths)
    except Exception as exc:
        return {"source": "vlm_pairwise_repairability", "error": str(exc)}
    parsed = raw if isinstance(raw, Mapping) else {}
    if not parsed and isinstance(raw, str):
        parsed = _parse_selected_index_text(raw)
    result = deepcopy(dict(parsed)) if isinstance(parsed, Mapping) else {}
    result["source"] = "vlm_pairwise_repairability"
    result["request"] = request
    result["raw_response"] = raw
    return result


def _pairwise_request(user_prompt: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    summaries = [
        {
            "index": item["index"],
            "image_path": item["image_path"],
            "satisfied_constraints": item["satisfied_constraints"],
            "failed_constraints": item["failed_constraints"],
            "repairability_score": item["repairability_score"],
            "edit_risk_score": item["edit_risk_score"],
        }
        for item in candidates
    ]
    return (
        "Choose which candidate image is the best base for local repair into the "
        "original user prompt. Prefer preserving original hard constraints over "
        "aesthetic quality. Return JSON with selected_index and reason.\n"
        f"Original user prompt: {user_prompt}\n"
        f"Candidate summaries: {summaries}"
    )


def _preferred_index(preference: Mapping[str, Any] | None, size: int) -> int | None:
    if not isinstance(preference, Mapping):
        return None
    for key in ("selected_index", "preferred_index", "best_index"):
        if key in preference:
            return _bounded_index(preference.get(key), size)
    return None


def _parse_selected_index_text(text: str) -> dict[str, Any]:
    import re

    match = re.search(r"(?:selected|preferred|best)[^0-9]{0,20}(\d+)", text, re.I)
    if not match:
        match = re.search(r"\bindex[^0-9]{0,10}(\d+)", text, re.I)
    if not match:
        return {"reason": text.strip()}
    return {"selected_index": int(match.group(1)), "reason": text.strip()}


def _normalize_repair_plan(
    repair_plan: Mapping[str, Any] | None,
    constraints: PromptConstraints,
) -> dict[str, Any]:
    if isinstance(repair_plan, Mapping):
        result = deepcopy(dict(repair_plan))
    else:
        result = {}
    action = str(result.get("primary_action") or result.get("action") or "").strip().lower()
    if not action:
        action = _infer_action_from_constraints(constraints)
    result["primary_action"] = action
    if not result.get("target_object"):
        result["target_object"] = _default_target_object(action, constraints)
    if not result.get("target_attribute"):
        result["target_attribute"] = {
            "recolor": "color",
            "object_insertion": "presence",
            "relation_repair": "relation",
        }.get(action, "")
    return result


def _infer_action_from_constraints(constraints: PromptConstraints) -> str:
    if constraints.actions or constraints.relations:
        return "relation_repair"
    if constraints.colors:
        return "recolor"
    return "regenerate"


def _default_target_object(action: str, constraints: PromptConstraints) -> str:
    if action == "recolor" and constraints.colors:
        return next(iter(constraints.colors))
    if constraints.subjects:
        return str(constraints.subjects[0])
    if constraints.colors:
        return next(iter(constraints.colors))
    return ""


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
        check = item.get("constraint_check", item)
        if index is not None and isinstance(check, Mapping):
            result[int(index)] = check
    return result


def _reward_scores_by_index(reward_ranking: Mapping[str, Any] | None) -> dict[int, float]:
    if not isinstance(reward_ranking, Mapping):
        return {}
    scores = reward_ranking.get("scores", [])
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        return {}
    result: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        index = _coerce_int(item.get("index"), default=None)
        score = _coerce_float(item.get("score"), default=None)
        if index is not None and score is not None:
            result[int(index)] = float(score)
    return result


def _check_label(item: Mapping[str, Any]) -> str:
    target = str(item.get("target") or "").strip()
    expected = str(item.get("expected") or "").strip()
    check_type = str(item.get("type") or "").strip()
    if target and expected:
        return f"{target}: {expected}"
    return target or expected or check_type


def _error_label(item: Mapping[str, Any]) -> str:
    target = str(item.get("prompt_span") or item.get("target") or "").strip()
    evidence = str(item.get("evidence") or item.get("reason") or item.get("type") or "").strip()
    if target and evidence:
        return f"{target}: {evidence}"
    return target or evidence


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


def _coerce_float(value: Any, *, default: float | None) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else 1.0
    return max(0.0, min(score, 1.0))


def _coerce_number(value: Any, *, default: float) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return float(default)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
