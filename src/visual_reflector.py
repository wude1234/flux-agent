"""Idea2Img-style visual selection and reflection behind VLM adapters."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .clients import VLMClient
from .prompt_constraints import PromptConstraints, extract_constraints, make_constraints_context


ERROR_TYPES = {
    "missing_object",
    "wrong_attribute",
    "wrong_count",
    "wrong_relation",
    "style_mismatch",
    "artifact",
}


class VisualReflector:
    """Use a VLM adapter to select draft images and produce structured critiques."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm

    def select_best(
        self,
        user_prompt: str,
        prompts: Sequence[str],
        image_paths: Sequence[str],
    ) -> dict[str, Any]:
        """Rank candidate images and return the selected image metadata."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        prompts = [_clean_text(prompt, "prompt") for prompt in prompts]
        image_paths = [_clean_text(path, "image_path") for path in image_paths]
        if not prompts:
            raise ValueError("prompts must not be empty")
        if len(prompts) != len(image_paths):
            raise ValueError("prompts and image_paths must have the same length")

        request = _selection_request(user_prompt, prompts, image_paths)
        raw_response = self.vlm.vision(request, list(image_paths))
        parsed = _parse_selection_response(raw_response, len(image_paths))
        selected_index = parsed["selected_index"]

        return {
            "selected_index": selected_index,
            "selected_image": image_paths[selected_index],
            "selected_prompt": prompts[selected_index],
            "scores": parsed["scores"],
            "warnings": parsed.get("warnings", []),
            "raw_response": raw_response,
            "request": request,
        }

    def reflect(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Critique one generated image against the user idea and current prompt."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        history = [deepcopy(dict(item)) for item in history or []]

        request = _reflection_request(user_prompt, prompt, image_path, history)
        raw_response = self.vlm.vision(request, [image_path])
        critique = _parse_critique_response(raw_response)
        critique.update(
            {
                "image_path": image_path,
                "prompt": prompt,
                "raw_response": raw_response,
                "request": request,
            }
        )
        return critique

    def check_constraints(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        constraints: PromptConstraints | Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Ask the VLM to verify user-grounded constraints explicitly."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        constraints = constraints or extract_constraints(user_prompt)
        history = [deepcopy(dict(item)) for item in history or []]

        request = _constraint_check_request(
            user_prompt,
            prompt,
            image_path,
            constraints,
            history,
        )
        raw_response = self.vlm.vision(request, [image_path])
        check = _parse_constraint_check_response(raw_response)
        check.update(
            {
                "image_path": image_path,
                "prompt": prompt,
                "raw_response": raw_response,
                "request": request,
            }
        )
        return check


def build_round_record(
    *,
    round_index: int,
    prompt: str,
    images: Sequence[str],
    selected_image: str,
    feedback: Mapping[str, Any] | str,
    revised_prompt: str,
) -> dict[str, Any]:
    """Build the per-round JSON-compatible record required by M1."""

    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    image_paths = [_clean_text(path, "image path") for path in images]
    selected_image = _clean_text(selected_image, "selected_image")
    if selected_image not in image_paths:
        raise ValueError("selected_image must be one of images")
    if isinstance(feedback, Mapping):
        feedback_value: dict[str, Any] | str = deepcopy(dict(feedback))
    elif isinstance(feedback, str):
        feedback_value = _clean_text(feedback, "feedback")
    else:
        raise TypeError("feedback must be a mapping or string")

    return {
        "round": round_index,
        "prompt": _clean_text(prompt, "prompt"),
        "images": image_paths,
        "selected_image": selected_image,
        "feedback": feedback_value,
        "revised_prompt": _clean_text(revised_prompt, "revised_prompt"),
    }


def _selection_request(
    user_prompt: str, prompts: Sequence[str], image_paths: Sequence[str]
) -> str:
    candidate_lines = [
        f"{index}. prompt={prompt!r}, image_path={image_paths[index]!r}"
        for index, prompt in enumerate(prompts)
    ]
    return "\n".join(
        [
            "You are the VisualCriticAgent for an Idea2Img-style loop.",
            "Select the draft image that best matches the user imagined IDEA.",
            "Priority order: first original user IDEA, then current prompt details,",
            "then aesthetics. A beautiful image with wrong user-specified colors,",
            "subjects, actions, or relations must not receive a high score.",
            make_constraints_context(extract_constraints(user_prompt)),
            "Judge object counts, attributes, entities, spatial relations, style,",
            "background, visible artifacts, and overall image quality.",
            "Return JSON with selected_index and scores. If JSON is impossible,",
            "wrap the selected index with <START> and <END>.",
            f"IDEA: {user_prompt}",
            "Candidates:",
            *candidate_lines,
        ]
    )


def _reflection_request(
    user_prompt: str,
    prompt: str,
    image_path: str,
    history: Sequence[Mapping[str, Any]],
) -> str:
    history_blob = json.dumps(
        _compact_reflection_history(history),
        ensure_ascii=False,
        sort_keys=True,
    )
    constraints = extract_constraints(user_prompt)
    return "\n".join(
        [
            "You are the VisualCriticAgent in an Idea2Img-style refinement loop.",
            "Compare the generated image with the original user's IDEA first,",
            "then with the current expanded prompt. The original IDEA is binding.",
            "Treat current prompt details that are not in the original IDEA as soft context, not hard requirements.",
            "If the image violates a user-specified color, subject, action, or",
            "spatial relation, report that as the main error even if aesthetics are good.",
            make_constraints_context(constraints),
            "Focus on one main user-grounded thing to improve. Avoid repeating prior feedback.",
            "Return JSON with score, errors, strengths, revision_hint, and user_grounded.",
            "Allowed error types: missing_object, wrong_attribute, wrong_count,",
            "wrong_relation, style_mismatch, artifact.",
            f"IDEA: {user_prompt}",
            f"Current generation prompt: {prompt}",
            f"Current image path: {image_path}",
            f"History: {history_blob}",
        ]
    )


def _compact_reflection_history(
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
            "selected_image": item.get("selected_image"),
        }
        if isinstance(feedback, Mapping):
            entry["score"] = feedback.get("score")
            entry["errors"] = _compact_history_errors(feedback.get("errors", []))
            entry["revision_hint"] = _truncate_text(
                str(feedback.get("revision_hint") or ""),
                400,
            )
            check = feedback.get("constraint_check")
            if isinstance(check, Mapping):
                entry["constraint_check"] = {
                    "passed": check.get("passed"),
                    "score": check.get("score"),
                    "source": check.get("source"),
                    "errors": _compact_history_errors(check.get("errors", [])),
                }
        compact.append(entry)
    return compact


def _compact_history_errors(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        records = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = [item for item in value if isinstance(item, Mapping)]
    else:
        records = []
    compact: list[dict[str, Any]] = []
    for item in records[:6]:
        compact.append(
            {
                "type": item.get("type"),
                "prompt_span": item.get("prompt_span"),
                "question_id": item.get("question_id"),
                "evidence": _truncate_text(
                    str(item.get("evidence") or item.get("description") or ""),
                    240,
                ),
            }
        )
    return compact


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _constraint_check_request(
    user_prompt: str,
    prompt: str,
    image_path: str,
    constraints: PromptConstraints | Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> str:
    history_blob = json.dumps(list(history), ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            "You are a strict visual constraint checker for a text-to-image agent.",
            "Evaluate only the original user's binding constraints, not beauty.",
            "The original user IDEA outranks the expanded prompt.",
            "Check every user-specified color-object binding, subject, action,",
            "and relation. Be skeptical: if the handle, grip, or color is not",
            "clearly visible, mark that constraint as failed.",
            make_constraints_context(constraints),
            "Return JSON with: passed, score, checks, errors, localized_errors, repair_plan, revision_hint.",
            "Each check should include type, target, expected, observed, passed,",
            "and description. Use score 0-1 for user-intent constraint match.",
            "For each failed localizable constraint, add localized_errors[] with:",
            "{error_type, target_object, target_attribute, expected, observed,",
            "bbox:[x,y,width,height], bbox_confidence, repair_kind, editability,",
            "repair_instruction, prompt_patch}. Use image pixel coordinates.",
            "repair_kind must be one of text_overlay, symbol_overlay, shape_overlay,",
            "bbox_shape_inpaint, existing_object_inpaint, layout_regenerate, count_rerank.",
            "Only give a bbox when the target region is visually localizable; otherwise omit bbox or set bbox_confidence low.",
            f"IDEA: {user_prompt}",
            f"Current generation prompt: {prompt}",
            f"Current image path: {image_path}",
            f"History: {history_blob}",
        ]
    )


def _parse_selection_response(response: str, num_images: int) -> dict[str, Any]:
    data = _extract_json_object(response)
    scores = _scores_from_json(data, num_images) if data else []
    selected_index = _selected_index_from_json(data) if data else None
    warnings: list[str] = []

    if selected_index is None:
        selected_index = _selected_index_from_tags(response)
    if selected_index is None:
        extracted_scores = _scores_from_text(response, num_images)
        if extracted_scores:
            scores = extracted_scores
            selected_index = max(scores, key=lambda item: item["score"])["index"]
    if selected_index is None:
        selected_index = 0

    bounded_index = _bounded_index(selected_index, num_images)
    if bounded_index != selected_index:
        warnings.append(
            f"selected_index {selected_index} out of range for {num_images} image(s); clamped to {bounded_index}"
        )
    selected_index = bounded_index
    if not scores:
        scores = [
            {
                "index": index,
                "score": 1.0 if index == selected_index else 0.0,
                "reason": "No structured scores returned by VLM.",
            }
            for index in range(num_images)
        ]
    return {"selected_index": selected_index, "scores": scores, "warnings": warnings}


def _parse_critique_response(response: str) -> dict[str, Any]:
    data = _extract_json_object(response)
    if data:
        score = _normalize_score(data.get("score", data.get("overall_score", 0.5)))
        errors = _normalize_errors(data.get("errors", []), response)
        strengths = _normalize_strengths(data.get("strengths", []))
        revision_hint = _clean_optional_text(data.get("revision_hint"))
        if not revision_hint:
            revision_hint = _clean_optional_text(data.get("feedback"))
        if not revision_hint:
            revision_hint = _first_start_end(response) or response.strip()
        if not errors and "errors" not in data and score < 0.95:
            errors = [_inferred_error(revision_hint)]
        return {
            "score": score,
            "errors": errors,
            "strengths": strengths,
            "revision_hint": revision_hint,
            "user_grounded": bool(data.get("user_grounded", True)),
        }

    reason = _first_start_end(response) or response.strip()
    score = _score_from_text(response, default=0.5)
    errors = [_inferred_error(reason)] if reason else []
    return {
        "score": score,
        "errors": errors,
        "strengths": [],
        "revision_hint": reason or "No critique returned.",
        "user_grounded": True,
    }


def _parse_constraint_check_response(response: str) -> dict[str, Any]:
    data = _extract_json_object(response)
    if data:
        score = _normalize_score(
            data.get("score", data.get("constraint_score", data.get("overall_score", 0.5)))
        )
        checks = _normalize_constraint_checks(
            data.get("checks", data.get("constraints", []))
        )
        errors = _normalize_errors(data.get("errors", []), response)
        localized_errors = _normalize_localized_errors(
            data.get("localized_errors", data.get("localized_errors_json", []))
        )
        for check in checks:
            if check.get("passed") is False:
                errors.append(_error_from_failed_check(check))
        errors = _dedupe_errors(errors)
        passed_value = data.get("passed", data.get("all_passed"))
        passed = bool(passed_value) if passed_value is not None else score >= 0.85
        if errors:
            passed = False
        revision_hint = _first_text(
            data.get("revision_hint"),
            data.get("feedback"),
            data.get("description"),
        )
        if not revision_hint and errors:
            revision_hint = "Fix user-grounded constraints: " + "; ".join(
                item["evidence"] for item in errors[:3] if item.get("evidence")
            )
        return {
            "passed": passed,
            "score": score,
            "checks": checks,
            "errors": errors,
            "localized_errors": localized_errors,
            "repair_plan": _normalize_repair_plan_hint(data.get("repair_plan", {}), localized_errors),
            "strengths": _normalize_strengths(data.get("strengths", [])),
            "revision_hint": revision_hint or "No constraint check revision returned.",
            "user_grounded": True,
        }

    reason = _first_start_end(response) or response.strip()
    score = _score_from_text(response, default=0.5)
    errors = [_inferred_error(reason)] if reason and score < 0.85 else []
    return {
        "passed": score >= 0.85 and not errors,
        "score": score,
        "checks": [],
        "errors": errors,
        "localized_errors": [],
        "strengths": [],
        "revision_hint": reason or "No constraint check returned.",
        "user_grounded": True,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
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


def _scores_from_json(
    data: Mapping[str, Any] | None, num_images: int
) -> list[dict[str, Any]]:
    if not data:
        return []
    raw_scores = data.get("scores", [])
    if isinstance(raw_scores, Mapping) and _looks_like_scalar_metric_map(raw_scores):
        return [
            {
                "index": 0,
                "score": _aggregate_metric_score(raw_scores),
                "reason": _first_text(
                    data.get("reason"),
                    data.get("feedback"),
                    data.get("description"),
                ),
            }
        ][:num_images]
    if isinstance(raw_scores, Mapping):
        raw_scores = [
            {"index": key, "score": value} for key, value in raw_scores.items()
        ]
    if not isinstance(raw_scores, list):
        return []

    scores: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(raw_scores):
        if isinstance(item, Mapping):
            index = _coerce_int(item.get("index", fallback_index), fallback_index)
            score_source = item.get("score")
            if score_source is None:
                score_source = item.get("overall_image_quality")
            score = _normalize_score(score_source if score_source is not None else 0.0)
            reason = _first_text(
                item.get("reason"),
                item.get("feedback"),
                item.get("description"),
            )
        else:
            index = fallback_index
            score = _normalize_score(item)
            reason = ""
        if 0 <= index < num_images:
            scores.append({"index": index, "score": score, "reason": reason})
    return scores


def _normalize_constraint_checks(raw_checks: Any) -> list[dict[str, Any]]:
    if isinstance(raw_checks, Mapping):
        raw_checks = [raw_checks]
    if not isinstance(raw_checks, list):
        return []

    checks: list[dict[str, Any]] = []
    for item in raw_checks:
        if not isinstance(item, Mapping):
            continue
        passed = item.get("passed", item.get("pass", item.get("ok")))
        if isinstance(passed, str):
            passed = passed.strip().lower() in {"true", "yes", "pass", "passed", "ok"}
        checks.append(
            {
                "type": _first_text(item.get("type"), item.get("constraint_type")),
                "target": _first_text(item.get("target"), item.get("object"), item.get("subject")),
                "expected": _first_text(item.get("expected"), item.get("required")),
                "observed": _first_text(item.get("observed"), item.get("actual")),
                "passed": bool(passed) if passed is not None else None,
                "description": _first_text(
                    item.get("description"),
                    item.get("evidence"),
                    item.get("reason"),
                ),
            }
        )
    return checks


def _error_from_failed_check(check: Mapping[str, Any]) -> dict[str, str]:
    evidence = _first_text(
        check.get("description"),
        (
            f"Expected {check.get('expected')} for {check.get('target')}, "
            f"observed {check.get('observed')}."
        ),
    )
    return {
        "type": _normalize_error_type(check.get("type"), evidence),
        "evidence": evidence,
        "prompt_span": _first_text(check.get("target"), check.get("expected")),
    }


def _dedupe_errors(errors: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in errors:
        normalized = {
            "type": str(item.get("type", "wrong_attribute")),
            "evidence": str(item.get("evidence", "")),
            "prompt_span": str(item.get("prompt_span", "")),
        }
        key = (
            normalized["type"].lower(),
            normalized["evidence"].lower(),
            normalized["prompt_span"].lower(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _looks_like_scalar_metric_map(raw_scores: Mapping[str, Any]) -> bool:
    if not raw_scores:
        return False
    if all(str(key).strip().lstrip("-").isdigit() for key in raw_scores):
        return False
    return all(_is_score_like(value) for value in raw_scores.values())


def _aggregate_metric_score(raw_scores: Mapping[str, Any]) -> float:
    preferred = raw_scores.get("overall_image_quality")
    if preferred is not None:
        return _normalize_score(preferred)
    values = [_normalize_score(value) for value in raw_scores.values() if _is_score_like(value)]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_score_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _selected_index_from_json(data: Mapping[str, Any] | None) -> int | None:
    if not data:
        return None
    for key in ("selected_index", "best_index", "index"):
        if key in data:
            return _coerce_int(data[key], -1)
    selected = data.get("selected")
    if isinstance(selected, Mapping) and "index" in selected:
        return _coerce_int(selected["index"], -1)
    return None


def _selected_index_from_tags(text: str) -> int | None:
    match = re.search(r"<START>\s*(\d+)\s*</?END>", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _scores_from_text(text: str, num_images: int) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?:Image\s*)?(\d+)[^\n]{0,160}?"
        r"(?:overall\s+)?score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    scores: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        index = int(match.group(1))
        if 0 <= index < num_images:
            scores.append(
                {
                    "index": index,
                    "score": _normalize_score(match.group(2)),
                    "reason": "",
                }
            )
    return scores


def _score_from_text(text: str, default: float) -> float:
    match = re.search(
        r"(?:overall\s+)?score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_score(match.group(1))
    return default


def _normalize_errors(raw_errors: Any, source_text: str) -> list[dict[str, str]]:
    if isinstance(raw_errors, Mapping):
        raw_errors = [raw_errors]
    if not isinstance(raw_errors, list):
        return [_inferred_error(str(raw_errors or source_text))]

    errors: list[dict[str, str]] = []
    for item in raw_errors:
        if isinstance(item, Mapping):
            evidence = _first_text(
                item.get("evidence"),
                item.get("description"),
                item.get("reason"),
                item.get("message"),
            )
            prompt_span = _clean_optional_text(item.get("prompt_span"))
            error_type = _normalize_error_type(item.get("type"), evidence)
        else:
            evidence = str(item).strip()
            prompt_span = ""
            error_type = _classify_error(evidence)
        if evidence or prompt_span:
            errors.append(
                {
                    "type": error_type,
                    "evidence": evidence,
                    "prompt_span": prompt_span,
                }
            )
    return errors


def _normalize_localized_errors(raw_errors: Any) -> list[dict[str, Any]]:
    if isinstance(raw_errors, Mapping):
        raw_errors = [raw_errors]
    if not isinstance(raw_errors, list):
        return []
    records: list[dict[str, Any]] = []
    for item in raw_errors:
        if not isinstance(item, Mapping):
            continue
        bbox = _normalize_bbox(item.get("bbox") or item.get("box") or item.get("target_bbox"))
        record: dict[str, Any] = {
            "error_type": _normalize_error_type(
                item.get("error_type", item.get("type")),
                _first_text(item.get("observed"), item.get("evidence"), item.get("description")),
            ),
            "target_object": _first_text(item.get("target_object"), item.get("target"), item.get("object")),
            "target_attribute": _first_text(item.get("target_attribute"), item.get("attribute")),
            "expected": _first_text(item.get("expected"), item.get("should_be")),
            "observed": _first_text(item.get("observed"), item.get("evidence"), item.get("description")),
            "bbox_confidence": _normalize_optional_score(
                item.get("bbox_confidence", item.get("confidence"))
            ),
            "repair_kind": _normalize_repair_kind_hint(item.get("repair_kind", item.get("route"))),
            "editability": _normalize_editability(item.get("editability")),
            "repair_instruction": _first_text(
                item.get("repair_instruction"),
                item.get("instruction"),
                item.get("revision_hint"),
            ),
            "prompt_patch": _first_text(item.get("prompt_patch"), item.get("patch")),
        }
        if bbox is not None:
            record["bbox"] = bbox
        records.append({key: value for key, value in record.items() if value not in ("", None)})
    return records


def _normalize_repair_plan_hint(
    raw_plan: Any,
    localized_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = dict(raw_plan) if isinstance(raw_plan, Mapping) else {}
    if not plan and localized_errors:
        first = localized_errors[0]
        repair_kind = str(first.get("repair_kind") or "").strip()
        plan = {
            "primary_action": "efficient_repair" if repair_kind else "none",
            "tool_sequence": [repair_kind] if repair_kind else [],
            "typed_route": repair_kind,
            "target_object": first.get("target_object", ""),
            "target_attribute": first.get("target_attribute", ""),
            "reason": first.get("repair_instruction") or first.get("observed") or "",
        }
        for key in ("bbox", "bbox_confidence", "expected", "observed", "prompt_patch"):
            if key in first:
                plan[key] = deepcopy(first[key])
    if "typed_route" in plan:
        plan["typed_route"] = _normalize_repair_kind_hint(plan.get("typed_route"))
    elif "repair_kind" in plan:
        plan["typed_route"] = _normalize_repair_kind_hint(plan.get("repair_kind"))
    return plan


def _normalize_bbox(value: Any) -> list[int] | None:
    if isinstance(value, Mapping):
        value = [
            value.get("x", value.get("left")),
            value.get("y", value.get("top")),
            value.get("width", value.get("w")),
            value.get("height", value.get("h")),
        ]
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return bbox


def _normalize_optional_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_repair_kind_hint(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "text": "text_overlay",
        "ocr": "text_overlay",
        "text_symbol": "text_overlay",
        "symbol": "symbol_overlay",
        "shape": "shape_overlay",
        "shape_overlay": "shape_overlay",
        "primitive_overlay": "shape_overlay",
        "flat_overlay": "shape_overlay",
        "insert": "bbox_shape_inpaint",
        "object_insertion": "bbox_shape_inpaint",
        "occlusion": "shape_overlay",
        "recolor": "existing_object_inpaint",
        "local_repair": "existing_object_inpaint",
        "relation_repair": "existing_object_inpaint",
        "spatial": "layout_regenerate",
        "layout": "layout_regenerate",
        "count": "count_rerank",
    }
    return aliases.get(text, text)


def _normalize_editability(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"easy", "yes", "true"}:
        return "high"
    if text in {"hard", "no", "false"}:
        return "low"
    return ""


def _normalize_strengths(raw_strengths: Any) -> list[str]:
    if isinstance(raw_strengths, str):
        return [raw_strengths.strip()] if raw_strengths.strip() else []
    if not isinstance(raw_strengths, list):
        return []
    strengths: list[str] = []
    for item in raw_strengths:
        if isinstance(item, Mapping):
            text = _first_text(item.get("description"), item.get("evidence"), item.get("text"))
        else:
            text = str(item).strip()
        if text:
            strengths.append(text)
    return strengths


def _inferred_error(text: str) -> dict[str, str]:
    text = text.strip()
    return {
        "type": _classify_error(text),
        "evidence": text,
        "prompt_span": "",
    }


def _normalize_error_type(value: Any, evidence: str) -> str:
    value = str(value or "").strip()
    if value in ERROR_TYPES:
        return value
    return _classify_error(evidence)


def _classify_error(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("missing", "absent", "not present")):
        return "missing_object"
    if any(word in lowered for word in ("count", "number", "exactly", "only ")):
        return "wrong_count"
    if any(word in lowered for word in ("relation", "spatial", "position", "pose")):
        return "wrong_relation"
    if any(word in lowered for word in ("style", "background", "lighting", "mood")):
        return "style_mismatch"
    if any(word in lowered for word in ("artifact", "distorted", "blurry", "deformed")):
        return "artifact"
    return "wrong_attribute"


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


def _bounded_index(index: int, num_images: int) -> int:
    if num_images < 1:
        raise ValueError("num_images must be at least 1")
    return max(0, min(index, num_images - 1))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_start_end(text: str) -> str | None:
    match = re.search(
        r"<START>(.*?)</?END>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None
    return match.group(1).strip()


def _clean_optional_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_optional_text(value)
        if text:
            return text
    return ""


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value
