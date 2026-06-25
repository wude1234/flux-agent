"""Image-grounded object state and geometry checks for M6.20."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .clients import VLMClient


GEOMETRY_RELATIONS = {"left_of", "right_of"}
CONTACT_REQUIRED_ACTIONS = {
    "carry",
    "carries",
    "carrying",
    "grasp",
    "grasps",
    "grasping",
    "grip",
    "grips",
    "gripping",
    "hold",
    "holds",
    "holding",
    "touch",
    "touches",
    "touching",
}


def augment_record_with_object_geometry(
    vlm: VLMClient,
    record: Mapping[str, Any],
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
) -> dict[str, Any]:
    """Add object-state geometry checks to a VQA constraint record.

    The first M6.20 use case is left/right validation. VLM QA often says a
    relation passes even when the visible object order is wrong. This adapter
    asks for bboxes, then verifies the relation in code.
    """

    output = deepcopy(dict(record))
    questions = _list_records(output.get("questions", []))
    relations = _geometry_relation_questions(questions)
    evidence_questions = _object_evidence_questions(
        questions,
        require_image_backed_evidence=_image_size_or_none(image_path) is not None,
    )
    if not relations and not evidence_questions:
        _attach_evidence_chain(output)
        return output

    targets = _evidence_targets([*relations, *evidence_questions])
    request = build_object_state_request(
        user_prompt=user_prompt,
        prompt=prompt,
        image_path=image_path,
        targets=targets,
    )
    try:
        raw_response = vlm.vision(request, [str(image_path)])
        image_size = _image_size_or_none(image_path)
        object_state = parse_object_state_response(
            raw_response,
            targets=targets,
            image_size=image_size,
        )
    except Exception as exc:
        output["object_state"] = {
            "available": False,
            "method": "vlm_bbox_object_state",
            "error": str(exc),
            "request": request,
        }
        return output

    geometry = verify_spatial_geometry(relations, object_state)
    object_evidence = verify_object_part_evidence(
        evidence_questions,
        output.get("answers", []),
        object_state,
    )
    object_state.update({"request": request, "raw_response": raw_response})
    output["object_state"] = object_state
    output["geometry_verification"] = geometry
    constraint_check = merge_geometry_into_constraint_check(
        output.get("constraint_check", {}),
        geometry,
    )
    constraint_check = merge_object_evidence_into_constraint_check(
        constraint_check,
        object_evidence,
    )
    output["constraint_check"] = constraint_check
    output["object_evidence_verification"] = object_evidence
    output["summary"] = deepcopy(
        output["constraint_check"].get("question_summary", output.get("summary", {}))
    )
    _attach_evidence_chain(output)
    return output


def build_object_state_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    targets: Sequence[str],
) -> str:
    schema = {
        "objects": [
            {
                "name": "target name",
                "visible": True,
                "bbox": [0, 0, 1, 1],
                "mask_path": None,
                "attributes": {"color": "dominant visible color if relevant"},
                "count": "visible instance count if the requested target is a group",
                "parts": [
                    {
                        "name": "target part name",
                        "visible": True,
                        "bbox": [0, 0, 1, 1],
                        "evidence": "visible part evidence",
                    }
                ],
                "contact_regions": [
                    {
                        "name": "contact or relation region",
                        "visible": True,
                        "bbox": [0, 0, 1, 1],
                        "evidence": "visible contact evidence",
                    }
                ],
                "protected": False,
                "confidence": 0.0,
                "evidence": "short visible evidence",
            }
        ]
    }
    return "\n".join(
        [
            "You are building image-grounded object state for a text-to-image verifier.",
            "Use only visible evidence from the image.",
            "Return a tight bbox in image pixel coordinates [x, y, width, height] for each requested target.",
            "For plural/group targets, return one bbox covering the full visible group.",
            "For count targets, include count as the number of distinct visible instances.",
            "Include visible parts or contact regions only when they are directly visible.",
            "If a target is not visible, set visible=false and bbox=null.",
            "Do not infer hidden or offscreen objects.",
            "Return exactly one JSON object.",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Targets JSON: {json.dumps(list(targets), ensure_ascii=False)}",
            f"Original user prompt: {user_prompt}",
            f"Current generation prompt: {prompt}",
            f"Image path: {image_path}",
        ]
    )


def parse_object_state_response(
    response: str,
    *,
    targets: Sequence[str],
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    data = _extract_json(response)
    raw_objects: Any = []
    if isinstance(data, Mapping):
        raw_objects = data.get("objects", data.get("object_states", []))
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        raw_objects = data
    objects: list[dict[str, Any]] = []
    if isinstance(raw_objects, Mapping):
        raw_objects = [
            {"name": key, **(value if isinstance(value, Mapping) else {"bbox": value})}
            for key, value in raw_objects.items()
        ]
    if isinstance(raw_objects, Sequence) and not isinstance(raw_objects, (str, bytes)):
        for item in raw_objects:
            parsed = _parse_object_item(item, image_size=image_size)
            if parsed:
                objects.append(parsed)
    return {
        "available": bool(objects),
        "method": "vlm_bbox_object_state",
        "targets": list(targets),
        "objects": objects,
    }


def build_evidence_chain(
    questions: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    *,
    object_state: Mapping[str, Any] | None = None,
    geometry: Mapping[str, Any] | None = None,
    object_evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build TVI-CoT-style Think/Look/Verify records for hard checks."""

    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
        return []
    answer_by_id = {
        str(answer.get("id") or ""): answer
        for answer in answers
        if isinstance(answer, Mapping)
    }
    geometry_by_id = {
        str(check.get("question_id") or ""): check
        for check in _list_records((geometry or {}).get("checks"))
    }
    evidence_by_id = {
        str(check.get("question_id") or ""): check
        for check in _list_records((object_evidence or {}).get("checks"))
    }
    objects = (object_state or {}).get("objects", []) if isinstance(object_state, Mapping) else []
    steps: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, Mapping):
            continue
        question = deepcopy(dict(item))
        question_id = str(question.get("id") or "")
        if not question_id:
            continue
        answer = deepcopy(dict(answer_by_id.get(question_id, {})))
        geometry_check = deepcopy(dict(geometry_by_id.get(question_id, {})))
        targets = _question_targets(question)
        matched_objects = [
            match
            for target in targets
            for match in [_find_object_state(objects, target)]
            if match is not None
        ]
        evidence_check = deepcopy(dict(evidence_by_id.get(question_id, {})))
        verify = _verify_record(question, answer, geometry_check, evidence_check)
        status = _evidence_status(answer, geometry_check, evidence_check)
        steps.append(
            {
                "question_id": question_id,
                "category": question.get("category"),
                "think": _think_for_question(question),
                "look_request": {
                    "targets": targets,
                    "evidence_type": _evidence_type_for_question(question),
                    "question": question.get("question"),
                    "depends_on": list(question.get("depends_on", []) or []),
                },
                "look_result": {
                    "objects": matched_objects,
                    "object_evidence_available": bool(matched_objects),
                    "answer_evidence": answer.get("evidence", ""),
                    "blocked_by": list(answer.get("blocked_by", []) or []),
                    "geometry_check": geometry_check or None,
                    "object_evidence_check": evidence_check or None,
                },
                "verify": verify,
                "status": status,
            }
        )
    return steps


def verify_spatial_geometry(
    relation_questions: Sequence[Mapping[str, Any]],
    object_state: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    objects = object_state.get("objects", [])
    for question in relation_questions:
        source = question.get("source_constraint", {})
        if not isinstance(source, Mapping):
            continue
        subject = str(source.get("subject") or "").strip()
        obj = str(source.get("object") or "").strip()
        relation = str(source.get("relation") or "").strip()
        if relation not in GEOMETRY_RELATIONS:
            continue
        subject_state = _find_object_state(objects, subject)
        object_state_item = _find_object_state(objects, obj)
        check = _geometry_check_from_states(question, subject_state, object_state_item, relation)
        checks.append(check)
        if check["passed"] is False:
            errors.append(
                {
                    "type": "wrong_relation",
                    "prompt_span": check["target"],
                    "question_id": check["question_id"],
                    "evidence": check["description"],
                }
            )
    return {
        "source": "object_state_geometry",
        "checks": checks,
        "errors": errors,
        "passed": not errors,
    }


def merge_geometry_into_constraint_check(
    check: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(check)) if isinstance(check, Mapping) else {}
    geometry_checks = [
        deepcopy(dict(item))
        for item in geometry.get("checks", [])
        if isinstance(item, Mapping)
    ]
    geometry_errors = [
        deepcopy(dict(item))
        for item in geometry.get("errors", [])
        if isinstance(item, Mapping)
    ]
    if not geometry_checks:
        return result

    checks = _list_records(result.get("checks"))
    checks.extend(geometry_checks)
    result["checks"] = checks
    if geometry_errors:
        errors = _list_records(result.get("errors"))
        errors.extend(geometry_errors)
        result["errors"] = _dedupe_errors(errors)
        result["passed"] = False
        result["score"] = min(_coerce_float(result.get("score"), 1.0), 0.755)
        result["constraint_score"] = min(
            _coerce_float(result.get("constraint_score", result.get("score")), 1.0),
            0.755,
        )
    summary = deepcopy(dict(result.get("question_summary", {})))
    if summary:
        _update_summary_with_geometry(summary, geometry_checks, geometry_errors)
        result["question_summary"] = summary
    result["geometry_source"] = "object_state_geometry"
    return result


def verify_object_part_evidence(
    evidence_questions: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    object_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify VQA yes/count answers against object/part/contact evidence.

    This is deliberately conservative. It only overrides a VLM answer when the
    object-state response contains explicit contradictory evidence, or when an
    action/part relation has no visible object/part/contact support at all.
    """

    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    objects = object_state.get("objects", []) if isinstance(object_state, Mapping) else []
    answer_by_id = {
        str(answer.get("id") or ""): answer
        for answer in _list_records(answers)
    }
    for question in evidence_questions:
        question_id = str(question.get("id") or "")
        if not question_id:
            continue
        answer = answer_by_id.get(question_id, {})
        if answer.get("blocked_by"):
            continue
        if answer.get("passed") is not True:
            continue
        category = str(question.get("category") or "")
        if category == "count":
            check = _count_evidence_check(question, objects)
        elif category == "part_visibility":
            check = _part_evidence_check(question, objects)
        elif category == "action_relation":
            check = _action_relation_evidence_check(question, objects)
        else:
            continue
        if not check:
            continue
        checks.append(check)
        if check["passed"] is False:
            errors.append(
                {
                    "type": check["error_type"],
                    "prompt_span": check["target"],
                    "question_id": check["question_id"],
                    "evidence": check["description"],
                }
            )
    return {
        "source": "object_part_state_evidence",
        "checks": checks,
        "errors": errors,
        "passed": not errors,
    }


def merge_object_evidence_into_constraint_check(
    check: Mapping[str, Any],
    object_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(check)) if isinstance(check, Mapping) else {}
    evidence_checks = [
        deepcopy(dict(item))
        for item in object_evidence.get("checks", [])
        if isinstance(item, Mapping)
    ]
    evidence_errors = [
        deepcopy(dict(item))
        for item in object_evidence.get("errors", [])
        if isinstance(item, Mapping)
    ]
    if not evidence_checks:
        return result
    checks = _list_records(result.get("checks"))
    checks.extend(evidence_checks)
    result["checks"] = checks
    if evidence_errors:
        errors = _list_records(result.get("errors"))
        errors.extend(evidence_errors)
        result["errors"] = _dedupe_errors(errors)
        result["passed"] = False
        result["score"] = min(_coerce_float(result.get("score"), 1.0), 0.755)
        result["constraint_score"] = min(
            _coerce_float(result.get("constraint_score", result.get("score")), 1.0),
            0.755,
        )
    summary = deepcopy(dict(result.get("question_summary", {})))
    if summary:
        _update_summary_with_geometry(summary, evidence_checks, evidence_errors)
        result["question_summary"] = summary
    result["object_evidence_source"] = "object_part_state_evidence"
    return result


def _geometry_check_from_states(
    question: Mapping[str, Any],
    subject_state: Mapping[str, Any] | None,
    object_state_item: Mapping[str, Any] | None,
    relation: str,
) -> dict[str, Any]:
    source = question.get("source_constraint", {})
    subject = str(source.get("subject") or "").strip() if isinstance(source, Mapping) else ""
    obj = str(source.get("object") or "").strip() if isinstance(source, Mapping) else ""
    question_id = str(question.get("id") or f"geometry:{subject}:{obj}:{relation}")
    target = f"{subject}:{obj}:{relation}"
    if not subject_state or not object_state_item:
        return {
            "type": "relation",
            "category": "spatial_geometry",
            "target": target,
            "expected": relation,
            "observed": "bbox_unavailable",
            "passed": False,
            "description": f"Cannot verify {relation}: bbox missing for {subject} or {obj}.",
            "question_id": question_id,
        }
    subject_bbox = subject_state.get("bbox")
    object_bbox = object_state_item.get("bbox")
    subject_center = _bbox_center(subject_bbox)
    object_center = _bbox_center(object_bbox)
    if subject_center is None or object_center is None:
        return {
            "type": "relation",
            "category": "spatial_geometry",
            "target": target,
            "expected": relation,
            "observed": "invalid_bbox",
            "passed": False,
            "description": f"Cannot verify {relation}: invalid bbox for {subject} or {obj}.",
            "question_id": question_id,
        }
    margin = max(4.0, 0.03 * max(_bbox_width(subject_bbox), _bbox_width(object_bbox), 1.0))
    delta = subject_center[0] - object_center[0]
    passed = delta < -margin if relation == "left_of" else delta > margin
    observed = (
        f"{subject}_center_x={subject_center[0]:.1f}, "
        f"{obj}_center_x={object_center[0]:.1f}, delta={delta:.1f}"
    )
    return {
        "type": "relation",
        "category": "spatial_geometry",
        "target": target,
        "expected": relation,
        "observed": observed,
        "passed": bool(passed),
        "description": (
            f"Geometry check for {relation}: {observed}. "
            f"Expected {subject} to be {'left' if relation == 'left_of' else 'right'} of {obj}."
        ),
        "question_id": question_id,
    }


def _geometry_relation_questions(value: Any) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return questions
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source = item.get("source_constraint", {})
        relation = str(source.get("relation") or "").strip() if isinstance(source, Mapping) else ""
        if item.get("category") == "spatial_relation" and relation in GEOMETRY_RELATIONS:
            questions.append(deepcopy(dict(item)))
    return questions


def _object_evidence_questions(
    questions: Sequence[Mapping[str, Any]],
    *,
    require_image_backed_evidence: bool,
) -> list[dict[str, Any]]:
    """Return questions whose VQA answer can be checked with object state."""

    if not require_image_backed_evidence:
        return []
    output: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category") or "")
        if category in {"count", "part_visibility", "action_relation"}:
            output.append(deepcopy(dict(item)))
    return output


def _relation_targets(relations: Sequence[Mapping[str, Any]]) -> list[str]:
    targets: list[str] = []
    for question in relations:
        source = question.get("source_constraint", {})
        if not isinstance(source, Mapping):
            continue
        for key in ("subject", "object"):
            value = str(source.get(key) or "").strip()
            if value and value not in targets:
                targets.append(value)
    return targets


def _evidence_targets(questions: Sequence[Mapping[str, Any]]) -> list[str]:
    targets: list[str] = []
    for question in questions:
        source = question.get("source_constraint", {})
        if not isinstance(source, Mapping):
            continue
        for key in ("subject", "object", "parent", "part"):
            value = str(source.get(key) or "").strip()
            if value and value not in targets:
                targets.append(value)
    return targets


def _count_evidence_check(
    question: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    source = question.get("source_constraint", {})
    if not isinstance(source, Mapping):
        return None
    target = str(source.get("object") or "").strip()
    expected = _coerce_int(source.get("count", question.get("expected_answer")), default=None)
    if not target or expected is None:
        return None
    target_state = _find_object_state(objects, target)
    if not target_state:
        return None
    observed_count = _coerce_int(target_state.get("count"), default=None)
    if observed_count is None:
        return None
    return _object_evidence_check(
        question,
        target=target,
        expected=str(expected),
        observed=str(observed_count),
        passed=observed_count == expected,
        description=(
            f"Object-state count for {target}: expected {expected}, "
            f"observed {observed_count}."
        ),
        error_type="wrong_count",
    )


def _part_evidence_check(
    question: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    source = question.get("source_constraint", {})
    if not isinstance(source, Mapping):
        return None
    parent = str(source.get("parent") or source.get("object") or "").strip()
    part = str(source.get("name") or source.get("part") or "").strip()
    if not parent or not part:
        return None
    parent_state = _find_object_state(objects, parent)
    if parent_state is None:
        return None
    part_state = _find_named_region(
        parent_state.get("parts", []),
        part,
    )
    passed = part_state is not None
    observed = "visible_part_bbox" if passed else "part_bbox_unavailable"
    return _object_evidence_check(
        question,
        target=part,
        expected="visible",
        observed=observed,
        passed=passed,
        description=(
            f"Object-state part evidence for {part}: "
            f"{'visible with bbox' if passed else 'no visible part bbox was returned'}."
        ),
        error_type="missing_object",
    )


def _action_relation_evidence_check(
    question: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    source = question.get("source_constraint", {})
    if not isinstance(source, Mapping):
        return None
    subject = str(source.get("subject") or "").strip()
    obj = str(source.get("object") or "").strip()
    part = str(source.get("part") or "").strip()
    action = str(source.get("action") or source.get("relation") or "relation").strip()
    if not _requires_contact_evidence(action):
        return None
    if not subject and not obj:
        return None
    subject_state = _find_object_state(objects, subject) if subject else None
    object_state = _find_object_state(objects, obj) if obj else None
    part_state = _find_named_region((object_state or {}).get("parts", []), part) if part else None
    contact_state = _find_contact_region(objects, subject, obj, part, action)
    missing: list[str] = []
    if subject and subject_state is None:
        return None
    if obj and object_state is None:
        return None
    if part and part_state is None:
        missing.append(f"part bbox: {part}")
    if contact_state is None:
        missing.append("visible contact/relation region")
    passed = not missing
    target = ":".join(part for part in (subject, obj, action) if part)
    return _object_evidence_check(
        question,
        target=target or str(question.get("id") or "action_relation"),
        expected="visible_contact_evidence",
        observed="visible_contact_evidence" if passed else "; ".join(missing),
        passed=passed,
        description=(
            f"Object-state action/relation evidence for {target or action}: "
            + ("visible contact region found." if passed else "missing " + "; ".join(missing) + ".")
        ),
        error_type="wrong_relation",
    )


def _requires_contact_evidence(action: str) -> bool:
    return str(action or "").strip().lower() in CONTACT_REQUIRED_ACTIONS


def _object_evidence_check(
    question: Mapping[str, Any],
    *,
    target: str,
    expected: str,
    observed: str,
    passed: bool,
    description: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "type": "object_state_evidence",
        "category": str(question.get("category") or "object_evidence"),
        "target": target,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "description": description,
        "question_id": str(question.get("id") or ""),
        "error_type": error_type,
        "evidence_source": "object_part_state_evidence",
    }


def _find_named_region(
    regions: Any,
    target: str,
) -> dict[str, Any] | None:
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        return None
    target_norm = _normalize_name(target)
    target_tokens = set(target_norm.split())
    for item in regions:
        if not isinstance(item, Mapping):
            continue
        if item.get("visible") is False:
            continue
        name = _normalize_name(item.get("name"))
        if not name:
            continue
        name_tokens = set(name.split())
        if name == target_norm or name in target_norm or target_norm in name:
            return deepcopy(dict(item))
        if target_tokens and name_tokens and target_tokens.issubset(name_tokens | target_tokens):
            if any(token in name_tokens for token in target_tokens):
                return deepcopy(dict(item))
    return None


def _find_contact_region(
    objects: Sequence[Mapping[str, Any]],
    subject: str,
    obj: str,
    part: str,
    action: str,
) -> dict[str, Any] | None:
    terms = {
        token
        for value in (subject, obj, part, action)
        for token in _normalize_name(value).split()
        if token
    }
    for object_item in objects:
        if not isinstance(object_item, Mapping):
            continue
        for region in _list_records(object_item.get("contact_regions")):
            if region.get("visible") is False:
                continue
            if region.get("bbox") is None:
                continue
            text = _normalize_name(
                " ".join(
                    [
                        str(region.get("name") or ""),
                        str(region.get("evidence") or ""),
                    ]
                )
            )
            region_terms = set(text.split())
            if not terms or terms & region_terms:
                return deepcopy(dict(region))
    return None


def _find_object_state(objects: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        return None
    target_norm = _normalize_name(target)
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        if item.get("visible") is False:
            continue
        name = _normalize_name(item.get("name"))
        if not name:
            continue
        if name == target_norm or name in target_norm or target_norm in name:
            return deepcopy(dict(item))
    return None


def _parse_object_item(
    item: Any,
    *,
    image_size: tuple[int, int] | None,
) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    name = str(item.get("name") or item.get("target") or item.get("object") or "").strip()
    if not name:
        return None
    visible = _to_bool(item.get("visible", item.get("found", True)))
    bbox = _coerce_bbox(item.get("bbox"), image_size=image_size)
    if not visible:
        bbox = None
    if visible and bbox is None:
        visible = False
    attributes = _parse_attributes(item)
    parts = _parse_named_regions(item.get("parts"), image_size=image_size)
    contact_regions = _parse_named_regions(
        item.get("contact_regions", item.get("contacts", item.get("relation_regions"))),
        image_size=image_size,
    )
    return {
        "name": name,
        "visible": visible,
        "bbox": bbox,
        "mask_path": str(item.get("mask_path") or item.get("mask") or "") or None,
        "attributes": attributes,
        "parts": parts,
        "contact_regions": contact_regions,
        "protected": _to_bool(item.get("protected", False)),
        "count": _coerce_int(item.get("count"), default=None),
        "confidence": _coerce_float(item.get("confidence", item.get("score")), 0.0),
        "evidence": str(item.get("evidence") or item.get("reason") or ""),
    }


def _parse_attributes(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("attributes", {})
    attributes = deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    for key in ("color", "material", "shape", "state"):
        value = item.get(key)
        if value is not None and key not in attributes:
            attributes[key] = value
    return attributes


def _parse_named_regions(
    value: Any,
    *,
    image_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = [
            {"name": key, **(item if isinstance(item, Mapping) else {"bbox": item})}
            for key, item in value.items()
        ]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    regions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("part") or item.get("target") or "").strip()
        if not name:
            continue
        visible = _to_bool(item.get("visible", item.get("found", True)))
        bbox = _coerce_bbox(item.get("bbox"), image_size=image_size)
        if not visible:
            bbox = None
        regions.append(
            {
                "name": name,
                "visible": visible and bbox is not None,
                "bbox": bbox,
                "mask_path": str(item.get("mask_path") or item.get("mask") or "") or None,
                "confidence": _coerce_float(item.get("confidence", item.get("score")), 0.0),
                "evidence": str(item.get("evidence") or item.get("reason") or ""),
            }
        )
    return regions


def _attach_evidence_chain(output: dict[str, Any]) -> None:
    evidence_chain = build_evidence_chain(
        output.get("questions", []),
        output.get("answers", []),
        object_state=output.get("object_state"),
        geometry=output.get("geometry_verification"),
        object_evidence=output.get("object_evidence_verification"),
    )
    output["evidence_chain"] = evidence_chain
    check = output.get("constraint_check")
    if isinstance(check, Mapping):
        updated = deepcopy(dict(check))
        updated["evidence_chain"] = evidence_chain
        output["constraint_check"] = updated


def _question_targets(question: Mapping[str, Any]) -> list[str]:
    source = question.get("source_constraint", {})
    if not isinstance(source, Mapping):
        source = {}
    targets: list[str] = []
    for key in ("object", "subject", "parent"):
        value = str(source.get(key) or "").strip()
        if value and value not in targets:
            targets.append(value)
    part = str(source.get("part") or "").strip()
    if part and part not in targets:
        targets.append(part)
    return targets


def _think_for_question(question: Mapping[str, Any]) -> str:
    category = str(question.get("category") or "")
    source = question.get("source_constraint", {})
    target = ""
    if isinstance(source, Mapping):
        target = str(
            source.get("part")
            or source.get("object")
            or source.get("subject")
            or source.get("phrase")
            or ""
        )
    if category == "entity_existence":
        return f"Need to verify whether the required object is visible: {target}."
    if category == "count":
        return f"Need to count distinct visible instances of: {target}."
    if category == "color_binding":
        return f"Need to verify the visible attribute binding for: {target}."
    if category == "part_visibility":
        return f"Need to inspect the requested object part: {target}."
    if category in {"action_relation", "spatial_relation"}:
        return "Need to verify the user-requested action or relation from visible evidence."
    return "Need to verify this user-grounded visual constraint."


def _evidence_type_for_question(question: Mapping[str, Any]) -> str:
    category = str(question.get("category") or "")
    if category == "count":
        return "distinct_instance_count"
    if category == "color_binding":
        return "object_attribute"
    if category == "part_visibility":
        return "object_part"
    if category == "spatial_relation":
        return "bbox_geometry"
    if category == "action_relation":
        return "part_contact_or_pose"
    if category == "entity_existence":
        return "object_presence"
    return "visual_evidence"


def _verify_record(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    geometry_check: Mapping[str, Any],
    evidence_check: Mapping[str, Any],
) -> dict[str, Any]:
    if geometry_check:
        return {
            "expected": geometry_check.get("expected", question.get("expected_answer")),
            "observed": geometry_check.get("observed"),
            "passed": geometry_check.get("passed") is True,
            "evidence_source": "object_state_geometry",
            "description": geometry_check.get("description", ""),
        }
    if evidence_check:
        return {
            "expected": evidence_check.get("expected", question.get("expected_answer")),
            "observed": evidence_check.get("observed"),
            "passed": evidence_check.get("passed") is True,
            "evidence_source": "object_part_state_evidence",
            "description": evidence_check.get("description", ""),
        }
    return {
        "expected": question.get("expected_answer"),
        "observed": answer.get("normalized_answer", answer.get("answer", "uncertain")),
        "passed": answer.get("passed") is True,
        "evidence_source": "vqa_answer",
        "description": answer.get("evidence", ""),
    }


def _evidence_status(
    answer: Mapping[str, Any],
    geometry_check: Mapping[str, Any],
    evidence_check: Mapping[str, Any],
) -> str:
    if answer.get("blocked_by"):
        return "blocked"
    if geometry_check:
        return "passed" if geometry_check.get("passed") is True else "failed"
    if evidence_check:
        return "passed" if evidence_check.get("passed") is True else "failed"
    observed = str(answer.get("normalized_answer", answer.get("answer", ""))).strip().lower()
    if observed in {"uncertain", "blocked"}:
        return "uncertain" if observed == "uncertain" else "blocked"
    if answer.get("passed") is True:
        return "passed"
    return "failed"


def _coerce_bbox(value: Any, *, image_size: tuple[int, int] | None) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, width, height = [int(round(float(part))) for part in value]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if image_size is None:
        return [x, y, width, height]
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return [x, y, width, height]
    x0 = min(max(0, x), image_width - 1)
    y0 = min(max(0, y), image_height - 1)
    x1 = min(max(x0 + 1, x + width), image_width)
    y1 = min(max(y0 + 1, y + height), image_height)
    return [x0, y0, x1 - x0, y1 - y0]


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    x, y, width, height = [float(part) for part in bbox]
    return x + width / 2.0, y + height / 2.0


def _bbox_width(bbox: Any) -> float:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return 0.0
    try:
        return float(bbox[2])
    except (TypeError, ValueError):
        return 0.0


def _update_summary_with_geometry(
    summary: dict[str, Any],
    checks: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> None:
    passed_constraints = list(summary.get("passed_constraints", []))
    failed_constraints = list(summary.get("failed_constraints", []))
    for check in checks:
        question_id = str(check.get("question_id") or "")
        if not question_id:
            continue
        if check.get("passed") is True and question_id not in passed_constraints:
            passed_constraints.append(question_id)
        elif check.get("passed") is False and question_id not in failed_constraints:
            failed_constraints.append(question_id)
    hard_failures = int(summary.get("hard_failures", 0) or 0) + len(errors)
    summary["passed_constraints"] = passed_constraints
    summary["failed_constraints"] = failed_constraints
    summary["hard_failures"] = hard_failures
    if errors:
        summary["passed"] = False
        summary["score"] = min(_coerce_float(summary.get("score"), 1.0), 0.755)


def _list_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _dedupe_errors(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for error in errors:
        key = (
            str(error.get("type") or ""),
            str(error.get("prompt_span") or ""),
            str(error.get("question_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(deepcopy(dict(error)))
    return result


def _extract_json(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _image_size_or_none(image_path: str | Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        return None


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _coerce_float(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else 1.0
    return max(0.0, min(score, 1.0))


def _coerce_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "visible", "found"}
