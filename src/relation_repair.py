"""Relation/action local repair tools for user-grounded contact failures."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .clients import VLMClient
from .local_editor import (
    InpaintEditor,
    InpaintRegion,
    MockInpaintEditor,
    expand_bbox,
    scale_bbox,
)
from .prompt_constraints import PromptConstraints


CONTACT_ERROR_KEYWORDS = (
    "grip",
    "gripping",
    "hold",
    "holding",
    "handle",
    "contact",
    "touch",
    "touching",
    "connected",
    "connection",
    "detached",
    "floating",
    "hidden hand",
    "not clearly",
)

DEFAULT_RELATION_NEGATIVE_PROMPT = (
    "detached handle, floating umbrella, hidden hand, no contact, broken handle, "
    "extra stick, extra umbrella, missing grip, changed robot color, changed umbrella color, "
    "changed face, changed head, changed identity, new character, distorted body"
)


class RelationActionRepairer:
    """Plan, edit, and VLM-verify local repairs for action/contact failures."""

    def __init__(
        self,
        vlm: VLMClient,
        editor: InpaintEditor | None = None,
        *,
        candidates: int = 3,
        pass_threshold: float = 0.82,
        use_image_grounded_region: bool = True,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self.vlm = vlm
        self.editor = editor or MockInpaintEditor(prefix="relation_inpaint")
        self.candidates = int(candidates)
        self.pass_threshold = float(pass_threshold)
        self.use_image_grounded_region = bool(use_image_grounded_region)

    def should_repair(
        self,
        user_prompt: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints | Mapping[str, Any] | None = None,
    ) -> bool:
        """Return true when feedback points to a user-grounded contact failure."""

        return should_trigger_relation_repair(
            user_prompt,
            critique,
            constraints=constraints,
        )

    def repair(
        self,
        *,
        user_prompt: str,
        prompt: str,
        image_path: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints | Mapping[str, Any] | None,
        output_dir: str | Path,
        layout_context: Mapping[str, Any] | None = None,
        round_index: int = 0,
    ) -> dict[str, Any]:
        """Run local relation repair and return a JSON-compatible record."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.should_repair(user_prompt, critique, constraints):
            return {
                "type": "relation_action_repair",
                "round": int(round_index),
                "accepted": False,
                "skipped": True,
                "reason": "no user-grounded relation/action contact failure detected",
            }

        visual_bbox, visual_diagnostics = (None, None)
        if self.use_image_grounded_region:
            visual_bbox, visual_diagnostics = locate_relation_repair_region(
                self.vlm,
                image_path,
                user_prompt=user_prompt,
                prompt=prompt,
                critique=critique,
                layout_context=layout_context,
            )

        region, diagnostics = plan_relation_repair_region(
            image_path,
            layout_context,
            user_prompt=user_prompt,
            prompt=prompt,
            critique=critique,
            constraints=constraints,
            visual_bbox=visual_bbox,
            visual_diagnostics=visual_diagnostics,
        )
        candidates: list[dict[str, Any]] = []
        for index in range(self.candidates):
            edit_result = self.editor.edit(
                image_path,
                _candidate_region(region, index),
                output_dir,
            )
            verification = self.verify(
                user_prompt=user_prompt,
                prompt=prompt,
                image_path=str(edit_result["edited_image"]),
                critique=critique,
            )
            candidates.append(
                {
                    "index": index,
                    "edited_image": str(edit_result["edited_image"]),
                    "mask_path": str(edit_result.get("mask_path", "")),
                    "edit_result": edit_result,
                    "verification": verification,
                    "score": verification["score"],
                    "passed": verification["passed"],
                }
            )

        selected = max(candidates, key=lambda item: float(item["score"]))
        evidence_quality = (
            selected.get("verification", {}).get("evidence_quality", {})
            if isinstance(selected.get("verification"), Mapping)
            else {}
        )
        accepted = (
            (bool(selected["passed"]) or float(selected["score"]) >= self.pass_threshold)
            and evidence_quality.get("passed", True) is True
        )
        return {
            "type": "relation_action_repair",
            "round": int(round_index),
            "accepted": accepted,
            "source_image": image_path,
            "edited_image": selected["edited_image"],
            "selected_index": int(selected["index"]),
            "score": float(selected["score"]),
            "pass_threshold": self.pass_threshold,
            "region": region.to_dict(),
            "detection": diagnostics,
            "candidates": candidates,
            "repair_plan": {
                "prompt": region.prompt,
                "negative_prompt": region.negative_prompt,
                "strategy": "local_contact_inpaint",
            },
        }

    def verify(
        self,
        *,
        user_prompt: str,
        prompt: str,
        image_path: str,
        critique: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Ask the VLM whether the repaired image satisfies the contact relation."""

        request = build_relation_verification_request(
            user_prompt=user_prompt,
            prompt=prompt,
            image_path=image_path,
            critique=critique,
        )
        raw_response = self.vlm.vision(request, [image_path])
        parsed = parse_relation_verification_response(
            raw_response,
            pass_threshold=self.pass_threshold,
        )
        parsed.update(
            {
                "image_path": image_path,
                "request": request,
                "raw_response": raw_response,
            }
        )
        return parsed


def should_trigger_relation_repair(
    user_prompt: str,
    critique: Mapping[str, Any],
    *,
    constraints: PromptConstraints | Mapping[str, Any] | None = None,
) -> bool:
    """Detect whether critique text indicates a repairable action/relation failure."""

    del constraints
    user_prompt = _clean_text(user_prompt, "user_prompt")
    if not _user_prompt_requests_contact(user_prompt):
        return False
    errors = critique.get("errors", []) or []
    if isinstance(errors, Mapping):
        errors = [errors]
    for item in errors if isinstance(errors, Sequence) else []:
        if isinstance(item, Mapping):
            error_type = str(item.get("type", "")).lower()
            text = _record_text(item)
            if error_type in {"wrong_relation", "relation_error", "action_error", "contact_error"}:
                return _has_contact_keywords(text) or True
            if _has_contact_keywords(text):
                return True
        elif _has_contact_keywords(str(item)):
            return True
    return _has_contact_keywords(str(critique.get("revision_hint", "")))


def plan_relation_repair_region(
    image_path: str | Path,
    layout_context: Mapping[str, Any] | None,
    *,
    user_prompt: str,
    prompt: str,
    critique: Mapping[str, Any],
    constraints: PromptConstraints | Mapping[str, Any] | None = None,
    visual_bbox: Sequence[int] | None = None,
    visual_diagnostics: Mapping[str, Any] | None = None,
) -> tuple[InpaintRegion, dict[str, Any]]:
    """Build an inpaint region around the hand/handle contact area."""

    image_size = _image_size(image_path)
    layout_bbox, layout_diagnostics = _relation_bbox_from_layout(
        layout_context,
        image_size=image_size,
    )
    target_bbox = _coerce_bbox(visual_bbox, image_size=image_size) if visual_bbox else None
    if target_bbox is not None:
        target_bbox = expand_bbox(target_bbox, image_size=image_size, expand=0.08)
        diagnostics = {
            "method": "image_grounded_vlm_contact_bbox",
            "image_size": [image_size[0], image_size[1]],
            "detected_bbox": target_bbox,
            "visual_locator": _strip_locator_runtime(visual_diagnostics or {}),
            "layout_fallback": layout_diagnostics,
            "layout_fallback_bbox": layout_bbox,
            "targeting": "actual_image_hand_handle_contact",
        }
    else:
        target_bbox = layout_bbox
        diagnostics = layout_diagnostics
        if visual_diagnostics:
            diagnostics = dict(diagnostics)
            diagnostics["visual_locator"] = _strip_locator_runtime(visual_diagnostics)
    if target_bbox is None:
        width, height = image_size
        target_bbox = [
            max(0, int(width * 0.28)),
            max(0, int(height * 0.30)),
            max(1, int(width * 0.44)),
            max(1, int(height * 0.52)),
        ]
        diagnostics = {
            "method": "fallback_center_contact_region",
            "image_size": [width, height],
            "reason": "layout missing or no robot/umbrella objects found",
        }
    prompt_text = build_relation_repair_prompt(
        user_prompt,
        prompt=prompt,
        critique=critique,
        constraints=constraints,
    )
    region = InpaintRegion(
        name="relation_action_contact",
        bbox=target_bbox,
        prompt=prompt_text,
        negative_prompt=DEFAULT_RELATION_NEGATIVE_PROMPT,
        reason="repair visible contact between subject hand/claw and object handle",
        canvas_size=[image_size[0], image_size[1]],
    )
    return region, diagnostics


def locate_relation_repair_region(
    vlm: VLMClient,
    image_path: str | Path,
    *,
    user_prompt: str,
    prompt: str,
    critique: Mapping[str, Any],
    layout_context: Mapping[str, Any] | None = None,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Ask the VLM to locate the actual hand/handle contact bbox in the image."""

    image_path = _clean_text(str(image_path), "image_path")
    image_size = _image_size(image_path)
    request = build_relation_region_localization_request(
        user_prompt=user_prompt,
        prompt=prompt,
        image_path=image_path,
        image_size=image_size,
        critique=critique,
        layout_context=layout_context,
    )
    diagnostics: dict[str, Any] = {
        "method": "vlm_contact_bbox_locator",
        "image_size": [image_size[0], image_size[1]],
        "request": request,
    }
    try:
        raw_response = vlm.vision(request, [image_path])
    except Exception as exc:
        diagnostics.update({"found": False, "error": str(exc)})
        return None, diagnostics
    parsed = parse_relation_region_localization_response(
        raw_response,
        image_size=image_size,
    )
    diagnostics.update(
        {
            "raw_response": raw_response,
            "found": parsed["found"],
            "bbox": parsed.get("bbox"),
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason", ""),
            "parser_warnings": parsed.get("warnings", []),
        }
    )
    if not parsed["found"]:
        return None, diagnostics
    return parsed["bbox"], diagnostics


def build_relation_repair_prompt(
    user_prompt: str,
    *,
    prompt: str,
    critique: Mapping[str, Any],
    constraints: PromptConstraints | Mapping[str, Any] | None = None,
) -> str:
    """Build a compact inpaint prompt focused on visible physical contact."""

    del constraints
    colors = _simple_color_phrases(user_prompt)
    color_text = ", ".join(colors[:3])
    relation = _contact_phrase(user_prompt)
    base = [
        "small local edit only inside the contact mask",
        relation,
        "visible physical contact",
        "the handle continues upward into the umbrella",
        "preserve the existing face, head, body shape, colors, scene, scale, lighting, rain, and camera view",
    ]
    if color_text:
        base.insert(1, color_text)
    hint = str(critique.get("revision_hint", "")).strip()
    if hint:
        base.append(f"fix: {hint}")
    compact_prompt = ", ".join(base)
    return _truncate_words(compact_prompt, max_words=42)


def build_relation_region_localization_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    image_size: tuple[int, int],
    critique: Mapping[str, Any],
    layout_context: Mapping[str, Any] | None = None,
) -> str:
    """Build a VLM request for locating the actual edit region in the image."""

    schema = {
        "found": True,
        "bbox": [0, 0, 1, 1],
        "confidence": 0.0,
        "reason": "short visual evidence",
    }
    layout_summary = _layout_summary(layout_context)
    return "\n".join(
        [
            "You are locating a local inpaint mask on the already generated image.",
            "Use the visible image content as the source of truth. Do not trust planned layout boxes if the generated objects moved.",
            "Find the smallest practical rectangular bbox, in current image pixel coordinates [x, y, width, height],",
            "covering the subject hand/claw and the object handle/stem contact area that should be repaired.",
            "The bbox should cover the hand/claw, nearby handle/stem, and a little surrounding context.",
            "Avoid the subject face/head, the umbrella canopy, unrelated body parts, and background when possible.",
            "If the hand or handle is partially missing, choose the best visible location where contact should occur.",
            "Return exactly one JSON object.",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Image size: {image_size[0]}x{image_size[1]}",
            f"Original user prompt: {_truncate_text(user_prompt, max_chars=500)}",
            f"Current generation prompt: {_truncate_text(prompt, max_chars=700)}",
            f"Previous critique summary: {json.dumps(_compact_relation_critique(critique), ensure_ascii=False, sort_keys=True)}",
            f"Optional planned layout summary, lower priority than image: {json.dumps(layout_summary, ensure_ascii=False, sort_keys=True)}",
            f"Image path: {image_path}",
        ]
    )


def parse_relation_region_localization_response(
    response: str,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Parse a VLM bbox locator response into a compact JSON-compatible record."""

    response = _clean_text(response, "response")
    data = _extract_json_object(response) or {}
    warnings: list[str] = []
    bbox_value = _first_present(
        data,
        ("bbox", "contact_bbox", "repair_bbox", "region_bbox", "mask_bbox"),
    )
    if bbox_value is None and isinstance(data.get("region"), Mapping):
        bbox_value = _first_present(
            data["region"],
            ("bbox", "contact_bbox", "repair_bbox", "region_bbox", "mask_bbox"),
        )
    bbox = _coerce_bbox(bbox_value, image_size=image_size)
    if bbox is None:
        warnings.append("missing_or_invalid_bbox")
    confidence = _normalize_score(data.get("confidence", data.get("score", 0.0))) or 0.0
    found = data.get("found", data.get("visible", data.get("has_region")))
    if found is None:
        found = bbox is not None
    found = _to_bool(found) and bbox is not None
    reason = str(data.get("reason") or data.get("rationale") or data.get("description") or "").strip()
    result: dict[str, Any] = {
        "found": found,
        "bbox": bbox,
        "confidence": round(float(confidence), 6),
        "reason": reason,
        "warnings": warnings,
    }
    return result


def build_relation_verification_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    critique: Mapping[str, Any],
) -> str:
    """Build a strict VLM verification prompt for repaired contact relations."""

    schema = {
        "score": 0.0,
        "passed": False,
        "checks": {
            "handle_visible": False,
            "hand_or_claw_visible": False,
            "visible_grip": False,
            "physical_contact": False,
            "handle_connected_to_umbrella": False,
            "not_merely_near_or_supported": False,
            "user_colors_preserved": False,
        },
        "errors": ["short visible failure"],
        "strengths": ["short visible success"],
        "revision_hint": "short next repair instruction",
    }
    return "\n".join(
        [
            "You are a strict visual verifier for a local relation/action repair.",
            "Judge only the original user intent and visible physical relation.",
            "Do not reward aesthetics if the hand/claw is detached or hidden.",
            "A relation passes only when the hand/claw/contact point and the handle/stem are both visible,",
            "there is a small local contact point, and the pose shows gripping/holding rather than nearby, attached, supported, or occluded contact.",
            "Also verify that user-specified colors and attributes are preserved.",
            "If a required user color changed, passed must be false.",
            "Return exactly one JSON object.",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Original user prompt: {_truncate_text(user_prompt, max_chars=500)}",
            f"Current generation prompt: {_truncate_text(prompt, max_chars=700)}",
            f"Previous critique summary: {json.dumps(_compact_relation_critique(critique), ensure_ascii=False, sort_keys=True)}",
            f"Image path: {image_path}",
            "Questions: Is the subject visibly gripping/holding the handle? "
            "Are the hand/claw and handle both visible? Is there physical contact? "
            "Is the handle connected to the umbrella? Is it more than merely nearby or supported? "
            "Are all user-specified colors preserved?",
        ]
    )


def parse_relation_verification_response(
    response: str,
    *,
    pass_threshold: float = 0.82,
) -> dict[str, Any]:
    """Parse strict relation verification JSON with text fallback."""

    response = _clean_text(response, "response")
    data = _extract_json_object(response) or {}
    score = _normalize_score(data.get("score", data.get("overall_score")))
    if score is None:
        score = _score_from_text(response)
    if score is None:
        checks = _normalize_checks(data.get("checks", {}))
        score = sum(1.0 for value in checks.values() if value) / max(1, len(checks))
    checks = _normalize_checks(data.get("checks", {}))
    evidence_quality = _relation_evidence_quality(checks, response, data)
    passed = bool(data.get("passed", score >= pass_threshold))
    errors = _normalize_string_list(data.get("errors", []))
    if not errors and "errors" not in data:
        errors = _errors_from_text(response)
    if not evidence_quality["passed"]:
        passed = False
        for failure in evidence_quality["failures"]:
            message = str(failure.get("message") or failure.get("type") or "").strip()
            if message and message not in errors:
                errors.append(message)
    strengths = _normalize_string_list(data.get("strengths", []))
    revision_hint = str(data.get("revision_hint") or data.get("suggestion") or "").strip()
    if not revision_hint and errors:
        revision_hint = errors[0]
    if errors and score < pass_threshold:
        passed = False
    return {
        "score": round(float(score), 6),
        "passed": passed,
        "checks": checks,
        "evidence_quality": evidence_quality,
        "errors": errors,
        "strengths": strengths,
        "revision_hint": revision_hint,
    }


def _candidate_region(region: InpaintRegion, index: int) -> InpaintRegion:
    bbox = _candidate_contact_bbox(region.bbox, index, canvas_size=region.canvas_size)
    prompt = region.prompt
    if index == 1:
        prompt += ", tiny hand detail only, preserve face and body"
    elif index >= 2:
        prompt += ", clear claw fingers wrapped around the umbrella handle, preserve identity"
    else:
        prompt += ", preserve identity and object colors"
    return InpaintRegion(
        name=f"{region.name}_{index}",
        bbox=bbox,
        prompt=_truncate_words(prompt, max_words=48),
        negative_prompt=region.negative_prompt,
        reason=region.reason,
        canvas_size=region.canvas_size,
    )


def _candidate_contact_bbox(
    bbox: Sequence[int],
    index: int,
    *,
    canvas_size: Sequence[int] | None,
) -> list[int]:
    image_size = _coerce_image_size(canvas_size)
    x, y, width, height = [int(value) for value in bbox]
    if index == 0:
        return expand_bbox([x, y, width, height], image_size=image_size, expand=0.0)
    if index == 1:
        return expand_bbox(
            [
                x + int(width * 0.10),
                y + int(height * 0.10),
                max(1, int(width * 0.80)),
                max(1, int(height * 0.78)),
            ],
            image_size=image_size,
            expand=0.0,
        )
    return expand_bbox(
        [
            x + int(width * 0.18),
            y + int(height * 0.16),
            max(1, int(width * 0.64)),
            max(1, int(height * 0.68)),
        ],
        image_size=image_size,
        expand=0.0,
    )


def _relation_bbox_from_layout(
    layout_context: Mapping[str, Any] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[list[int] | None, dict[str, Any]]:
    if not layout_context:
        return None, {}
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return None, {"method": "layout_invalid"}
    canvas_size = _canvas_size(layout.get("canvas_size", [1024, 1024]))
    objects = layout.get("objects", [])
    if not isinstance(objects, Sequence):
        return None, {"method": "layout_missing_objects"}
    robot = _find_layout_object(
        objects,
        ("robot", "person", "subject"),
        relation_needles=("hand", "claw", "gripping", "holding"),
    )
    held = _find_layout_object(
        objects,
        ("umbrella", "handle", "stick", "object"),
        relation_needles=("held", "gripped", "gripping", "holding"),
        exclude=robot,
    )
    if robot is None and held is None:
        return None, {"method": "layout_no_relation_targets"}
    boxes = []
    for obj in (robot, held):
        if obj is None:
            continue
        bbox = obj.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            boxes.append(
                scale_bbox(
                    [int(value) for value in bbox],
                    from_size=canvas_size,
                    to_size=image_size,
                )
            )
    if not boxes:
        return None, {"method": "layout_targets_without_bbox"}
    union = _union_bbox(boxes)
    subject_bbox = boxes[0] if robot is not None and boxes else None
    object_bbox = boxes[1] if robot is not None and held is not None and len(boxes) > 1 else None
    if object_bbox is None and held is not None and boxes:
        object_bbox = boxes[0]
    focused = _focus_contact_band(
        union,
        image_size=image_size,
        subject_bbox=subject_bbox,
        object_bbox=object_bbox,
    )
    return focused, {
        "method": "layout_hand_handle_contact_band",
        "image_size": [image_size[0], image_size[1]],
        "layout_canvas_size": [canvas_size[0], canvas_size[1]],
        "robot_object": _object_summary(robot),
        "held_object": _object_summary(held),
        "union_bbox": union,
        "subject_bbox": subject_bbox,
        "object_bbox": object_bbox,
        "detected_bbox": focused,
        "targeting": "subject_upper_body_to_handle_lower_stem",
    }


def _focus_contact_band(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    subject_bbox: Sequence[int] | None = None,
    object_bbox: Sequence[int] | None = None,
) -> list[int]:
    x, y, width, height = [int(value) for value in bbox]
    if height <= 0 or width <= 0:
        return expand_bbox([x, y, width, height], image_size=image_size, expand=0.0)
    if subject_bbox and object_bbox:
        sx, sy, sw, sh = [int(value) for value in subject_bbox]
        ox, oy, ow, oh = [int(value) for value in object_bbox]
        subject_cx = sx + sw // 2
        object_cx = ox + ow // 2
        handle_x = int(round(subject_cx * 0.65 + object_cx * 0.35))
        contact_w = max(32, min(width, int(max(sw, ow) * 0.50)))
        contact_x = handle_x - contact_w // 2
        handle_lower = oy + int(oh * 0.72)
        subject_hand_band = sy + int(sh * 0.55)
        contact_y = int(round(subject_hand_band * 0.82 + handle_lower * 0.18))
        contact_h = max(40, min(height, int((sh + oh) * 0.26)))
        return expand_bbox(
            [contact_x, contact_y, contact_w, contact_h],
            image_size=image_size,
            expand=0.08,
        )
    contact_y = y + int(height * 0.38)
    contact_h = max(1, int(height * 0.42))
    contact_x = x + int(width * 0.28)
    contact_w = max(1, int(width * 0.44))
    return expand_bbox(
        [contact_x, contact_y, contact_w, contact_h],
        image_size=image_size,
        expand=0.06,
    )


def _find_layout_object(
    objects: Sequence[Any],
    needles: Sequence[str],
    *,
    relation_needles: Sequence[str] = (),
    exclude: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    fallback: Mapping[str, Any] | None = None
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        if exclude is not None and obj is exclude:
            continue
        primary_text = " ".join(
            str(obj.get(key, "")) for key in ("name", "description")
        ).lower()
        if any(needle in primary_text for needle in needles):
            return obj
        relation_text = str(obj.get("relations", "")).lower()
        if relation_needles and any(needle in relation_text for needle in relation_needles):
            fallback = fallback or obj
    return fallback


def _union_bbox(boxes: Sequence[Sequence[int]]) -> list[int]:
    x0 = min(int(box[0]) for box in boxes)
    y0 = min(int(box[1]) for box in boxes)
    x1 = max(int(box[0]) + int(box[2]) for box in boxes)
    y1 = max(int(box[1]) + int(box[3]) for box in boxes)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _image_size(image_path: str | Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _user_prompt_requests_contact(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return _has_contact_keywords(lowered)


def _has_contact_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CONTACT_ERROR_KEYWORDS)


def _record_text(record: Mapping[str, Any]) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in ("type", "evidence", "description", "reason", "prompt_span", "observed")
    )


def _simple_color_phrases(text: str) -> list[str]:
    lowered = text.lower()
    phrases: list[str] = []
    for match in re.finditer(
        r"\b(red|blue|green|yellow|black|white|orange|purple|pink)\s+([a-z0-9-]+)",
        lowered,
    ):
        phrase = f"{match.group(1)} {match.group(2)}"
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _contact_phrase(user_prompt: str) -> str:
    lowered = user_prompt.lower()
    if "gripping" in lowered or "grip" in lowered:
        return "the subject hand or claw clearly grips the umbrella handle"
    if "holding" in lowered or "hold" in lowered:
        return "the subject hand or claw clearly holds the object handle"
    return "the subject and target object have clear visible physical contact"


def _normalize_checks(value: Any) -> dict[str, bool]:
    if isinstance(value, Mapping):
        return {str(key): _to_bool(item) for key, item in value.items()}
    if isinstance(value, list):
        result: dict[str, bool] = {}
        for item in value:
            if isinstance(item, Mapping):
                key = str(item.get("name") or item.get("type") or item.get("target") or "")
                if key:
                    result[key] = _to_bool(item.get("passed", item.get("value")))
        return result
    return {}


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _relation_evidence_quality(
    checks: Mapping[str, bool],
    response: str,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require local visual evidence for contact/action claims."""

    lowered_keys = {str(key).lower(): bool(value) for key, value in checks.items()}
    data = data or {}
    evidence_text = " ".join(
        [
            " ".join(_normalize_string_list(data.get("errors", []))),
            " ".join(_normalize_string_list(data.get("strengths", []))),
            str(data.get("revision_hint") or data.get("suggestion") or ""),
        ]
    ).lower()
    has_natural_evidence_text = bool(evidence_text.strip())
    failures: list[dict[str, Any]] = []

    handle_visible = _any_check_true(
        lowered_keys,
        ("handle_visible", "handle visible", "stem_visible", "object_part_visible"),
    ) or (
        _any_check_true(lowered_keys, ("handle_connected_to_umbrella", "handle_connected"))
        and "hidden handle" not in evidence_text
    )
    hand_visible = _any_check_true(
        lowered_keys,
        ("hand_or_claw_visible", "hand visible", "claw_visible", "contact_point_visible"),
    )
    grip_visible = _any_check_true(
        lowered_keys,
        ("visible_grip", "visible grip", "grip_visible", "clearly_gripping"),
    )
    physical_contact = _any_check_true(
        lowered_keys,
        ("physical_contact", "physical contact", "contact", "touching"),
    )
    not_merely_near = _any_check_true(
        lowered_keys,
        (
            "not_merely_near_or_supported",
            "not merely near or supported",
            "not_nearby",
            "not_supported",
        ),
    )
    if has_natural_evidence_text and _contains_weak_contact_language(evidence_text):
        not_merely_near = False

    required = {
        "handle_visible": handle_visible,
        "hand_or_claw_visible": hand_visible,
        "visible_grip": grip_visible,
        "physical_contact": physical_contact,
        "not_merely_near_or_supported": not_merely_near,
    }
    for key, value in required.items():
        if not value:
            failures.append(
                {
                    "type": f"missing_relation_evidence:{key}",
                    "message": (
                        "Relation/action repair lacks required local evidence: "
                        f"{key}"
                    ),
                }
            )
    return {
        "passed": not failures,
        "required_checks": required,
        "failures": failures,
    }


def _any_check_true(checks: Mapping[str, bool], names: Sequence[str]) -> bool:
    needles = [str(name).lower().replace("_", " ") for name in names]
    for key, value in checks.items():
        normalized = str(key).lower().replace("_", " ")
        if any(needle in normalized for needle in needles) and value:
            return True
    return False


def _contains_weak_contact_language(text: str) -> bool:
    weak_terms = (
        "nearby",
        "near ",
        "close to",
        "next to",
        "attached",
        "mounted",
        "supported",
        "resting",
        "occluded",
        "hidden",
        "not clearly",
        "loose",
        "no visible grip",
        "not visible",
    )
    return any(term in text for term in weak_terms)


def _errors_from_text(text: str) -> list[str]:
    lowered = text.lower()
    if "no contact" in lowered or "detached" in lowered:
        return ["The hand/claw is not visibly in physical contact with the handle."]
    if "hidden" in lowered:
        return ["The hand/claw or handle is hidden."]
    return []


def _score_from_text(text: str) -> float | None:
    match = re.search(r"(?:score|overall)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not match:
        return None
    return _normalize_score(match.group(1))


def _normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return min(1.0, max(0.0, score))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "passed", "ok", "1"}
    return bool(value)


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


def _canvas_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or len(value) != 2:
        return 1024, 1024
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        return 1024, 1024
    return width, height


def _coerce_image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or len(value) != 2:
        return 1024, 1024
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return 1024, 1024
    if width <= 0 or height <= 0:
        return 1024, 1024
    return width, height


def _object_summary(obj: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if obj is None:
        return None
    return {
        "name": str(obj.get("name", "")),
        "bbox": list(obj.get("bbox", [])) if isinstance(obj.get("bbox"), Sequence) else [],
    }


def _layout_summary(layout_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(layout_context, Mapping):
        return {}
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return {}
    objects = layout.get("objects", [])
    summarized = []
    if isinstance(objects, Sequence):
        for obj in objects:
            if not isinstance(obj, Mapping):
                continue
            summarized.append(
                {
                    "name": str(obj.get("name", "")),
                    "bbox": list(obj.get("bbox", []))
                    if isinstance(obj.get("bbox"), Sequence)
                    else [],
                    "relations": obj.get("relations", []),
                }
            )
    return {
        "canvas_size": list(layout.get("canvas_size", []))
        if isinstance(layout.get("canvas_size"), Sequence)
        else [],
        "objects": summarized,
    }


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _coerce_bbox(value: Any, *, image_size: tuple[int, int]) -> list[int] | None:
    width, height = image_size
    if isinstance(value, Mapping):
        if all(key in value for key in ("x", "y", "width", "height")):
            value = [value["x"], value["y"], value["width"], value["height"]]
        elif all(key in value for key in ("x", "y", "w", "h")):
            value = [value["x"], value["y"], value["w"], value["h"]]
        elif all(key in value for key in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = [value[key] for key in ("x1", "y1", "x2", "y2")]
            value = [x1, y1, float(x2) - float(x1), float(y2) - float(y1)]
        else:
            return None
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 4:
            value = [float(number) for number in numbers[:4]]
    if not isinstance(value, Sequence) or len(value) < 4:
        return None
    try:
        x, y, box_w, box_h = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0:
        x *= width
        box_w *= width
        y *= height
        box_h *= height
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + box_w))
    y1 = int(round(y + box_h))
    x0 = min(max(0, x0), width - 1)
    y0 = min(max(0, y0), height - 1)
    x1 = min(max(x0 + 1, x1), width)
    y1 = min(max(y0 + 1, y1), height)
    result = [x0, y0, x1 - x0, y1 - y0]
    if result[2] < 4 or result[3] < 4:
        return None
    return result


def _strip_locator_runtime(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    request = result.pop("request", None)
    if request:
        result["request_preview"] = str(request)[:400]
    raw_response = result.pop("raw_response", None)
    if raw_response:
        result["raw_response_preview"] = str(raw_response)[:400]
    return result


def _strip_runtime(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    for key in ("request", "raw_response"):
        result.pop(key, None)
    return result


def _compact_relation_critique(critique: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only short contact-relevant feedback for VLM repair prompts."""

    result: dict[str, Any] = {}
    score = critique.get("score")
    if score is not None:
        result["score"] = score
    hint = _truncate_text(str(critique.get("revision_hint") or ""), max_chars=360)
    if hint:
        result["revision_hint"] = hint
    errors: list[dict[str, Any]] = []
    for item in _iter_relation_error_records(critique):
        text = _record_text(item)
        error_type = str(item.get("type") or item.get("category") or "").lower()
        if not (_has_contact_keywords(text) or "relation" in error_type or "action" in error_type):
            continue
        compact: dict[str, Any] = {}
        for key in (
            "type",
            "category",
            "target",
            "prompt_span",
            "evidence",
            "description",
            "observed",
            "question",
            "answer",
        ):
            value = item.get(key)
            if value is not None and str(value).strip():
                compact[key] = _truncate_text(str(value), max_chars=220)
        if compact:
            errors.append(compact)
        if len(errors) >= 8:
            break
    if errors:
        result["errors"] = errors
    return result


def _iter_relation_error_records(critique: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for value in (critique.get("errors", []), critique.get("checks", [])):
        records.extend(_as_mapping_records(value))
    for key in (
        "constraint_check",
        "evaluation",
        "relation_repair_verification",
        "pre_relation_repair_feedback",
    ):
        nested = critique.get(key)
        if not isinstance(nested, Mapping):
            continue
        records.extend(_as_mapping_records(nested.get("errors", [])))
        records.extend(_as_mapping_records(nested.get("checks", [])))
    return records


def _as_mapping_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _truncate_text(text: str, *, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _truncate_words(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;")


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned
