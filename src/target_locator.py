"""VLM target object/part localization for local repair masks."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .clients import VLMClient


def locate_target_region(
    vlm: VLMClient,
    image_path: str | Path,
    *,
    user_prompt: str,
    prompt: str,
    target_name: str,
    target_region: str = "full",
    repair_goal: str = "",
    critique: Mapping[str, Any] | None = None,
    layout_context: Mapping[str, Any] | None = None,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Ask a VLM to locate the visible target object/part bbox in image pixels."""

    image_path = str(image_path)
    image_size = _image_size(image_path)
    request = build_target_region_localization_request(
        user_prompt=user_prompt,
        prompt=prompt,
        image_path=image_path,
        image_size=image_size,
        target_name=target_name,
        target_region=target_region,
        repair_goal=repair_goal,
        critique=critique or {},
        layout_context=layout_context,
    )
    diagnostics: dict[str, Any] = {
        "method": "vlm_target_region_locator",
        "image_size": [image_size[0], image_size[1]],
        "target_name": target_name,
        "target_region": target_region,
        "request": request,
    }
    try:
        raw_response = vlm.vision(request, [image_path])
    except Exception as exc:
        diagnostics.update({"found": False, "error": str(exc)})
        return None, diagnostics
    parsed = parse_target_region_localization_response(raw_response, image_size=image_size)
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


def build_target_region_localization_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    image_size: tuple[int, int],
    target_name: str,
    target_region: str = "full",
    repair_goal: str = "",
    critique: Mapping[str, Any] | None = None,
    layout_context: Mapping[str, Any] | None = None,
) -> str:
    """Build a strict VLM prompt for target object/part bbox localization."""

    schema = {
        "found": True,
        "bbox": [0, 0, 1, 1],
        "confidence": 0.0,
        "target_visible": True,
        "reason": "short visual evidence",
    }
    return "\n".join(
        [
            "You are locating a local image-edit mask on the already generated image.",
            "Use the visible image as the source of truth. Planned layout is only a weak hint.",
            "Return the tightest practical rectangular bbox in current image pixel coordinates [x, y, width, height].",
            "The bbox must cover the requested target object or object part, not similarly colored background.",
            "If the target part is requested, localize that part only; otherwise localize the visible object.",
            "Avoid unrelated objects, people, faces, background, and similarly colored regions.",
            "If the target is not visible, set found=false and use bbox=null.",
            "Return exactly one JSON object.",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Image size: {image_size[0]}x{image_size[1]}",
            f"Target object/name: {target_name}",
            f"Target part/region: {target_region}",
            f"Repair goal: {repair_goal}",
            f"Original user prompt: {user_prompt}",
            f"Current generation prompt: {_truncate_text(prompt, 1200)}",
            f"Previous critique summary: {json.dumps(_critique_summary(critique or {}, target_name=target_name), ensure_ascii=False, sort_keys=True)}",
            f"Optional planned layout summary, lower priority than image: {json.dumps(_layout_summary(layout_context), ensure_ascii=False, sort_keys=True)}",
            f"Image path: {image_path}",
        ]
    )


def parse_target_region_localization_response(
    response: str,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Parse target bbox JSON with tolerant field names."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")
    data = _extract_json_object(response) or {}
    warnings: list[str] = []
    bbox_value = _first_present(
        data,
        ("bbox", "target_bbox", "object_bbox", "part_bbox", "region_bbox", "mask_bbox"),
    )
    if bbox_value is None and isinstance(data.get("region"), Mapping):
        bbox_value = _first_present(
            data["region"],
            ("bbox", "target_bbox", "object_bbox", "part_bbox", "region_bbox", "mask_bbox"),
        )
    bbox = _coerce_bbox(bbox_value, image_size=image_size)
    if bbox is None:
        warnings.append("missing_or_invalid_bbox")
    confidence = _normalize_score(data.get("confidence", data.get("score", 0.0))) or 0.0
    found = data.get("found", data.get("visible", data.get("target_visible")))
    if found is None:
        found = bbox is not None
    found = _to_bool(found) and bbox is not None
    reason = str(data.get("reason") or data.get("rationale") or data.get("description") or "").strip()
    return {
        "found": found,
        "bbox": bbox,
        "confidence": round(float(confidence), 6),
        "reason": reason,
        "warnings": warnings,
    }


def layout_with_target_bbox(
    layout_context: Mapping[str, Any],
    target_name: str,
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Return a layout copy whose matching target object bbox uses image coordinates."""

    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        raise TypeError("layout_context must contain a layout mapping")
    updated = {"layout": json.loads(json.dumps(layout))}
    inner = updated["layout"]
    inner["canvas_size"] = [int(image_size[0]), int(image_size[1])]
    objects = inner.get("objects", [])
    if not isinstance(objects, list):
        return updated
    target = target_name.strip().lower()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip().lower()
        if target and (target in name or name in target):
            obj["bbox"] = [int(value) for value in bbox]
            obj["bbox_source"] = "vlm_target_region_locator"
            return updated
    objects.append(
        {
            "name": target_name,
            "bbox": [int(value) for value in bbox],
            "bbox_source": "vlm_target_region_locator",
        }
    )
    return updated


def _layout_summary(layout_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not layout_context:
        return {}
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return {}
    objects = layout.get("objects", [])
    summary_objects: list[dict[str, Any]] = []
    if isinstance(objects, Sequence):
        for obj in objects:
            if not isinstance(obj, Mapping):
                continue
            summary_objects.append(
                {
                    "name": str(obj.get("name", "")),
                    "bbox": list(obj.get("bbox", []))
                    if isinstance(obj.get("bbox"), Sequence)
                    else [],
                    "description": str(obj.get("description", "")),
                }
            )
    return {
        "canvas_size": list(layout.get("canvas_size", []))
        if isinstance(layout.get("canvas_size"), Sequence)
        else [],
        "objects": summary_objects[:12],
    }


def _strip_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"raw_response", "prompt", "request"}:
            continue
        try:
            json.dumps(item)
        except TypeError:
            output[key] = str(item)
        else:
            output[key] = item
    return output


def _critique_summary(
    value: Mapping[str, Any],
    *,
    target_name: str,
) -> dict[str, Any]:
    target = str(target_name or "").strip().lower()
    errors: list[dict[str, Any]] = []
    for item in _error_records(value.get("errors", [])):
        text = " ".join(
            str(item.get(key) or "")
            for key in ("target", "prompt_span", "evidence", "description", "question_id")
        ).lower()
        if target and target not in text and target.split()[-1] not in text:
            continue
        errors.append(
            {
                "type": item.get("type"),
                "target": item.get("target") or item.get("prompt_span"),
                "evidence": _truncate_text(
                    str(item.get("evidence") or item.get("description") or ""),
                    500,
                ),
                "question_id": item.get("question_id"),
            }
        )
    check = value.get("constraint_check")
    if isinstance(check, Mapping):
        for item in _error_records(check.get("errors", [])):
            text = " ".join(
                str(item.get(key) or "")
                for key in ("target", "prompt_span", "evidence", "description", "question_id")
            ).lower()
            if target and target not in text and target.split()[-1] not in text:
                continue
            errors.append(
                {
                    "type": item.get("type"),
                    "target": item.get("target") or item.get("prompt_span"),
                    "evidence": _truncate_text(
                        str(item.get("evidence") or item.get("description") or ""),
                        500,
                    ),
                    "question_id": item.get("question_id"),
                }
            )
    return {
        "score": value.get("score"),
        "errors": errors[:8],
        "revision_hint": _truncate_text(str(value.get("revision_hint") or ""), 500),
    }


def _error_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _image_size(image_path: str | Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return int(image.size[0]), int(image.size[1])


def _coerce_bbox(value: Any, *, image_size: tuple[int, int]) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, width, height = [int(round(float(part))) for part in value]
    except (TypeError, ValueError):
        return None
    image_width, image_height = image_size
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    x0 = min(max(0, x), image_width - 1)
    y0 = min(max(0, y), image_height - 1)
    x1 = min(max(x0 + 1, x + width), image_width)
    y1 = min(max(y0 + 1, y + height), image_height)
    return [x0, y0, x1 - x0, y1 - y0]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fenced)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _first_present(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "visible", "found"}
