"""LayerCraft-style layout planning adapter for M5."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from .clients import LLMClient
from .prompt_constraints import extract_constraints


DEFAULT_CANVAS_SIZE = (1024, 1024)


class LayoutPlanner:
    """Plan a background and ordered foreground object boxes through an LLM."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        strict_background: bool = False,
    ) -> None:
        self.llm = llm
        self.strict_background = bool(strict_background)

    def plan(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    ) -> dict[str, Any]:
        """Return a normalized layout plan for a prompt."""

        prompt = _clean_text(prompt, "prompt")
        canvas_size = _validate_canvas_size(canvas_size)
        request = build_layout_request(prompt, canvas_size)
        raw_response = self.llm.text(request)
        layout = parse_layout_response(raw_response, canvas_size=canvas_size)
        normalized = validate_layout(
            layout,
            canvas_size=canvas_size,
            strict_background=self.strict_background,
        )
        normalized["raw_response"] = raw_response
        normalized["request"] = request
        return normalized


def build_layout_request(
    prompt: str,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> str:
    """Build the project-local ChainArchitect-style planning prompt."""

    prompt = _clean_text(prompt, "prompt")
    canvas_size = _validate_canvas_size(canvas_size)
    return "\n".join(
        [
            "You are ChainArchitect for a multimodal text-to-image agent.",
            "Convert the user prompt into an inspectable layout plan.",
            "Do not call tools. Return one JSON object only.",
            "The background must describe only the static environment and camera viewpoint.",
            "Do not put foreground object words, synonyms, or related concepts in the background.",
            "Plan every visible foreground object with a pixel bbox [x, y, width, height].",
            "Use a far-to-near or back-to-front generation order when objects overlap.",
            "Keep object colors, actions, and spatial relations from the original user prompt.",
            "Schema:",
            json.dumps(
                {
                    "canvas_size": [canvas_size[0], canvas_size[1]],
                    "background": {
                        "description": "environment only",
                        "viewpoint": "camera/view/framing",
                    },
                    "objects": [
                        {
                            "name": "object name",
                            "description": "visual description with color/action",
                            "bbox": [0, 0, 256, 256],
                            "order": 1,
                            "relations": ["relation to other objects"],
                            "requires_reference": False,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            f"Prompt: {prompt}",
            f"Canvas size: {canvas_size[0]}x{canvas_size[1]} pixels.",
        ]
    )


def parse_layout_response(
    response: str,
    *,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> dict[str, Any]:
    """Parse a layout JSON response from an LLM."""

    response = _clean_text(response, "response")
    data = _extract_json_object(response)
    if data is None:
        raise ValueError("layout response did not contain a JSON object")
    if isinstance(data.get("layout"), Mapping):
        data = dict(data["layout"])
    if "background_prompt" in data and "background" not in data:
        data["background"] = {
            "description": data["background_prompt"],
            "viewpoint": data.get("viewpoint", "unspecified camera view"),
        }
    data.setdefault("canvas_size", [canvas_size[0], canvas_size[1]])
    return deepcopy(data)


def validate_layout(
    layout: Mapping[str, Any],
    *,
    canvas_size: tuple[int, int] | None = None,
    strict_background: bool = False,
) -> dict[str, Any]:
    """Normalize and validate a LayerCraft-style layout plan."""

    if not isinstance(layout, Mapping):
        raise TypeError("layout must be a mapping")
    canvas_size = _validate_canvas_size(
        canvas_size or _canvas_from_layout(layout) or DEFAULT_CANVAS_SIZE
    )
    background = _normalize_background(layout.get("background"))
    objects = _normalize_objects(layout.get("objects", []), canvas_size)
    background, warnings = _clean_background_against_objects(
        background,
        objects,
        strict_background=strict_background,
    )
    result = {
        "canvas_size": [canvas_size[0], canvas_size[1]],
        "background": background,
        "objects": objects,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def layout_to_prompt_package(
    layout: Mapping[str, Any],
    *,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Convert a validated layout into the M5 prompt package."""

    normalized = validate_layout(layout)
    background = normalized["background"]
    objects = [deepcopy(dict(item)) for item in normalized["objects"]]
    background_prompt = ", ".join(
        item
        for item in (background["description"], background["viewpoint"])
        if item and item != "unspecified camera view"
    )
    layout_lines = [
        f"background: {background_prompt}",
        "foreground objects in generation order:",
    ]
    for obj in objects:
        relations = "; ".join(obj.get("relations", []))
        relation_text = f"; relations: {relations}" if relations else ""
        layout_lines.append(
            f"{obj['order']}. {obj['name']} at bbox {obj['bbox']}: "
            f"{obj['description']}{relation_text}"
        )
    if user_prompt:
        layout_lines.append(f"original user prompt: {_clean_text(user_prompt, 'user_prompt')}")

    package = {
        "canvas_size": normalized["canvas_size"],
        "background_prompt": background_prompt,
        "objects": objects,
        "generation_order": [obj["name"] for obj in objects],
        "layout_prompt": " ".join(layout_lines),
    }
    if "warnings" in normalized:
        package["warnings"] = list(normalized["warnings"])
    return package


def layout_to_enriched_prompt(
    user_prompt: str,
    prompt_package: Mapping[str, Any],
) -> str:
    """Build a plain text prompt from a layout package for non-layout backends."""

    user_prompt = _clean_text(user_prompt, "user_prompt")
    background = _clean_text(prompt_package.get("background_prompt", ""), "background_prompt")
    objects = prompt_package.get("objects", [])
    if not isinstance(objects, list):
        raise TypeError("prompt_package objects must be a list")

    object_parts: list[str] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        bbox = obj.get("bbox", [])
        bbox_text = _bbox_text(bbox)
        position = _bbox_position(bbox, prompt_package.get("canvas_size", [1024, 1024]))
        relations = "; ".join(str(item) for item in obj.get("relations", []))
        relation_text = f", {relations}" if relations else ""
        object_parts.append(
            f"{obj.get('name')}: {obj.get('description')}, box {bbox_text}{relation_text}"
        )
    return _normalize_spaces(
        "; ".join(
            [
                f"original prompt: {user_prompt}",
                f"environment-only background: {background}",
                "foreground layout: " + " | ".join(object_parts),
                "keep each object inside its bbox and preserve listed relations",
            ]
        )
    )


def layout_to_generation_hint(prompt_package: Mapping[str, Any]) -> str:
    """Build a compact generation hint for non-layout image backends."""

    background = _clean_text(prompt_package.get("background_prompt", ""), "background_prompt")
    objects = prompt_package.get("objects", [])
    if not isinstance(objects, list):
        raise TypeError("prompt_package objects must be a list")

    object_hints: list[str] = []
    used_relations: set[str] = set()
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        position = _bbox_position(
            obj.get("bbox", []),
            prompt_package.get("canvas_size", [1024, 1024]),
        )
        name = str(obj.get("name", "")).strip()
        description = str(obj.get("description", "")).strip()
        relations = ", ".join(
            relation
            for relation in _compact_relations(obj.get("relations", []))
            if _claim_relation(relation, used_relations)
        )
        relation_text = f", {relations}" if relations else ""
        if name and description:
            object_hints.append(
                f"{name} positioned {position}: {description}{relation_text}"
            )
    return _normalize_spaces(
        f"cinematic composition, {background}, " + ", ".join(object_hints)
    )


def _compact_relations(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_relations = [value]
    elif isinstance(value, Sequence):
        raw_relations = [str(item) for item in value]
    else:
        raw_relations = []
    result: list[str] = []
    for relation in raw_relations:
        cleaned = _normalize_spaces(relation)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(token in lowered for token in ("grip", "handle", "hold")):
            cleaned = "held by/holding target object"
        if cleaned.lower() not in {item.lower() for item in result}:
            result.append(cleaned)
    return result[:2]


def _claim_relation(relation: str, used_relations: set[str]) -> bool:
    key = relation.lower()
    if key in used_relations:
        return False
    used_relations.add(key)
    return True


def should_plan_layout(prompt: str) -> bool:
    """Heuristic gate for future orchestrator integration."""

    prompt = _clean_text(prompt, "prompt")
    lowered = prompt.lower()
    constraints = extract_constraints(prompt)
    relation_terms = (
        "next to",
        "behind",
        "in front of",
        "above",
        "under",
        "gripping",
        "holding",
        "left of",
        "right of",
    )
    if len(constraints.subjects) >= 2 and any(term in lowered for term in relation_terms):
        return True
    return lowered.count(",") >= 2 and len(constraints.subjects) >= 2


def build_mock_layout_response(
    prompt: str,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> str:
    """Return a deterministic mock layout for smoke tests and CLI demos."""

    prompt = _clean_text(prompt, "prompt")
    width, height = _validate_canvas_size(canvas_size)
    constraints = extract_constraints(prompt)
    object_names = _mock_layout_object_names(constraints)
    object_positions = _initial_mock_positions(object_names)
    _apply_relation_positions(object_positions, constraints.intent_spec.relations if constraints.intent_spec else [])
    objects = []
    color_items = list(constraints.colors.items())
    color_lookup = {name: color for name, color in color_items}
    if not object_names:
        object_names = ["main subject"]
    for index, name in enumerate(object_names):
        color = color_lookup.get(name) or color_lookup.get(_head_name(name), "")
        head = name.split()[-1]
        if "umbrella" in head:
            bbox = [int(width * 0.34), int(height * 0.08), int(width * 0.38), int(height * 0.28)]
            description = f"{color} {name} canopy and visible handle".strip()
            relations = ["above the robot", "handle aligned with robot hand"]
        elif "robot" in head:
            bbox = [int(width * 0.40), int(height * 0.44), int(width * 0.22), int(height * 0.38)]
            description = f"small {color} {name} with visible hand".strip()
            relations = ["below umbrella", "hand reaches umbrella handle"]
        else:
            bbox = _bbox_from_position(
                object_positions.get(name, _fallback_position(index)),
                width,
                height,
                subject=name,
            )
            description = f"{color} {name}".strip()
            relations = _relations_for_object(name, constraints.intent_spec.relations if constraints.intent_spec else [])
        objects.append(
            {
                "name": name,
                "description": description,
                "bbox": bbox,
                "order": index + 1,
                "relations": relations,
                "requires_reference": False,
            }
        )
    return json.dumps(
        {
            "canvas_size": [width, height],
            "background": {
                "description": "cinematic rainy street with wet pavement and distant lights",
                "viewpoint": "front view, medium shot, object space centered",
            },
            "objects": objects,
        },
        ensure_ascii=False,
    )


def _mock_layout_object_names(constraints: Any) -> list[str]:
    intent = getattr(constraints, "intent_spec", None)
    names: list[str] = []
    for name in getattr(constraints, "subjects", []) or []:
        _append_mock_object_name(names, str(name))
    for name in getattr(constraints, "colors", {}) or {}:
        _append_mock_object_name(names, str(name))
    if intent:
        for relation in intent.relations:
            _append_mock_object_name(names, relation.get("subject", ""))
            _append_mock_object_name(names, relation.get("object", ""))
        for relation in intent.interaction_relations:
            _append_mock_object_name(names, relation.get("subject", ""))
            _append_mock_object_name(names, relation.get("object", ""))
    return names[:8]


def _append_mock_object_name(names: list[str], value: str) -> None:
    cleaned = _normalize_spaces(str(value or "").strip().lower())
    if not cleaned:
        return
    for existing in names:
        if cleaned.startswith(f"{existing} "):
            return
    if cleaned in names:
        return
    for existing in names:
        if _same_layout_object(existing, cleaned):
            return
    names.append(cleaned)


def _same_layout_object(left: str, right: str) -> bool:
    return left == right or _head_name(left) == _head_name(right)


def _initial_mock_positions(names: Sequence[str]) -> dict[str, tuple[float, float]]:
    count = max(1, len(names))
    positions: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(names):
        x = 0.22 + (0.56 * index / max(1, count - 1)) if count > 1 else 0.5
        positions[name] = (min(0.78, max(0.18, x)), 0.56)
    return positions


def _apply_relation_positions(
    positions: dict[str, tuple[float, float]],
    relations: Sequence[Mapping[str, str]],
) -> None:
    for relation in relations:
        subject = _find_layout_name(positions, relation.get("subject", ""))
        obj = _find_layout_name(positions, relation.get("object", ""))
        if not subject or not obj:
            continue
        sx, sy = positions[subject]
        ox, oy = positions[obj]
        relation_name = _normalize_relation_name(
            relation.get("relation", "") or relation.get("phrase", "")
        )
        if relation_name in {"left_of", "left of"}:
            sx, sy = ox - 0.28, oy
        elif relation_name in {"right_of", "right of"}:
            sx, sy = ox + 0.28, oy
        elif relation_name in {"under", "below", "beneath"}:
            sx, sy = ox, oy + 0.28
        elif relation_name in {"above", "over"}:
            sx, sy = ox, oy - 0.28
        elif relation_name in {"in_front_of", "in front of"}:
            sx, sy = ox, oy + 0.18
        elif relation_name == "behind":
            sx, sy = ox, oy - 0.18
        elif relation_name == "inside":
            sx, sy = ox, oy
        else:
            continue
        positions[subject] = (_clamp_position(sx), _clamp_position(sy))
    _spread_overlapping_positions(positions)


def _find_layout_name(
    positions: Mapping[str, tuple[float, float]],
    value: str,
) -> str | None:
    cleaned = _normalize_spaces(str(value or "").strip().lower())
    if not cleaned:
        return None
    if cleaned in positions:
        return cleaned
    cleaned_head = _head_name(cleaned)
    for name in positions:
        if name == cleaned or _head_name(name) == cleaned_head:
            return name
    return None


def _normalize_relation_name(value: str) -> str:
    return _normalize_spaces(str(value or "").strip().lower()).replace(" ", "_")


def _clamp_position(value: float) -> float:
    return min(0.82, max(0.18, value))


def _spread_overlapping_positions(positions: dict[str, tuple[float, float]]) -> None:
    seen: dict[tuple[int, int], int] = {}
    for name, (x, y) in list(positions.items()):
        key = (round(x, 2), round(y, 2))
        count = seen.get(key, 0)
        if count:
            x = _clamp_position(x + 0.12 * count)
            y = _clamp_position(y + 0.08 * count)
            positions[name] = (x, y)
        seen[key] = count + 1


def _bbox_from_position(
    position: tuple[float, float],
    width: int,
    height: int,
    *,
    subject: str,
) -> list[int]:
    center_x, center_y = position
    box_width = int(width * (0.18 if _is_small_layout_subject(subject) else 0.22))
    box_height = int(height * (0.22 if _is_small_layout_subject(subject) else 0.30))
    x = int(round(center_x * width - box_width / 2))
    y = int(round(center_y * height - box_height / 2))
    x = max(0, min(width - box_width, x))
    y = max(0, min(height - box_height, y))
    return [x, y, box_width, box_height]


def _fallback_position(index: int) -> tuple[float, float]:
    return (min(0.82, 0.22 + 0.18 * index), 0.56)


def _is_small_layout_subject(subject: str) -> bool:
    return _head_name(subject) in {
        "bird",
        "birds",
        "key",
        "feather",
        "symbol",
        "seat",
        "spoon",
    }


def _relations_for_object(
    name: str,
    relations: Sequence[Mapping[str, str]],
) -> list[str]:
    result: list[str] = []
    for relation in relations:
        subject = str(relation.get("subject", ""))
        obj = str(relation.get("object", ""))
        relation_name = str(relation.get("relation", "") or relation.get("phrase", ""))
        phrase = str(relation.get("phrase", ""))
        statement = _relation_statement(subject, relation_name, obj, phrase)
        if _same_layout_object(name, subject):
            result.append(statement)
        elif _same_layout_object(name, obj):
            result.append(statement)
    return _dedupe(result)[:3]


def _head_name(value: str) -> str:
    parts = str(value or "").strip().lower().split()
    return parts[-1] if parts else ""


def _relation_statement(
    subject: str,
    relation_name: str,
    obj: str,
    phrase: str,
) -> str:
    subject = _normalize_spaces(subject).strip()
    relation_name = _normalize_spaces(relation_name).strip()
    obj = _normalize_spaces(obj).strip()
    phrase = _normalize_spaces(phrase).strip()
    if subject and relation_name and obj:
        return f"{subject} {relation_name} {obj}"
    return phrase or relation_name


def _normalize_background(raw_background: Any) -> dict[str, str]:
    if isinstance(raw_background, str):
        description = raw_background
        viewpoint = "unspecified camera view"
    elif isinstance(raw_background, Mapping):
        description = _first_text(
            raw_background.get("description"),
            raw_background.get("prompt"),
            raw_background.get("background_prompt"),
        )
        viewpoint = _first_text(raw_background.get("viewpoint"), raw_background.get("camera"))
    else:
        raise ValueError("Layout missing background")
    description = _clean_text(description, "background description")
    viewpoint = viewpoint or "unspecified camera view"
    return {
        "description": _normalize_spaces(description),
        "viewpoint": _normalize_spaces(viewpoint),
    }


def _bbox_text(bbox: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "unknown"
    return "-".join(str(int(value)) for value in bbox)


def _bbox_position(bbox: Any, canvas_size: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "center"
    if not isinstance(canvas_size, list) or len(canvas_size) != 2:
        canvas_size = [1024, 1024]
    width, height = max(1, int(canvas_size[0])), max(1, int(canvas_size[1]))
    x, y, box_width, box_height = [int(value) for value in bbox]
    cx = (x + box_width / 2) / width
    cy = (y + box_height / 2) / height
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    vertical = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "middle":
        return horizontal
    return f"{vertical}-{horizontal}"


def _normalize_objects(raw_objects: Any, canvas_size: tuple[int, int]) -> list[dict[str, Any]]:
    if raw_objects is None:
        raw_objects = []
    if not isinstance(raw_objects, list):
        raise ValueError("Layout objects must be a list")

    objects: list[dict[str, Any]] = []
    for fallback_order, raw_obj in enumerate(raw_objects, start=1):
        if not isinstance(raw_obj, Mapping):
            raise ValueError("Each layout object must be a mapping")
        name = _first_text(raw_obj.get("name"), raw_obj.get("label"), raw_obj.get("object"))
        name = _clean_text(name, "object name")
        description = _first_text(raw_obj.get("description"), raw_obj.get("prompt"), name)
        bbox = _normalize_bbox(raw_obj, canvas_size, name)
        order = _coerce_int(raw_obj.get("order", fallback_order), fallback_order)
        relations = _normalize_relations(raw_obj.get("relations", []))
        objects.append(
            {
                "name": name,
                "description": _normalize_spaces(description),
                "bbox": bbox,
                "order": order,
                "relations": relations,
                "requires_reference": bool(raw_obj.get("requires_reference", False)),
            }
        )
    objects.sort(key=lambda item: item["order"])
    return objects


def _normalize_bbox(
    raw_obj: Mapping[str, Any],
    canvas_size: tuple[int, int],
    name: str,
) -> list[int]:
    raw_bbox = (
        raw_obj.get("bbox")
        if raw_obj.get("bbox") is not None
        else raw_obj.get("box", raw_obj.get("bounding_box"))
    )
    if isinstance(raw_bbox, Mapping):
        raw_bbox = [
            raw_bbox.get("x", raw_bbox.get("left")),
            raw_bbox.get("y", raw_bbox.get("top")),
            raw_bbox.get("width", raw_bbox.get("w")),
            raw_bbox.get("height", raw_bbox.get("h")),
        ]
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError(f"Object '{name}' bbox must be [x, y, width, height]")

    values = [_coerce_float(value) for value in raw_bbox]
    if all(0.0 <= value <= 1.0 for value in values):
        values = [
            values[0] * canvas_size[0],
            values[1] * canvas_size[1],
            values[2] * canvas_size[0],
            values[3] * canvas_size[1],
        ]
    bbox = [int(round(value)) for value in values]
    _validate_bbox(bbox, canvas_size, name)
    return bbox


def _validate_bbox(bbox: list[int], canvas_size: tuple[int, int], name: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"Object '{name}' bbox must be [x, y, width, height]")
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"Object '{name}' bbox width and height must be positive")
    if x < 0 or y < 0 or x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise ValueError(f"Object '{name}' bbox is outside the canvas")


def _clean_background_against_objects(
    background: Mapping[str, str],
    objects: Sequence[Mapping[str, Any]],
    *,
    strict_background: bool,
) -> tuple[dict[str, str], list[str]]:
    description = str(background["description"])
    viewpoint = str(background["viewpoint"])
    terms = _foreground_terms(objects)
    found = [term for term in terms if _contains_word(description, term)]
    if found and strict_background:
        raise ValueError(
            "Background duplicates foreground object terms: " + ", ".join(found)
        )

    warnings: list[str] = []
    cleaned_description = description
    for term in found:
        cleaned_description = _remove_word(cleaned_description, term)
    cleaned_description = _normalize_spaces(cleaned_description).strip(" ,;")
    if found:
        warnings.append(
            "removed foreground terms from background: " + ", ".join(found)
        )
    if not cleaned_description:
        cleaned_description = "environment-only background"
        warnings.append("background became empty after foreground cleanup")
    return {
        "description": cleaned_description,
        "viewpoint": viewpoint,
    }, warnings


def _foreground_terms(objects: Sequence[Mapping[str, Any]]) -> list[str]:
    terms: list[str] = []
    for obj in objects:
        name = str(obj.get("name", "")).strip().lower()
        if name:
            terms.append(name)
            head = name.split()[-1]
            if len(head) > 2:
                terms.append(head)
    return _dedupe(terms)


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


def _canvas_from_layout(layout: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = layout.get("canvas_size", layout.get("canvas"))
    if isinstance(raw, Mapping):
        raw = [raw.get("width"), raw.get("height")]
    if isinstance(raw, list) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    return None


def _validate_canvas_size(canvas_size: tuple[int, int]) -> tuple[int, int]:
    if len(canvas_size) != 2:
        raise ValueError("canvas_size must be (width, height)")
    width, height = int(canvas_size[0]), int(canvas_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("canvas_size values must be positive")
    return width, height


def _normalize_relations(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _contains_word(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE))


def _remove_word(text: str, term: str) -> str:
    return re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _normalize_spaces(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,;.])", r"\1", value)
    value = re.sub(r"([,;])\s*", r"\1 ", value)
    return value.strip()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox values must be numeric: {value!r}") from exc
