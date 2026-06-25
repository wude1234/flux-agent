"""M6 evaluator adapters and schemas."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Protocol, Sequence

from .clients import VLMClient


DEFAULT_EVALUATION_CRITERIA = (
    "alignment",
    "attribute_binding",
    "object_relationship",
    "background_consistency",
    "factuality",
    "artifact_quality",
    "aesthetic",
    "safety",
)

ERROR_TYPE_ALIASES = {
    "attribute": "wrong_attribute",
    "color": "wrong_attribute",
    "count": "wrong_count",
    "counting": "wrong_count",
    "relation": "wrong_relation",
    "spatial": "wrong_relation",
    "fact": "factuality",
    "factual": "factuality",
    "hallucination": "factuality",
    "artifact": "artifact",
    "quality": "artifact",
    "safety": "safety",
}


class Evaluator(Protocol):
    """Image evaluator interface used by later agent stages."""

    def evaluate(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class VLMJudgeEvaluator:
    """Lightweight VLM evaluator with a reward-model-compatible schema."""

    def __init__(
        self,
        vlm: VLMClient,
        *,
        criteria: Sequence[str] = DEFAULT_EVALUATION_CRITERIA,
        pass_threshold: float = 0.75,
    ) -> None:
        self.vlm = vlm
        self.criteria = tuple(_clean_text(item, "criterion") for item in criteria)
        if not self.criteria:
            raise ValueError("criteria must not be empty")
        self.pass_threshold = float(pass_threshold)

    def evaluate(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_prompt = _clean_text(user_prompt, "user_prompt")
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        context = deepcopy(dict(context or {}))
        request = build_vlm_judge_request(
            user_prompt=user_prompt,
            prompt=prompt,
            image_path=image_path,
            criteria=self.criteria,
            context=context,
        )
        raw_response = self.vlm.vision(request, [image_path])
        evaluation = parse_vlm_judge_response(
            raw_response,
            criteria=self.criteria,
            pass_threshold=self.pass_threshold,
        )
        evaluation.update(
            {
                "evaluator": "vlm_judge",
                "image_path": image_path,
                "prompt": prompt,
                "user_prompt": user_prompt,
                "context": context,
                "request": request,
                "raw_response": raw_response,
            }
        )
        return evaluation


def build_vlm_judge_request(
    *,
    user_prompt: str,
    prompt: str,
    image_path: str,
    criteria: Sequence[str] = DEFAULT_EVALUATION_CRITERIA,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Build the M6 VLM judge prompt."""

    context_blob = json.dumps(dict(context or {}), ensure_ascii=False, sort_keys=True)
    schema = {
        "score": 0.0,
        "passed": False,
        "criteria_scores": {criterion: 0.0 for criterion in criteria},
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": "visible mismatch",
                "prompt_span": "phrase to revise",
                "severity": "major",
            }
        ],
        "strengths": ["visible thing done well"],
        "revision_hint": "one concise repair instruction",
    }
    return "\n".join(
        [
            "You are a multimodal evaluator for a text-to-image agent.",
            "Judge the image against the original user prompt first, then the expanded prompt.",
            "Return exactly one JSON object. Scores are floats in [0, 1].",
            "Use criteria_scores for the listed criteria. Keep errors actionable for prompt revision.",
            f"Criteria: {', '.join(criteria)}",
            f"Schema: {json.dumps(schema, ensure_ascii=False)}",
            f"Original user prompt: {user_prompt}",
            f"Expanded prompt: {prompt}",
            f"Image path: {image_path}",
            f"Extra context: {context_blob}",
        ]
    )


def parse_vlm_judge_response(
    response: str,
    *,
    criteria: Sequence[str] = DEFAULT_EVALUATION_CRITERIA,
    pass_threshold: float = 0.75,
) -> dict[str, Any]:
    """Parse a VLM evaluator response into the unified schema."""

    response = _clean_text(response, "response")
    data = _extract_json_object(response) or {}
    criteria_scores = _normalize_criteria_scores(data, criteria)
    score = _normalize_score(data.get("score", data.get("overall_score")))
    if score is None:
        score = _score_from_text(response)
    if score is None:
        score = _mean(criteria_scores.values()) if criteria_scores else 0.0
    passed = bool(data.get("passed", score >= pass_threshold))
    errors = _normalize_errors(data.get("errors", []))
    if not errors and "errors" not in data:
        errors = _errors_from_text(response)
    strengths = _normalize_string_list(data.get("strengths", []))
    revision_hint = str(data.get("revision_hint") or data.get("suggestion") or "").strip()
    if not revision_hint and errors:
        revision_hint = str(errors[0].get("evidence", "Revise the prompt to fix evaluator errors."))
    return {
        "score": round(float(score), 6),
        "passed": passed,
        "criteria_scores": criteria_scores,
        "errors": errors,
        "strengths": strengths,
        "revision_hint": revision_hint,
    }


def evaluation_to_optimizer_errors(
    evaluation: Mapping[str, Any],
    *,
    original_prompt: str,
) -> list[dict[str, Any]]:
    """Convert M6 evaluator output into M2 ``ErrorAnalyzer``-friendly records."""

    original_prompt = _clean_text(original_prompt, "original_prompt")
    errors = evaluation.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    if not isinstance(errors, list):
        return []
    converted: list[dict[str, Any]] = []
    for item in errors:
        if not isinstance(item, Mapping):
            evidence = str(item).strip()
            error_type = "wrong_attribute"
            prompt_span = ""
        else:
            evidence = str(item.get("evidence") or item.get("message") or "").strip()
            error_type = normalize_error_type(item.get("type"))
            prompt_span = str(item.get("prompt_span") or "").strip()
        if not evidence:
            continue
        converted.append(
            {
                "original_prompt": original_prompt,
                "failed_sentence": prompt_span or original_prompt,
                "error": evidence,
                "error_type": error_type,
                "prompt_span": prompt_span,
                "source": str(evaluation.get("evaluator", "evaluator")),
            }
        )
    return converted


def normalize_error_type(value: Any) -> str:
    """Normalize evaluator-specific error labels to project labels."""

    lowered = str(value or "").strip().lower().replace("-", "_")
    if lowered in {
        "missing_object",
        "wrong_attribute",
        "wrong_material",
        "wrong_object_type",
        "wrong_count",
        "wrong_relation",
        "wrong_spatial_relation",
        "wrong_symbol_text",
        "style_mismatch",
        "artifact",
        "factuality",
        "safety",
    }:
        return lowered
    for needle, mapped in ERROR_TYPE_ALIASES.items():
        if needle in lowered:
            return mapped
    return "wrong_attribute"


def _normalize_criteria_scores(
    data: Mapping[str, Any],
    criteria: Sequence[str],
) -> dict[str, float]:
    raw = data.get("criteria_scores") or data.get("subscores") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    aliases = {
        "fidelity": "artifact_quality",
        "artifact": "artifact_quality",
        "overall": "alignment",
        "attribute-binding": "attribute_binding",
        "object-relationship": "object_relationship",
        "background-consistency": "background_consistency",
    }
    result: dict[str, float] = {}
    for criterion in criteria:
        raw_value = raw.get(criterion, data.get(criterion))
        if raw_value is None:
            for alias, canonical in aliases.items():
                if canonical == criterion:
                    raw_value = raw.get(alias, data.get(alias))
                    if raw_value is not None:
                        break
        result[criterion] = _normalize_score(raw_value) or 0.0
    return result


def _normalize_errors(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    errors: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            evidence = str(
                item.get("evidence") or item.get("message") or item.get("description") or ""
            ).strip()
            if not evidence:
                continue
            errors.append(
                {
                    "type": normalize_error_type(item.get("type")),
                    "evidence": evidence,
                    "prompt_span": str(item.get("prompt_span") or "").strip(),
                    "severity": str(item.get("severity") or "major").strip(),
                }
            )
        else:
            evidence = str(item).strip()
            if evidence:
                errors.append(
                    {
                        "type": normalize_error_type(evidence),
                        "evidence": evidence,
                        "prompt_span": "",
                        "severity": "major",
                    }
                )
    return errors


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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


def _normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 5.0 if score <= 5.0 else score / 10.0
    return max(0.0, min(1.0, score))


def _score_from_text(text: str) -> float | None:
    lowered = text.lower()
    patterns = (
        r"(?:score|overall|rating)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(10|5|1)",
        r"(?:score|overall|rating)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*(10|5|1)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        numerator = float(match.group(1))
        denominator = float(match.group(2)) if len(match.groups()) > 1 and match.group(2) else None
        if denominator:
            return max(0.0, min(1.0, numerator / denominator))
        return _normalize_score(numerator)
    return None


def _errors_from_text(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    markers = ("error", "issue", "problem", "mismatch", "wrong", "missing", "fail")
    lines = [
        line.strip(" -•\t")
        for line in text.splitlines()
        if line.strip(" -•\t")
    ]
    errors: list[dict[str, Any]] = []
    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in markers):
            errors.append(
                {
                    "type": normalize_error_type(line),
                    "evidence": line,
                    "prompt_span": "",
                    "severity": "major",
                }
            )
    if not errors and any(marker in lowered for marker in markers):
        errors.append(
            {
                "type": normalize_error_type(text),
                "evidence": text.strip(),
                "prompt_span": "",
                "severity": "major",
            }
        )
    return errors


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned
