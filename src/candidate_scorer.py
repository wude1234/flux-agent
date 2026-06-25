"""Mockable GenPilot-style candidate prompt scorer."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .clients import VLMClient


DEFAULT_WEIGHTS = {
    "alignment": 0.35,
    "attribute_binding": 0.20,
    "object_relationship": 0.20,
    "background_consistency": 0.15,
    "aesthetic": 0.10,
}


@dataclass(frozen=True)
class CandidateScore:
    """A scored candidate prompt."""

    prompt: str
    score: float
    subscores: dict[str, float]
    reason: str
    candidate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score": self.score,
            "subscores": dict(self.subscores),
            "reason": self.reason,
            "candidate": deepcopy(self.candidate),
        }


class CandidateScorer:
    """Rank prompt candidates using a VLM adapter or deterministic fallback."""

    def __init__(
        self,
        vlm: VLMClient | None = None,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.vlm = vlm
        self.weights = _normalize_weights(weights or DEFAULT_WEIGHTS)

    def score_candidates(
        self,
        user_prompt: str,
        candidates: Sequence[Mapping[str, Any] | str],
        *,
        image_paths: Sequence[str] | None = None,
        errors: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        user_prompt = _clean_text(user_prompt, "user_prompt")
        normalized = [_normalize_candidate(candidate) for candidate in candidates]
        if not normalized:
            return []
        image_paths = [str(path) for path in image_paths or []]
        errors = [deepcopy(dict(item)) for item in errors or []]

        parsed_scores: list[dict[str, Any]] = []
        if self.vlm is not None:
            request = _score_request(user_prompt, normalized, errors)
            raw_response = self.vlm.vision(request, list(image_paths))
            parsed_scores = _parse_score_response(raw_response, len(normalized))

        scored: list[CandidateScore] = []
        for index, candidate in enumerate(normalized):
            parsed = _find_score(parsed_scores, index)
            subscores = (
                _normalize_subscores(parsed.get("subscores", parsed))
                if parsed
                else _fallback_subscores(candidate, errors)
            )
            score = _weighted_sum(subscores, self.weights)
            reason = str(parsed.get("reason", "")) if parsed else _fallback_reason(candidate)
            scored.append(
                CandidateScore(
                    prompt=candidate["prompt"],
                    score=score,
                    subscores=subscores,
                    reason=reason,
                    candidate=candidate,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return [item.to_dict() for item in scored]


def _score_request(
    user_prompt: str,
    candidates: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> str:
    return "\n".join(
        [
            "You are a GenPilot-style MLLM scorer for candidate image prompts.",
            "Score each candidate from 0 to 1 on alignment, attribute_binding,",
            "object_relationship, background_consistency, and aesthetic.",
            "Return JSON: {\"scores\": [{\"index\": 0, \"subscores\": {...},",
            "\"reason\": \"...\"}]}",
            f"User idea: {user_prompt}",
            f"Known errors: {json.dumps(list(errors), ensure_ascii=False, sort_keys=True)}",
            f"Candidates: {json.dumps(list(candidates), ensure_ascii=False, sort_keys=True)}",
        ]
    )


def _parse_score_response(response: str, num_candidates: int) -> list[dict[str, Any]]:
    data = _extract_json_object(response)
    if not data:
        return []
    raw_scores = data.get("scores", [])
    if isinstance(raw_scores, Mapping):
        raw_scores = [
            {"index": key, "subscores": value} for key, value in raw_scores.items()
        ]
    if not isinstance(raw_scores, list):
        return []
    scores: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(raw_scores):
        if not isinstance(item, Mapping):
            continue
        index = _coerce_int(item.get("index", fallback_index), fallback_index)
        if 0 <= index < num_candidates:
            record = deepcopy(dict(item))
            record["index"] = index
            scores.append(record)
    return scores


def _find_score(scores: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    for score in scores:
        if int(score.get("index", -1)) == index:
            return deepcopy(dict(score))
    return {}


def _fallback_subscores(
    candidate: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    prompt = candidate["prompt"].lower()
    score = 0.55
    if any(word in prompt for word in ("exact", "clearly", "explicit", "visible")):
        score += 0.15
    if candidate.get("fixes"):
        score += 0.1
    if any(str(error.get("error_type", "")) in candidate.get("fixes", []) for error in errors):
        score += 0.1
    score = min(score, 0.9)
    return {
        "alignment": score,
        "attribute_binding": score,
        "object_relationship": max(0.0, score - 0.05),
        "background_consistency": max(0.0, score - 0.1),
        "aesthetic": 0.6,
    }


def _fallback_reason(candidate: Mapping[str, Any]) -> str:
    if candidate.get("expected_improvement"):
        return str(candidate["expected_improvement"])
    return "Deterministic fallback score from candidate metadata."


def _normalize_candidate(candidate: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(candidate, str):
        return {"prompt": _clean_text(candidate, "candidate prompt")}
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a mapping or string")
    record = deepcopy(dict(candidate))
    record["prompt"] = _clean_text(str(record.get("prompt", "")), "candidate prompt")
    return record


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    normalized = {key: float(weights.get(key, 0.0)) for key in DEFAULT_WEIGHTS}
    total = sum(normalized.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in normalized.items()}


def _normalize_subscores(raw: Mapping[str, Any]) -> dict[str, float]:
    aliases = {
        "Attribute-Binding": "attribute_binding",
        "Object-Relationship": "object_relationship",
        "Background-Consistency": "background_consistency",
        "attribute-binding": "attribute_binding",
        "object-relationship": "object_relationship",
        "background-consistency": "background_consistency",
    }
    result: dict[str, float] = {}
    for key in DEFAULT_WEIGHTS:
        raw_value = raw.get(key)
        if raw_value is None:
            for alias, canonical in aliases.items():
                if canonical == key and alias in raw:
                    raw_value = raw[alias]
                    break
        result[key] = _normalize_score(raw_value if raw_value is not None else 0.0)
    return result


def _weighted_sum(subscores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return round(sum(subscores.get(key, 0.0) * weights[key] for key in weights), 6)


def _normalize_score(value: Any) -> float:
    try:
        if isinstance(value, list):
            value = value[0] if value else 0
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 5.0 if score <= 5.0 else score / 10.0
    return max(0.0, min(1.0, score))


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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value

