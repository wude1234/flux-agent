"""M6 reward-style image reranking adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from .clients import VLMClient


REWARD_ASPECTS = ("alignment", "fidelity", "safety", "overall")


class RewardBackend(Protocol):
    """Scalar reward backend interface."""

    def score(self, prompt: str, image_path: str, *, aspect: str = "overall") -> float:
        ...


@dataclass
class MockRewardBackend:
    """Deterministic reward backend for tests and smoke runs."""

    scores: Mapping[str, float] | None = None
    default_score: float = 0.5
    calls: list[dict[str, Any]] = field(default_factory=list)

    def score(self, prompt: str, image_path: str, *, aspect: str = "overall") -> float:
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        aspect = _normalize_aspect(aspect)
        self.calls.append({"prompt": prompt, "image_path": image_path, "aspect": aspect})
        if self.scores and image_path in self.scores:
            return _normalize_score(self.scores[image_path])
        if self.scores:
            stem = Path(image_path).name
            if stem in self.scores:
                return _normalize_score(self.scores[stem])
        return _normalize_score(self.default_score)


class UnavailableRewardBackend:
    """Placeholder for real LLaVA-Reward checkpoints."""

    def __init__(self, *, reason: str = "LLaVA-Reward checkpoint is not configured") -> None:
        self.reason = reason

    def score(self, prompt: str, image_path: str, *, aspect: str = "overall") -> float:
        del prompt, image_path, aspect
        raise RuntimeError(self.reason)


class VLMRewardBackend:
    """API/VLM-backed reward proxy inspired by LLaVA-Reward aspects."""

    def __init__(self, vlm: VLMClient) -> None:
        self.vlm = vlm
        self.calls: list[dict[str, Any]] = []

    def score(self, prompt: str, image_path: str, *, aspect: str = "overall") -> float:
        prompt = _clean_text(prompt, "prompt")
        image_path = _clean_text(image_path, "image_path")
        aspect = _normalize_aspect(aspect)
        request = build_vlm_reward_request(prompt, image_path, aspect=aspect)
        raw_response = self.vlm.vision(request, [image_path])
        score = parse_vlm_reward_response(raw_response)
        self.calls.append(
            {
                "prompt": prompt,
                "image_path": image_path,
                "aspect": aspect,
                "request": request,
                "raw_response": raw_response,
                "score": score,
            }
        )
        return score


class RewardReranker:
    """Rank images using a reward backend with LLaVA-Reward-style aspects."""

    def __init__(
        self,
        backend: RewardBackend,
        *,
        aspects: Sequence[str] = ("overall",),
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.backend = backend
        self.aspects = tuple(_normalize_aspect(aspect) for aspect in aspects)
        if not self.aspects:
            raise ValueError("aspects must not be empty")
        self.weights = _normalize_weights(weights, self.aspects)

    def rank(
        self,
        prompt: str,
        image_paths: Sequence[str],
    ) -> dict[str, Any]:
        prompt = _clean_text(prompt, "prompt")
        image_paths = [_clean_text(path, "image_path") for path in image_paths]
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        ranked: list[dict[str, Any]] = []
        for index, image_path in enumerate(image_paths):
            aspect_scores = {
                aspect: _normalize_score(
                    self.backend.score(prompt, image_path, aspect=aspect)
                )
                for aspect in self.aspects
            }
            score = round(
                sum(aspect_scores[aspect] * self.weights[aspect] for aspect in self.aspects),
                6,
            )
            ranked.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "score": score,
                    "aspect_scores": aspect_scores,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return {
            "reranker": "reward_reranker",
            "prompt": prompt,
            "selected_index": ranked[0]["index"],
            "selected_image": ranked[0]["image_path"],
            "scores": ranked,
            "aspects": list(self.aspects),
            "weights": dict(self.weights),
        }


def pairwise_preference_probability(chosen_reward: float, rejected_reward: float) -> float:
    """Bradley-Terry-style preference probability used by reward rerankers."""

    import math

    delta = float(chosen_reward) - float(rejected_reward)
    return round(1.0 / (1.0 + math.exp(-delta)), 6)


def build_vlm_reward_request(prompt: str, image_path: str, *, aspect: str = "overall") -> str:
    """Build a compact VLM scoring prompt for reward proxy mode."""

    aspect = _normalize_aspect(aspect)
    aspect_focus = {
        "alignment": "text-image alignment, object counts, attributes, and relations",
        "fidelity": "visual fidelity, artifacts, deformations, and image quality",
        "safety": "unsafe, harmful, or policy-sensitive visual content",
        "overall": "overall preference quality across alignment, fidelity, safety, and aesthetics",
    }[aspect]
    return "\n".join(
        [
            "You are an API reward model proxy for a text-to-image agent.",
            "Score the image for one LLaVA-Reward-style aspect.",
            "Return exactly one JSON object: {\"score\": 0.0, \"reason\": \"short reason\"}.",
            "The score must be in [0, 1], where 1 is best.",
            f"Aspect: {aspect}",
            f"Aspect focus: {aspect_focus}",
            f"Prompt: {prompt}",
            f"Image path: {image_path}",
        ]
    )


def parse_vlm_reward_response(response: str) -> float:
    """Parse API reward proxy output with JSON and text fallbacks."""

    response = _clean_text(response, "response")
    data = _extract_json_object(response)
    if isinstance(data, Mapping):
        score = data.get("score", data.get("reward", data.get("rating")))
        return _normalize_score(score)
    match = re.search(r"(?:score|reward|rating)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(10|5|1)", response, re.I)
    if match:
        return _normalize_score(float(match.group(1)) / float(match.group(2)))
    match = re.search(r"(?:score|reward|rating)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", response, re.I)
    if match:
        return _normalize_score(match.group(1))
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*(10|5|1)\b", response)
    if match:
        return _normalize_score(float(match.group(1)) / float(match.group(2)))
    return 0.0


def ranking_to_evaluation(ranking: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a reward ranking result into the M6 evaluator schema."""

    scores = ranking.get("scores", [])
    if not isinstance(scores, list) or not scores:
        return {
            "evaluator": "reward_reranker",
            "score": 0.0,
            "passed": False,
            "criteria_scores": {},
            "errors": [],
            "strengths": [],
            "revision_hint": "",
        }
    top = deepcopy(dict(scores[0]))
    score = float(top.get("score", 0.0))
    aspect_scores = top.get("aspect_scores", {})
    if not isinstance(aspect_scores, Mapping):
        aspect_scores = {}
    return {
        "evaluator": "reward_reranker",
        "score": score,
        "passed": score >= 0.75,
        "criteria_scores": dict(aspect_scores),
        "errors": [] if score >= 0.75 else [{"type": "artifact", "evidence": "Reward score below pass threshold.", "prompt_span": "", "severity": "major"}],
        "strengths": ["Selected by reward reranker."],
        "revision_hint": "" if score >= 0.75 else "Generate or edit a higher-reward candidate image.",
    }


def _normalize_aspect(value: str) -> str:
    aspect = _clean_text(value, "aspect").lower().replace("-", "_")
    aliases = {
        "artifact": "fidelity",
        "quality": "fidelity",
        "preference": "overall",
    }
    aspect = aliases.get(aspect, aspect)
    if aspect not in REWARD_ASPECTS:
        raise ValueError(f"unsupported reward aspect: {value}")
    return aspect


def _normalize_weights(
    weights: Mapping[str, float] | None,
    aspects: Sequence[str],
) -> dict[str, float]:
    if not weights:
        return {aspect: 1.0 / len(aspects) for aspect in aspects}
    normalized = {aspect: float(weights.get(aspect, 0.0)) for aspect in aspects}
    total = sum(normalized.values())
    if total <= 0:
        return {aspect: 1.0 / len(aspects) for aspect in aspects}
    return {aspect: value / total for aspect, value in normalized.items()}


def _normalize_score(value: Any) -> float:
    try:
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


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned
