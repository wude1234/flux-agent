"""GenPilot-style candidate prompt generation with memory-aware filtering."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .clients import LLMClient
from .error_analyzer import PromptError
from .prompt_constraints import extract_constraints, make_constraints_context


@dataclass(frozen=True)
class PromptCandidate:
    """A revised full prompt plus metadata about the fix it attempts."""

    prompt: str
    modified_sentence: str
    fixes: list[str]
    expected_improvement: str
    risk: str = ""
    source: str = "genpilot"

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _clean_text(self.prompt, "prompt"))
        object.__setattr__(
            self,
            "modified_sentence",
            _clean_text(self.modified_sentence, "modified_sentence"),
        )
        object.__setattr__(self, "fixes", [str(item) for item in self.fixes])
        object.__setattr__(
            self,
            "expected_improvement",
            str(self.expected_improvement or "Improve prompt-image alignment."),
        )
        object.__setattr__(self, "risk", str(self.risk or ""))
        object.__setattr__(self, "source", str(self.source or "genpilot"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "modified_sentence": self.modified_sentence,
            "fixes": list(self.fixes),
            "expected_improvement": self.expected_improvement,
            "risk": self.risk,
            "source": self.source,
        }


class PromptOptimizer:
    """Generate unique candidate prompts for one or more localized errors."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate_candidate_prompts(
        self,
        prompt_error: PromptError | Mapping[str, Any],
        *,
        num_candidates: int = 3,
        memory: Sequence[Mapping[str, Any]] | None = None,
        existing_candidates: Sequence[Mapping[str, Any] | str] | None = None,
    ) -> list[dict[str, Any]]:
        """Port GenPilot's modify-and-merge loop behind ``LLMClient``."""

        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        error = (
            prompt_error
            if isinstance(prompt_error, PromptError)
            else PromptError.from_dict(prompt_error)
        )
        memory_records = [deepcopy(dict(item)) for item in memory or []]
        blocked = _prompt_fingerprint_set(memory_records, existing_candidates or [])

        request = _candidate_request(error, num_candidates, memory_records)
        response = self.llm.text(request)
        parsed = _parse_candidate_response(response, error)
        candidates = _filter_unique(parsed, blocked, num_candidates)

        variant_index = 1
        while len(candidates) < num_candidates:
            fallback = _fallback_candidate(error, variant_index)
            variant_index += 1
            if _fingerprint(fallback.prompt) in blocked:
                continue
            candidates.append(fallback)
            blocked.add(_fingerprint(fallback.prompt))

        return [candidate.to_dict() for candidate in candidates]

    def optimize(
        self,
        prompt_errors: Sequence[PromptError | Mapping[str, Any]],
        *,
        num_candidates: int = 3,
        memory: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate up to ``num_candidates`` unique candidates across errors."""

        candidates: list[dict[str, Any]] = []
        for prompt_error in prompt_errors:
            needed = num_candidates - len(candidates)
            if needed <= 0:
                break
            candidates.extend(
                self.generate_candidate_prompts(
                    prompt_error,
                    num_candidates=needed,
                    memory=memory,
                    existing_candidates=candidates,
                )
            )
        return candidates[:num_candidates]


def _candidate_request(
    error: PromptError,
    num_candidates: int,
    memory: Sequence[Mapping[str, Any]],
) -> str:
    memory_blob = json.dumps(list(memory), ensure_ascii=False, sort_keys=True)
    constraints = extract_constraints(error.original_prompt)
    return "\n".join(
        [
            "You are the PromptOptimizerAgent using a GenPilot-style refinement loop.",
            "Given an image-generation prompt error, first minimally modify the",
            "failed sentence, then merge it back into the full original prompt.",
            make_constraints_context(constraints),
            "Optimize user-grounded failures first. If a critique complains about",
            "extra details that were introduced by a previous expansion rather than",
            "requested by the user, do not over-optimize those extras.",
            "Do not change user-specified colors, subjects, actions, or relations.",
            "Keep unrelated prompt details unchanged. Avoid previous failed attempts.",
            "Return JSON: {\"candidates\": [{\"modified_sentence\": ...,",
            "\"prompt\": ..., \"fixes\": [...], \"expected_improvement\": ...,",
            "\"risk\": ...}]}",
            f"Original prompt: {error.original_prompt}",
            f"Failed sentence: {error.failed_sentence}",
            f"Error type: {error.error_type}",
            f"Error: {error.error}",
            f"Revision history / memory: {memory_blob}",
            f"Generate exactly {num_candidates} diverse candidate prompts.",
        ]
    )


def _parse_candidate_response(
    response: str,
    error: PromptError,
) -> list[PromptCandidate]:
    data = _extract_json_object(response)
    raw_candidates: list[Any] = []
    if data:
        if isinstance(data.get("candidates"), list):
            raw_candidates = data["candidates"]
        elif data.get("prompt") or data.get("modified_sentence"):
            raw_candidates = [data]

    candidates: list[PromptCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, Mapping):
            continue
        prompt = str(item.get("prompt") or "").strip()
        modified_sentence = str(item.get("modified_sentence") or "").strip()
        if not modified_sentence:
            modified_sentence = _modified_sentence_from_prompt(prompt, error)
        if not prompt and modified_sentence:
            prompt = merge_modified(error.original_prompt, error.failed_sentence, modified_sentence)
        if not prompt:
            continue
        fixes = item.get("fixes") if isinstance(item.get("fixes"), list) else [error.error_type]
        candidates.append(
            PromptCandidate(
                prompt=prompt,
                modified_sentence=modified_sentence or error.failed_sentence,
                fixes=[str(fix) for fix in fixes],
                expected_improvement=str(
                    item.get("expected_improvement")
                    or f"Address {error.error_type}: {error.error}"
                ),
                risk=str(item.get("risk") or ""),
            )
        )

    if candidates:
        return candidates

    tagged = re.findall(r"<START>(.*?)</?END>", response, flags=re.I | re.S)
    for item in tagged:
        modified_sentence = _clean_prompt(item)
        prompt = merge_modified(error.original_prompt, error.failed_sentence, modified_sentence)
        candidates.append(
            PromptCandidate(
                prompt=prompt,
                modified_sentence=modified_sentence,
                fixes=[error.error_type],
                expected_improvement=f"Address {error.error_type}: {error.error}",
            )
        )
    return candidates


def merge_modified(
    original_prompt: str,
    failed_sentence: str,
    modified_sentence: str,
) -> str:
    """Merge a modified sentence back into a full prompt with minimal change."""

    original_prompt = _clean_text(original_prompt, "original_prompt")
    failed_sentence = _clean_text(failed_sentence, "failed_sentence")
    modified_sentence = _clean_text(modified_sentence, "modified_sentence")
    if failed_sentence in original_prompt:
        return original_prompt.replace(failed_sentence, modified_sentence, 1)
    return f"{original_prompt} {modified_sentence}"


def _fallback_candidate(error: PromptError, variant_index: int) -> PromptCandidate:
    modifier = _modifier_for_error(error, variant_index)
    modified_sentence = _append_modifier(error.failed_sentence, modifier)
    return PromptCandidate(
        prompt=merge_modified(error.original_prompt, error.failed_sentence, modified_sentence),
        modified_sentence=modified_sentence,
        fixes=[error.error_type],
        expected_improvement=f"Make the prompt explicitly address: {error.error}",
        risk="Rule-based fallback may over-emphasize one detail.",
    )


def _modifier_for_error(error: PromptError, variant_index: int) -> str:
    hint = _repair_focus_hint(error)
    strategy = {
        1: "with direct visual wording",
        2: "with composition guidance",
        3: "with redundant emphasis",
        4: "with a concrete checklist phrase",
    }.get(variant_index, f"with alternative phrasing {variant_index}")
    if error.error_type == "wrong_count":
        return f"{strategy}, the exact requested count clearly visible: {hint}"
    if error.error_type == "missing_object":
        return f"{strategy}, make the required object clearly visible: {hint}"
    if error.error_type == "wrong_relation":
        return f"{strategy}, state the required relation plainly: {hint}"
    if error.error_type == "style_mismatch":
        return f"{strategy}, background and style constrained to match: {hint}"
    if error.error_type == "artifact":
        return f"{strategy}, clean, coherent, artifact-free"
    if variant_index == 1:
        return f"{strategy}, make the required attribute explicit: {hint}"
    return f"{strategy}, clearly emphasize the required detail: {hint}"


def _repair_focus_hint(error: PromptError) -> str:
    span = _sanitize_focus_text(error.failed_sentence)
    if span:
        return span
    return _sanitize_focus_text(error.error) or "the original user-specified detail"


def _sanitize_focus_text(value: str) -> str:
    text = str(value or "").strip().rstrip(".")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(?:not|no)\s+(?:present|visible|satisfied|correct)\b.*", "", text, flags=re.I)
    text = re.sub(r"\b(?:violating|violate|missing|failed|failure|incorrect|wrong)\b.*", "", text, flags=re.I)
    text = " ".join(text.split(" ,.;:-"))
    text = " ".join(text.split())
    return text.strip(" ,.;:-")


def _append_modifier(sentence: str, modifier: str) -> str:
    sentence = sentence.strip()
    suffix = "." if sentence.endswith(".") else ""
    base = sentence[:-1] if suffix else sentence
    return f"{base}, {modifier}{suffix}"


def _modified_sentence_from_prompt(prompt: str, error: PromptError) -> str:
    if error.failed_sentence in prompt:
        return error.failed_sentence
    return prompt or error.failed_sentence


def _filter_unique(
    candidates: Sequence[PromptCandidate],
    blocked: set[str],
    limit: int,
) -> list[PromptCandidate]:
    result: list[PromptCandidate] = []
    for candidate in candidates:
        key = _fingerprint(candidate.prompt)
        if key in blocked:
            continue
        result.append(candidate)
        blocked.add(key)
        if len(result) == limit:
            break
    return result


def _prompt_fingerprint_set(
    memory: Sequence[Mapping[str, Any]],
    existing_candidates: Sequence[Mapping[str, Any] | str],
) -> set[str]:
    result: set[str] = set()
    for record in memory:
        for key in ("prompt", "selected_prompt", "modified_prompt", "candidate_prompt"):
            if record.get(key):
                result.add(_fingerprint(str(record[key])))
        if isinstance(record.get("candidate"), Mapping):
            candidate = record["candidate"]
            if candidate.get("prompt"):
                result.add(_fingerprint(str(candidate["prompt"])))
    for item in existing_candidates:
        if isinstance(item, Mapping):
            prompt = item.get("prompt")
        else:
            prompt = item
        if prompt:
            result.add(_fingerprint(str(prompt)))
    return result


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


def _clean_prompt(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^\s*(?:\d+[\).\s-]+|[-*]\s+)", "", text)
    return " ".join(text.split())


def _fingerprint(prompt: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", prompt.lower()))


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value
