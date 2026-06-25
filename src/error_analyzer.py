"""Lightweight GenPilot-style error normalization for prompt optimization."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ERROR_TYPE_TO_ASPECT = {
    "missing_object": "alignment",
    "wrong_attribute": "attribute_binding",
    "wrong_material": "attribute_binding",
    "wrong_object_type": "alignment",
    "wrong_count": "attribute_binding",
    "wrong_relation": "object_relationship",
    "wrong_spatial_relation": "object_relationship",
    "style_mismatch": "background_consistency",
    "artifact": "aesthetic",
}


@dataclass(frozen=True)
class PromptError:
    """A GenPilot-style error mapped to one editable prompt fragment."""

    original_prompt: str
    failed_sentence: str
    error: str
    error_type: str = "wrong_attribute"
    source: str = "visual_reflector"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "original_prompt", _clean_text(self.original_prompt, "original_prompt")
        )
        object.__setattr__(
            self, "failed_sentence", _clean_text(self.failed_sentence, "failed_sentence")
        )
        object.__setattr__(self, "error", _clean_text(self.error, "error"))
        object.__setattr__(self, "error_type", _normalize_error_type(self.error_type))
        object.__setattr__(self, "source", str(self.source or "unknown"))

    @property
    def aspect(self) -> str:
        return ERROR_TYPE_TO_ASPECT.get(self.error_type, "alignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "failed_sentence": self.failed_sentence,
            "error": self.error,
            "error_type": self.error_type,
            "aspect": self.aspect,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptError":
        return cls(
            original_prompt=str(data["original_prompt"]),
            failed_sentence=str(data["failed_sentence"]),
            error=str(data["error"]),
            error_type=str(data.get("error_type", "wrong_attribute")),
            source=str(data.get("source", "visual_reflector")),
        )


class ErrorAnalyzer:
    """Map M1 critiques or raw error text into editable ``PromptError`` records."""

    def analyze(
        self,
        original_prompt: str,
        critique: Mapping[str, Any] | str,
    ) -> list[PromptError]:
        original_prompt = _clean_text(original_prompt, "original_prompt")
        if isinstance(critique, str):
            return [
                PromptError(
                    original_prompt=original_prompt,
                    failed_sentence=_guess_failed_sentence(original_prompt, critique),
                    error=critique,
                    error_type=_classify_error(critique),
                    source="text",
                )
            ]
        if not isinstance(critique, Mapping):
            raise TypeError("critique must be a mapping or string")

        critique_record = deepcopy(dict(critique))
        raw_errors = critique_record.get("errors") or []
        if not raw_errors and critique_record.get("revision_hint"):
            raw_errors = [
                {
                    "type": _classify_error(str(critique_record["revision_hint"])),
                    "evidence": critique_record["revision_hint"],
                    "prompt_span": "",
                }
            ]
        if isinstance(raw_errors, Mapping):
            raw_errors = [raw_errors]

        prompt_errors: list[PromptError] = []
        for item in raw_errors:
            if isinstance(item, Mapping):
                evidence = _first_non_empty(
                    item.get("evidence"),
                    item.get("message"),
                    critique_record.get("revision_hint"),
                )
                prompt_span = _first_non_empty(item.get("prompt_span"), "")
                error_type = _normalize_error_type(
                    item.get("type") or _classify_error(evidence)
                )
            else:
                evidence = str(item).strip()
                prompt_span = ""
                error_type = _classify_error(evidence)
            if not evidence:
                continue
            failed_sentence = prompt_span or _guess_failed_sentence(
                original_prompt, evidence
            )
            prompt_errors.append(
                PromptError(
                    original_prompt=original_prompt,
                    failed_sentence=failed_sentence,
                    error=evidence,
                    error_type=error_type,
                    source=str(critique_record.get("source", "visual_reflector")),
                )
            )

        return prompt_errors


def split_prompt_sentences(prompt: str) -> list[str]:
    prompt = _clean_text(prompt, "prompt")
    parts = [
        part.strip(" ,;\n\t")
        for part in re.split(r"(?<=[.!?])\s+|;\s+|\n+", prompt)
        if part.strip(" ,;\n\t")
    ]
    return parts or [prompt]


def _guess_failed_sentence(prompt: str, error: str) -> str:
    sentences = split_prompt_sentences(prompt)
    error_terms = _terms(error)
    best_sentence = sentences[0]
    best_score = -1
    for sentence in sentences:
        sentence_terms = set(_terms(sentence))
        overlap = sum(1 for term in error_terms if term in sentence_terms)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    return best_sentence


def _classify_error(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("missing", "absent", "not present")):
        return "missing_object"
    if any(word in lowered for word in ("count", "number", "exactly", "too many")):
        return "wrong_count"
    if any(word in lowered for word in ("relation", "spatial", "left", "right", "pose")):
        return "wrong_relation"
    if any(word in lowered for word in ("style", "background", "lighting", "mood")):
        return "style_mismatch"
    if any(word in lowered for word in ("artifact", "distorted", "blurry", "deformed")):
        return "artifact"
    return "wrong_attribute"


def _normalize_error_type(value: Any) -> str:
    value = str(value or "").strip()
    if value in ERROR_TYPE_TO_ASPECT:
        return value
    return _classify_error(value)


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
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
