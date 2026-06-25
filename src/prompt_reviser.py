"""Idea2Img-style prompt generation and revision behind LLM adapters."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .clients import LLMClient
from .prompt_constraints import extract_constraints, make_constraints_context


class PromptReviser:
    """Generate initial prompts and revisions using a mockable LLM adapter."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate_initial_prompts(
        self,
        user_prompt: str,
        *,
        n: int = 3,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        """Port Idea2Img's initial prompt generator through ``LLMClient.text``."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        if n < 1:
            raise ValueError("n must be at least 1")
        request = _initial_prompt_request(user_prompt, n, history or [])
        response = self.llm.text(request)
        prompts = _parse_prompt_list(response)
        return _ensure_prompt_count(prompts, n, fallback=user_prompt)

    def revise(
        self,
        user_prompt: str,
        current_prompt: str,
        critique: Mapping[str, Any] | str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Return the first revised prompt for the next generation round."""

        return self.revise_candidates(
            user_prompt=user_prompt,
            current_prompt=current_prompt,
            critique=critique,
            history=history,
            n=1,
        )[0]

    def revise_candidates(
        self,
        user_prompt: str,
        current_prompt: str,
        critique: Mapping[str, Any] | str,
        history: Sequence[Mapping[str, Any]] | None = None,
        *,
        n: int = 1,
    ) -> list[str]:
        """Port Idea2Img's revision prompt through ``LLMClient.text``."""

        user_prompt = _clean_text(user_prompt, "user_prompt")
        current_prompt = _clean_text(current_prompt, "current_prompt")
        if n < 1:
            raise ValueError("n must be at least 1")
        critique_record = _normalize_critique(critique)
        history_records = [deepcopy(dict(item)) for item in history or []]

        request = _revision_prompt_request(
            user_prompt=user_prompt,
            current_prompt=current_prompt,
            critique=critique_record,
            history=history_records,
            n=n,
        )
        response = self.llm.text(request)
        prompts = _parse_prompt_list(response)
        fallback = _fallback_revision(current_prompt, critique_record)
        return _ensure_prompt_count(prompts, n, fallback=fallback)


def _initial_prompt_request(
    user_prompt: str, n: int, history: Sequence[Mapping[str, Any]]
) -> str:
    history_blob = json.dumps(list(history), ensure_ascii=False, sort_keys=True)
    constraints = extract_constraints(user_prompt)
    return "\n".join(
        [
            "You are the PromptAgent for an Idea2Img-style T2I loop.",
            "Convert the user's imagined IDEA into self-contained image prompts.",
            make_constraints_context(constraints),
            "Rules:",
            "- Original user constraints outrank any aesthetic expansion.",
            "- Preserve user-specified colors exactly, especially object colors.",
            "- Preserve user-specified subjects, actions, and spatial relations.",
            "- Describe the concrete scene first, then add comma-separated modifiers.",
            "- Add mood, style, lighting, spatial details, and visual attributes.",
            "- Reduce abstract psychological descriptions.",
            "- Explain unusual entities as visible scene details.",
            "- Do not mention 'given image'; describe visual references in words.",
            "- Keep each prompt within 77 CLIP-style tokens for SDXL.",
            "- Return JSON with a prompts list, or wrap each prompt with <START>/<END>.",
            f"IDEA: {user_prompt}",
            f"History: {history_blob}",
            f"Write exactly {n} diverse prompts.",
        ]
    )


def _revision_prompt_request(
    *,
    user_prompt: str,
    current_prompt: str,
    critique: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    n: int,
) -> str:
    critique_blob = json.dumps(critique, ensure_ascii=False, sort_keys=True)
    history_blob = json.dumps(list(history), ensure_ascii=False, sort_keys=True)
    constraints = extract_constraints(user_prompt)
    return "\n".join(
        [
            "You are the PromptAgent revising prompts in an Idea2Img-style loop.",
            "Improve the current generation prompt using visual feedback.",
            make_constraints_context(constraints),
            "Rules:",
            "- Fix failures against the original user IDEA before polishing details.",
            "- Never alter original user colors, subjects, actions, or relations.",
            "- Address the key reason from the critique.",
            "- Focus on one main improvement when possible.",
            "- Avoid prompts identical to previous rounds.",
            "- Keep each prompt within 77 CLIP-style tokens for SDXL.",
            "- Return JSON with a revised_prompt or prompts list, or use <START>/<END>.",
            f"IDEA: {user_prompt}",
            f"Current prompt: {current_prompt}",
            f"Visual critique: {critique_blob}",
            f"History: {history_blob}",
            f"Write exactly {n} revised prompt(s).",
        ]
    )


def _parse_prompt_list(response: str) -> list[str]:
    data = _extract_json_object(response)
    if data:
        if isinstance(data.get("prompts"), list):
            return [_clean_prompt_candidate(item) for item in data["prompts"]]
        if data.get("revised_prompt"):
            return [_clean_prompt_candidate(data["revised_prompt"])]
        if data.get("prompt"):
            return [_clean_prompt_candidate(data["prompt"])]

    tagged = [
        _clean_prompt_candidate(match)
        for match in re.findall(
            r"<START>(.*?)</?END>", response, flags=re.IGNORECASE | re.DOTALL
        )
    ]
    if tagged:
        return [prompt for prompt in tagged if prompt]

    lines = [
        _clean_prompt_candidate(line)
        for line in response.splitlines()
        if _clean_prompt_candidate(line)
    ]
    return lines[:1]


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


def _normalize_critique(critique: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(critique, Mapping):
        record = deepcopy(dict(critique))
    elif isinstance(critique, str):
        record = {"revision_hint": _clean_text(critique, "critique")}
    else:
        raise TypeError("critique must be a mapping or string")

    record.setdefault("score", 0.0)
    record.setdefault("errors", [])
    record.setdefault("strengths", [])
    record.setdefault("revision_hint", "")
    return record


def _fallback_revision(current_prompt: str, critique: Mapping[str, Any]) -> str:
    hint = str(critique.get("revision_hint", "")).strip()
    if not hint:
        errors = critique.get("errors", [])
        if errors:
            first_error = errors[0]
            if isinstance(first_error, Mapping):
                hint = str(first_error.get("evidence", "")).strip()
            else:
                hint = str(first_error).strip()
    if not hint:
        return current_prompt
    return f"{current_prompt}, refined to address: {hint}"


def _ensure_prompt_count(prompts: Sequence[str], n: int, *, fallback: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        prompt = _clean_prompt_candidate(prompt)
        key = prompt.lower()
        if prompt and key not in seen:
            cleaned.append(prompt)
            seen.add(key)
        if len(cleaned) == n:
            return cleaned
    while len(cleaned) < n:
        cleaned.append(fallback)
    return cleaned


def _clean_prompt_candidate(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^\s*(?:\d+[\).\s-]+|[-*]\s+)", "", text)
    text = re.sub(r"^(?:prompt|revised prompt)\s*[:=-]\s*", "", text, flags=re.I)
    text = " ".join(text.split())
    return text


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value
