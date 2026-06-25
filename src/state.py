"""Shared state primitives for the M0 multimodal T2I agent skeleton."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Iterable, Mapping


class CreativityLevel(str, Enum):
    """How aggressively the agent may fill missing visual details."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AgentConfig:
    """Small, adapter-friendly subset of the upstream T2I-Copilot config."""

    human_in_loop: bool = True
    creativity_level: CreativityLevel = CreativityLevel.MEDIUM
    n_images: int = 1
    max_rounds: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "creativity_level", CreativityLevel(self.creativity_level)
        )
        if self.n_images < 1:
            raise ValueError("n_images must be at least 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "human_in_loop": self.human_in_loop,
            "creativity_level": self.creativity_level.value,
            "n_images": self.n_images,
            "max_rounds": self.max_rounds,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        return cls(
            human_in_loop=bool(data.get("human_in_loop", True)),
            creativity_level=CreativityLevel(
                data.get("creativity_level", CreativityLevel.MEDIUM)
            ),
            n_images=int(data.get("n_images", 1)),
            max_rounds=int(data.get("max_rounds", 1)),
            seed=data.get("seed"),
        )


@dataclass
class AgentState:
    """Explicit shared state passed between agents and tools.

    M0 keeps the surface intentionally small while preserving the useful
    T2I-Copilot concepts: original prompt, refined generation prompt,
    regeneration rounds, generated image paths, feedback, candidates, and memory.
    """

    user_prompt: str
    refined_prompt: str | None = None
    image_paths: list[str] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    round_index: int = 0
    human_in_loop: bool = True
    creativity_level: CreativityLevel = CreativityLevel.MEDIUM
    seed: int | None = None

    def __post_init__(self) -> None:
        self.user_prompt = _clean_prompt(self.user_prompt, "user_prompt")
        if self.refined_prompt is not None:
            self.refined_prompt = _clean_prompt(self.refined_prompt, "refined_prompt")
        self.image_paths = _coerce_str_list(self.image_paths, "image_paths")
        self.feedback = _coerce_record_list(self.feedback, "feedback")
        self.candidates = _coerce_record_list(self.candidates, "candidates")
        self.memory = _coerce_record_list(self.memory, "memory")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        self.creativity_level = CreativityLevel(self.creativity_level)

    @property
    def active_prompt(self) -> str:
        """Return the prompt that should currently be sent to generation."""

        return self.refined_prompt or self.user_prompt

    @classmethod
    def from_config(cls, user_prompt: str, config: AgentConfig) -> "AgentState":
        return cls(
            user_prompt=user_prompt,
            human_in_loop=config.human_in_loop,
            creativity_level=config.creativity_level,
            seed=config.seed,
        )

    def add_candidate(
        self, candidate: Mapping[str, Any] | str, **metadata: Any
    ) -> dict[str, Any]:
        """Append a prompt candidate and return the stored copy."""

        if isinstance(candidate, str):
            record = {"prompt": _clean_prompt(candidate, "candidate prompt")}
        else:
            record = deepcopy(dict(candidate))
            record["prompt"] = _clean_prompt(record.get("prompt", ""), "candidate")
        record.update(deepcopy(metadata))
        self.candidates.append(record)
        return deepcopy(record)

    def select_candidate(self, index: int = 0) -> str:
        """Set ``refined_prompt`` from a candidate and return it."""

        try:
            prompt = self.candidates[index]["prompt"]
        except IndexError as exc:
            raise IndexError(f"candidate index out of range: {index}") from exc
        except KeyError as exc:
            raise ValueError(f"candidate at index {index} has no prompt") from exc

        self.refined_prompt = _clean_prompt(prompt, "selected candidate prompt")
        return self.refined_prompt

    def add_images(self, image_paths: Iterable[str] | str) -> list[str]:
        paths = _coerce_str_list(image_paths, "image_paths")
        self.image_paths.extend(paths)
        return list(paths)

    def add_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        record = deepcopy(dict(feedback))
        self.feedback.append(record)
        return deepcopy(record)

    def remember(self, record: Mapping[str, Any]) -> dict[str, Any]:
        item = deepcopy(dict(record))
        self.memory.append(item)
        return deepcopy(item)

    def advance_round(self) -> int:
        self.round_index += 1
        return self.round_index

    def apply_update(self, update: Mapping[str, Any]) -> "AgentState":
        """Apply a validated update dict to this state.

        This gives later agents a small common state-update hook without adding
        an orchestrator in M0.
        """

        valid_fields = {item.name for item in fields(self)}
        for key, value in update.items():
            if key not in valid_fields:
                raise KeyError(f"unknown AgentState field: {key}")
            setattr(self, key, deepcopy(value))
        self.__post_init__()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_prompt": self.user_prompt,
            "refined_prompt": self.refined_prompt,
            "image_paths": list(self.image_paths),
            "feedback": deepcopy(self.feedback),
            "candidates": deepcopy(self.candidates),
            "memory": deepcopy(self.memory),
            "round_index": self.round_index,
            "human_in_loop": self.human_in_loop,
            "creativity_level": self.creativity_level.value,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentState":
        return cls(
            user_prompt=data["user_prompt"],
            refined_prompt=data.get("refined_prompt"),
            image_paths=list(data.get("image_paths", [])),
            feedback=list(data.get("feedback", [])),
            candidates=list(data.get("candidates", [])),
            memory=list(data.get("memory", [])),
            round_index=int(data.get("round_index", 0)),
            human_in_loop=bool(data.get("human_in_loop", True)),
            creativity_level=CreativityLevel(
                data.get("creativity_level", CreativityLevel.MEDIUM)
            ),
            seed=data.get("seed"),
        )


def _clean_prompt(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _coerce_str_list(value: Iterable[str] | str, field_name: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} entries must be strings")
        if not item:
            raise ValueError(f"{field_name} entries must not be empty")
        result.append(item)
    return result


def _coerce_record_list(
    records: Iterable[Mapping[str, Any]], field_name: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be mappings")
        result.append(deepcopy(dict(item)))
    return result

