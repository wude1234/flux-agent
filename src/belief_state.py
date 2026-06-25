"""Belief-state schema for proactive text-to-image clarification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    """A possible value for an attribute or relation."""

    name: str
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, "candidate name"))
        object.__setattr__(self, "probability", _clamp_probability(self.probability))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "probability": self.probability}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        return cls(name=str(data["name"]), probability=float(data.get("probability", 0)))


@dataclass(frozen=True)
class Attribute:
    """An uncertain attribute attached to an entity or background slot."""

    name: str
    candidates: list[Candidate] = field(default_factory=list)
    importance_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, "attribute name"))
        object.__setattr__(self, "importance_score", _clamp_probability(self.importance_score))
        object.__setattr__(self, "candidates", _normalize_candidates(self.candidates))

    @property
    def entropy(self) -> float:
        return normalized_entropy([candidate.probability for candidate in self.candidates])

    @property
    def ask_score(self) -> float:
        return round(self.importance_score * self.entropy, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "importance_score": self.importance_score,
            "entropy": self.entropy,
            "ask_score": self.ask_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Attribute":
        raw_candidates = data.get("candidates", data.get("value", []))
        if isinstance(raw_candidates, Mapping):
            raw_candidates = [
                {"name": name, "probability": probability}
                for name, probability in raw_candidates.items()
            ]
        return cls(
            name=str(data["name"]),
            candidates=[
                candidate if isinstance(candidate, Candidate) else Candidate.from_dict(candidate)
                for candidate in raw_candidates
            ],
            importance_score=float(data.get("importance_score", data.get("importance_to_ask_score", 0))),
        )


@dataclass(frozen=True)
class Relation(Attribute):
    """An uncertain relation between two entities."""

    name_entity_1: str = ""
    name_entity_2: str = ""
    description: str = ""
    is_bidirectional: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "name_entity_1", str(self.name_entity_1 or "").strip())
        object.__setattr__(self, "name_entity_2", str(self.name_entity_2 or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "is_bidirectional", bool(self.is_bidirectional))

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "name_entity_1": self.name_entity_1,
                "name_entity_2": self.name_entity_2,
                "description": self.description,
                "is_bidirectional": self.is_bidirectional,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Relation":
        raw_candidates = data.get("candidates", data.get("value", data.get("spatial_relation", [])))
        if isinstance(raw_candidates, Mapping):
            raw_candidates = [
                {"name": name, "probability": probability}
                for name, probability in raw_candidates.items()
            ]
        return cls(
            name=str(data["name"]),
            candidates=[
                candidate if isinstance(candidate, Candidate) else Candidate.from_dict(candidate)
                for candidate in raw_candidates
            ],
            importance_score=float(data.get("importance_score", data.get("importance_to_ask_score", 0))),
            name_entity_1=str(data.get("name_entity_1", "")),
            name_entity_2=str(data.get("name_entity_2", "")),
            description=str(data.get("description", "")),
            is_bidirectional=bool(data.get("is_bidirectional", False)),
        )


@dataclass(frozen=True)
class Entity:
    """A possible scene entity in the user's intended image."""

    name: str
    importance_score: float = 0.0
    attributes: list[Attribute] = field(default_factory=list)
    descriptions: str = ""
    entity_type: str = "explicit"
    probability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, "entity name"))
        object.__setattr__(self, "importance_score", _clamp_probability(self.importance_score))
        object.__setattr__(self, "probability", _clamp_probability(self.probability))
        object.__setattr__(self, "descriptions", str(self.descriptions or "").strip())
        object.__setattr__(self, "entity_type", str(self.entity_type or "explicit").strip())
        object.__setattr__(
            self,
            "attributes",
            [item if isinstance(item, Attribute) else Attribute.from_dict(item) for item in self.attributes],
        )

    @property
    def existence_entropy(self) -> float:
        return normalized_entropy([self.probability, 1.0 - self.probability])

    @property
    def ask_score(self) -> float:
        return round(self.importance_score * self.existence_entropy, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "importance_score": self.importance_score,
            "attributes": [attribute.to_dict() for attribute in self.attributes],
            "descriptions": self.descriptions,
            "entity_type": self.entity_type,
            "probability": self.probability,
            "existence_entropy": self.existence_entropy,
            "ask_score": self.ask_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        return cls(
            name=str(data["name"]),
            importance_score=float(data.get("importance_score", data.get("importance_to_ask_score", 0))),
            attributes=[
                item if isinstance(item, Attribute) else Attribute.from_dict(item)
                for item in data.get("attributes", [])
            ],
            descriptions=str(data.get("descriptions", data.get("description", ""))),
            entity_type=str(data.get("entity_type", "explicit")),
            probability=float(data.get("probability", data.get("probability_of_appearing", 1))),
        )


@dataclass(frozen=True)
class BeliefState:
    """Probabilistic scene graph used by the proactive clarifier."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BeliefState":
        raw_entities = data.get("entities", data.get("all_entities", []))
        raw_relations = data.get("relations", data.get("all_relations", []))
        return cls(
            entities=[
                item if isinstance(item, Entity) else Entity.from_dict(item)
                for item in raw_entities
            ],
            relations=[
                item if isinstance(item, Relation) else Relation.from_dict(item)
                for item in raw_relations
            ],
            prompt=data.get("prompt"),
        )

    def clarification_targets(self) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        for entity in self.entities:
            if entity.ask_score > 0:
                targets.append(
                    {
                        "kind": "entity",
                        "entity": entity.name,
                        "missing_slot": "existence",
                        "ask_score": entity.ask_score,
                        "importance_score": entity.importance_score,
                        "entropy": entity.existence_entropy,
                        "candidates": [
                            {"name": "yes", "probability": entity.probability},
                            {"name": "no", "probability": 1.0 - entity.probability},
                        ],
                    }
                )
            for attribute in entity.attributes:
                if attribute.ask_score > 0:
                    targets.append(
                        {
                            "kind": "attribute",
                            "entity": entity.name,
                            "attribute": attribute.name,
                            "missing_slot": attribute.name,
                            "ask_score": attribute.ask_score,
                            "importance_score": attribute.importance_score,
                            "entropy": attribute.entropy,
                            "candidates": [candidate.to_dict() for candidate in attribute.candidates],
                        }
                    )
        for relation in self.relations:
            if relation.ask_score > 0:
                targets.append(
                    {
                        "kind": "relation",
                        "relation": relation.name,
                        "entity_1": relation.name_entity_1,
                        "entity_2": relation.name_entity_2,
                        "missing_slot": "relation",
                        "ask_score": relation.ask_score,
                        "importance_score": relation.importance_score,
                        "entropy": relation.entropy,
                        "candidates": [candidate.to_dict() for candidate in relation.candidates],
                    }
                )
        targets.sort(key=lambda item: item["ask_score"], reverse=True)
        return targets


def normalized_entropy(probabilities: Sequence[float]) -> float:
    probs = [_clamp_probability(probability) for probability in probabilities]
    total = sum(probs)
    if total <= 0 or len(probs) <= 1:
        return 0.0
    probs = [prob / total for prob in probs]
    entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return round(entropy / max_entropy, 6)


def heuristic_belief_state(prompt: str) -> BeliefState:
    """Build a small fallback belief state when no LLM JSON is available."""

    prompt = _clean_text(prompt, "prompt")
    lowered = prompt.lower()
    entities: list[Entity] = []

    if "scientist" in lowered and "unusual object" in lowered:
        entities.append(
            Entity(
                name="unusual object",
                importance_score=0.95,
                descriptions="central object held by the scientist",
                entity_type="explicit",
                probability=1.0,
                attributes=[
                    Attribute(
                        name="identity",
                        importance_score=0.95,
                        candidates=[
                            Candidate("strange crystal device", 0.34),
                            Candidate("alien artifact", 0.33),
                            Candidate("experimental instrument", 0.33),
                        ],
                    )
                ],
            )
        )

    if any(term in lowered for term in ("city", "room", "landscape", "breakfast", "table", "office")):
        subject = _first_matching(
            lowered,
            ["futuristic city", "city", "room", "landscape", "breakfast", "table", "office"],
        )
        entities.append(
            Entity(
                name=subject,
                importance_score=0.8,
                descriptions=f"main scene: {subject}",
                entity_type="explicit",
                probability=1.0,
                attributes=[],
            )
        )

    background_attributes: list[Attribute] = []
    if not _mentions_style(lowered):
        background_attributes.append(
            Attribute(
                name="image style",
                importance_score=0.9,
                candidates=[
                    Candidate("cinematic realistic", 0.34),
                    Candidate("illustration", 0.33),
                    Candidate("minimal concept art", 0.33),
                ],
            )
        )
    if not _mentions_viewpoint(lowered):
        background_attributes.append(
            Attribute(
                name="viewpoint",
                importance_score=0.65,
                candidates=[
                    Candidate("wide establishing view", 0.34),
                    Candidate("street-level view", 0.33),
                    Candidate("close-up", 0.33),
                ],
            )
        )
    if not _mentions_mood(lowered):
        background_attributes.append(
            Attribute(
                name="mood",
                importance_score=0.55,
                candidates=[
                    Candidate("calm", 0.34),
                    Candidate("dramatic", 0.33),
                    Candidate("mysterious", 0.33),
                ],
            )
        )

    if background_attributes:
        entities.append(
            Entity(
                name="overall image",
                importance_score=0.2,
                attributes=background_attributes,
                descriptions="global visual presentation",
                entity_type="background",
                probability=1.0,
            )
        )

    if not entities:
        entities.append(
            Entity(
                name="overall image",
                importance_score=0.3,
                attributes=[],
                descriptions="prompt appears visually specified",
                entity_type="background",
                probability=1.0,
            )
        )
    return BeliefState(entities=entities, relations=[], prompt=prompt)


def _normalize_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    items = [item if isinstance(item, Candidate) else Candidate.from_dict(item) for item in candidates]
    total = sum(item.probability for item in items)
    if not items or total <= 0:
        return items
    return [Candidate(item.name, item.probability / total) for item in items]


def _mentions_style(text: str) -> bool:
    return any(
        word in text
        for word in (
            "cinematic",
            "realistic",
            "photo",
            "photoreal",
            "illustration",
            "anime",
            "watercolor",
            "oil painting",
            "concept art",
        )
    )


def _mentions_viewpoint(text: str) -> bool:
    return any(word in text for word in ("close-up", "wide", "aerial", "front view", "viewpoint", "portrait"))


def _mentions_mood(text: str) -> bool:
    return any(word in text for word in ("night", "day", "sunset", "rainy", "dramatic", "peaceful", "mysterious"))


def _first_matching(text: str, options: Sequence[str]) -> str:
    for option in options:
        if option in text:
            return option
    return options[0]


def _clamp_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value

