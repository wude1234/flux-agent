"""User-intent constraints and prompt-budget utilities for M4.1."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CLIP_TOKEN_BUDGET = 77

COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "grey",
    "purple",
    "pink",
    "orange",
    "brown",
    "lavender",
    "amber",
    "navy",
    "mint",
    "ruby",
    "bronze",
    "lime",
    "maroon",
    "ivory",
    "olive",
    "violet",
    "aqua",
    "crimson",
    "silver",
    "gold",
    "golden",
    "teal",
    "cyan",
    "magenta",
    "indigo",
    "turquoise",
}

MATERIAL_WORDS = {
    "ceramic",
    "cloth",
    "cardboard",
    "fabric",
    "glass",
    "leather",
    "marble",
    "metal",
    "metallic",
    "paper",
    "plastic",
    "porcelain",
    "rubber",
    "steel",
    "stone",
    "velvet",
    "wax",
    "wood",
    "wooden",
}

ATTRIBUTE_WORDS = {
    "checkered",
    "clear",
    "huge",
    "matte",
    "plaid",
    "shiny",
    "striped",
    "tiny",
    "transparent",
    *MATERIAL_WORDS,
}

ACTION_PATTERNS = (
    r"\bhides?\b",
    r"\bcovers?\b",
    r"\boccluding\b",
    r"\boccludes?\b",
    r"\b(?:clearly\s+)?gripping\b",
    r"\bgrips?\b",
    r"\bgrasping\b",
    r"\bgrasps?\b",
    r"\bholds?\b",
    r"\bholding\b",
    r"\btouching\b",
    r"\btouches?\b",
    r"\battached to\b",
    r"\battaches?\s+to\b",
    r"\bwearing\b",
    r"\bwears?\b",
    r"\bshowing\b",
    r"\bshows?\b",
    r"\bdisplaying\b",
    r"\bdisplays?\b",
    r"\bsits?\b",
    r"\bperches?\b",
    r"\bstanding\b",
    r"\bsitting\b",
    r"\briding\b",
    r"\brides?\b",
    r"\bcarrying\b",
    r"\bcarries?\b",
    r"\bbalancing\b",
)

RELATION_PATTERNS = (
    r"\bnext to\b",
    r"\bin front of\b",
    r"\bbehind\b",
    r"\babove\b",
    r"\bunder\b",
    r"\bon top of\b",
    r"\bleft of\b",
    r"\bright of\b",
)

DISPLAY_ACTIONS = {
    "show",
    "shows",
    "showing",
    "display",
    "displays",
    "displaying",
}

DISPLAY_CONTAINER_RELATIONS = {
    "next to",
    "in front of",
    "behind",
    "left of",
    "right of",
}

DISPLAY_SURFACE_WORDS = {
    "badge",
    "cover",
    "front",
    "label",
    "logo",
    "page",
    "screen",
    "shirt",
    "sign",
    "surface",
}

RELATION_CONNECTOR_WORDS = {
    "above",
    "across",
    "adjacent",
    "after",
    "against",
    "along",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "front",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "next",
    "left",
    "of",
    "on",
    "onto",
    "outside",
    "over",
    "right",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "with",
    "within",
}

PROTECTED_STYLE_TERMS = {
    "cinematic",
    "photo",
    "photograph",
    "rainy",
    "street",
}

SUBJECT_STOP_WORDS = {
    "and",
    "are",
    "attached",
    "balancing",
    "contain",
    "contains",
    "containing",
    "carries",
    "carry",
    "be",
    "been",
    "being",
    "carrying",
    "clearly",
    "cover",
    "covers",
    "covered",
    "covering",
    "grips",
    "gripping",
    "grasps",
    "grasping",
    "hold",
    "holds",
    "holding",
    "hide",
    "hides",
    "hidden",
    "hiding",
    "has",
    "have",
    "having",
    "is",
    "labeled",
    "labelled",
    "located",
    "placed",
    "positioned",
    "perch",
    "perches",
    "perching",
    "read",
    "reads",
    "reading",
    "remain",
    "remains",
    "show",
    "shows",
    "showing",
    "display",
    "displays",
    "displaying",
    "sit",
    "sits",
    "sitting",
    "stay",
    "stays",
    "standing",
    "touches",
    "touching",
    "wears",
    "wearing",
    "rides",
    "riding",
    "attaches",
    "was",
    "were",
    "while",
    "must",
    "no",
    "not",
    *RELATION_CONNECTOR_WORDS,
}

SUBJECT_TRAILING_ACTION_WORDS = {
    "is",
    "are",
    "sit",
    "sits",
    "sitting",
    "stand",
    "stands",
    "standing",
    "rest",
    "rests",
    "resting",
    "swim",
    "swims",
    "swimming",
    "perch",
    "perches",
    "perching",
    "lie",
    "lies",
    "lying",
    "hang",
    "hangs",
    "hanging",
}

SUBJECT_DROP_WORDS = {
    "a",
    "an",
    "the",
    "small",
    "large",
    "tiny",
    "big",
    "visible",
    "clear",
    "clean",
    "studio",
    "realistic",
    "outdoor",
    "plain",
    "transparent",
    "wooden",
    "color",
    "exact",
    "leakage",
    *COLOR_WORDS,
}

INVALID_SUBJECTS = {
    "wooden",
    "small",
    "large",
    "tiny",
    "big",
    "clean",
    "object",
    "objects",
    "studio",
    "realistic",
    "outdoor",
    *RELATION_CONNECTOR_WORDS,
    "cabinet reads",
    "has",
    "has no",
    "has no window",
    "lower",
    "lower half",
    "upper",
    "upper half",
    "left half",
    "right half",
    "bottom half",
    "top half",
    "patterns must not",
    "must not",
}

NUMBER_WORDS_FOR_SUBJECTS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}

COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

SUBJECT_DROP_WORDS.update(NUMBER_WORDS_FOR_SUBJECTS)
INVALID_SUBJECTS.update(NUMBER_WORDS_FOR_SUBJECTS)

LOW_PRIORITY_TERMS = [
    "hyperrealistic detail",
    "35mm film grain",
    "3 5 mm film grain",
    "volumetric rain mist",
    "moody indigo twilight",
    "cinematic shallow depth of field",
    "shallow depth of field",
]


@dataclass(frozen=True)
class IntentSpec:
    """Structured user intent used by the FLUX-first agent loop."""

    original_prompt: str
    subjects: list[str] = field(default_factory=list)
    colors: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    interaction_relations: list[dict[str, str]] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    style: str = ""
    background: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "subjects": list(self.subjects),
            "colors": dict(self.colors),
            "attributes": {
                str(key): list(value) for key, value in self.attributes.items()
            },
            "counts": dict(self.counts),
            "actions": list(self.actions),
            "relations": [dict(item) for item in self.relations],
            "interaction_relations": [
                dict(item) for item in self.interaction_relations
            ],
            "negative_constraints": list(self.negative_constraints),
            "style": self.style,
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntentSpec":
        return cls(
            original_prompt=str(value.get("original_prompt", "")),
            subjects=[str(item) for item in value.get("subjects", [])],
            colors=dict(value.get("colors", {})),
            attributes={
                str(key): [str(item) for item in items]
                for key, items in dict(value.get("attributes", {})).items()
                if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
            },
            counts={
                str(key): int(count)
                for key, count in dict(value.get("counts", {})).items()
            },
            actions=[str(item) for item in value.get("actions", [])],
            relations=[
                {str(key): str(val) for key, val in dict(item).items()}
                for item in value.get("relations", [])
                if isinstance(item, Mapping)
            ],
            interaction_relations=[
                {str(key): str(val) for key, val in dict(item).items()}
                for item in value.get("interaction_relations", [])
                if isinstance(item, Mapping)
            ],
            negative_constraints=[
                str(item) for item in value.get("negative_constraints", [])
            ],
            style=str(value.get("style", "")),
            background=str(value.get("background", "")),
        )


@dataclass(frozen=True)
class PromptConstraints:
    """Lightweight extracted constraints from the original user prompt."""

    original_prompt: str
    colors: dict[str, str] = field(default_factory=dict)
    subjects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    protected_phrases: list[str] = field(default_factory=list)
    intent_spec: IntentSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "colors": dict(self.colors),
            "subjects": list(self.subjects),
            "actions": list(self.actions),
            "relations": list(self.relations),
            "protected_phrases": list(self.protected_phrases),
            "intent_spec": self.intent_spec.to_dict() if self.intent_spec else None,
        }


def extract_constraints(user_prompt: str) -> PromptConstraints:
    """Extract conservative color/entity/action/relation constraints."""

    intent = extract_intent_spec(user_prompt)
    relation_phrases = [
        item["phrase"] for item in intent.relations if item.get("phrase")
    ]
    protected = _protected_phrases(
        intent.original_prompt,
        intent.colors,
        intent.actions,
        relation_phrases,
    )
    return PromptConstraints(
        original_prompt=intent.original_prompt,
        colors=intent.colors,
        subjects=intent.subjects,
        actions=intent.actions,
        relations=relation_phrases,
        protected_phrases=protected,
        intent_spec=intent,
    )


def extract_intent_spec(user_prompt: str) -> IntentSpec:
    """Extract structured intent without letting grammar fragments become subjects."""

    prompt = _clean_text(user_prompt)
    lowered = prompt.lower()
    colors = _extract_color_constraints(lowered)
    subjects = _extract_subjects(lowered, colors)
    attributes = _extract_attribute_constraints(lowered, colors, subjects)
    actions = _extract_patterns(lowered, ACTION_PATTERNS)
    occlusion_relations = _extract_occlusion_relation_specs(
        lowered,
        subjects,
        colors,
    )
    interaction_relations = _extract_interaction_relation_specs(
        lowered,
        subjects,
        colors,
    )
    interaction_relations = _dedupe_relation_specs(
        [*occlusion_relations, *interaction_relations]
    )
    for relation in interaction_relations:
        target = relation.get("object", "")
        if target and target not in subjects and not _is_part_of_known_object(target, colors, subjects):
            subjects.append(target)
    subjects = _remove_covered_part_subjects(subjects, colors, interaction_relations)
    subjects = _remove_occlusion_hidden_part_subjects(subjects, interaction_relations)
    subjects = _remove_negative_clause_subjects(subjects, prompt)
    relations = _extract_relation_specs(
        lowered,
        subjects,
        interaction_relations=interaction_relations,
    )
    return IntentSpec(
        original_prompt=prompt,
        subjects=subjects,
        colors=colors,
        attributes=attributes,
        counts=_extract_count_constraints(lowered),
        actions=actions,
        relations=relations,
        interaction_relations=interaction_relations,
        negative_constraints=_extract_negative_constraints(prompt),
        style=_extract_style(prompt),
        background=_extract_background(prompt),
    )


def lock_prompt_to_user_constraints(
    prompt: str,
    constraints: PromptConstraints | Mapping[str, Any] | str,
    *,
    token_budget: int = DEFAULT_CLIP_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Preserve original user constraints and fit a CLIP-style token budget."""

    constraints = _ensure_constraints(constraints)
    original_prompt = _clean_text(prompt)
    locked_prompt = original_prompt
    applied: list[str] = []
    warnings: list[str] = []

    locked_prompt, removed_meta = _remove_meta_constraint_segments(locked_prompt)
    if removed_meta:
        applied.extend(f"removed meta constraint segment: {item}" for item in removed_meta)

    for object_name, color in constraints.colors.items():
        if _is_ambiguous_generic_color_binding(object_name, color, constraints.colors):
            applied.append(f"skipped ambiguous generic color binding: {color} {object_name}")
            continue
        locked_prompt, color_actions = _protect_color(
            locked_prompt,
            object_name=object_name,
            color=color,
        )
        applied.extend(color_actions)

    for phrase in constraints.protected_phrases:
        if phrase.lower() not in locked_prompt.lower():
            locked_prompt = f"{phrase}, {locked_prompt}"
            applied.append(f"restored protected phrase: {phrase}")

    locked_prompt = _normalize_spacing(locked_prompt)
    if approx_clip_token_count(locked_prompt) > token_budget:
        locked_prompt, truncate_actions = truncate_prompt_for_clip(
            locked_prompt,
            constraints,
            token_budget=token_budget,
        )
        applied.extend(truncate_actions)
    if approx_clip_token_count(locked_prompt) > token_budget:
        warnings.append(
            f"prompt still exceeds token budget: {approx_clip_token_count(locked_prompt)}>{token_budget}"
        )

    violations = constraint_violations(locked_prompt, constraints)
    return {
        "prompt": locked_prompt,
        "original_prompt": original_prompt,
        "token_count": approx_clip_token_count(locked_prompt),
        "token_budget": token_budget,
        "applied": applied,
        "violations": violations,
        "warnings": warnings,
        "constraints": constraints.to_dict(),
    }


def constraint_violations(
    prompt: str,
    constraints: PromptConstraints | Mapping[str, Any] | str,
) -> list[dict[str, str]]:
    constraints = _ensure_constraints(constraints)
    lowered = prompt.lower()
    violations: list[dict[str, str]] = []
    for object_name, color in constraints.colors.items():
        if _is_ambiguous_generic_color_binding(object_name, color, constraints.colors):
            continue
        object_terms = _object_terms(object_name)
        if color not in lowered or not any(term in lowered for term in object_terms):
            violations.append(
                {
                    "type": "missing_user_color",
                    "evidence": f"Expected {color} {object_name}.",
                    "prompt_span": object_name,
                }
            )
        elif _conflicting_color_near_object(lowered, object_name, color):
            violations.append(
                {
                    "type": "conflicting_user_color",
                    "evidence": f"Prompt may conflict with user color {color} for {object_name}.",
                    "prompt_span": object_name,
                }
            )
    for phrase in constraints.actions + constraints.relations:
        if phrase and phrase.lower() not in lowered:
            violations.append(
                {
                    "type": "missing_user_action_or_relation",
                    "evidence": f"Expected original user phrase: {phrase}.",
                    "prompt_span": phrase,
                }
            )
    return violations


def make_constraints_context(
    constraints: PromptConstraints | Mapping[str, Any] | str,
) -> str:
    constraints = _ensure_constraints(constraints)
    return (
        "Original user-intent constraints, highest priority: "
        f"{constraints.to_dict()}. Do not change these colors, subjects, "
        "actions, or relations when expanding or revising prompts."
    )


def approx_clip_token_count(prompt: str) -> int:
    """Approximate CLIP token count without loading a tokenizer."""

    # This is intentionally conservative enough to catch overlong SDXL prompts
    # in tests and CLI runs without importing transformers.
    return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", prompt))


def truncate_prompt_for_clip(
    prompt: str,
    constraints: PromptConstraints | Mapping[str, Any] | str,
    *,
    token_budget: int = DEFAULT_CLIP_TOKEN_BUDGET,
) -> tuple[str, list[str]]:
    constraints = _ensure_constraints(constraints)
    segments = _split_segments(prompt)
    protected_segments, optional_segments = _partition_segments(segments, constraints)
    actions: list[str] = []

    kept = list(protected_segments)
    for segment in optional_segments:
        candidate = _join_segments([*kept, segment])
        if approx_clip_token_count(candidate) <= token_budget:
            kept.append(segment)
        else:
            actions.append(f"dropped optional segment: {segment}")

    result = _join_segments(kept)
    while approx_clip_token_count(result) > token_budget and len(kept) > 1:
        removed = kept.pop()
        actions.append(f"dropped tail segment: {removed}")
        result = _join_segments(kept)

    return result, actions


def _extract_color_constraints(lowered: str) -> dict[str, str]:
    colors: dict[str, str] = {}
    color_alt = "|".join(sorted(COLOR_WORDS, key=len, reverse=True))
    stop_alt = "|".join(
        sorted(
            (SUBJECT_STOP_WORDS | RELATION_CONNECTOR_WORDS | COLOR_WORDS),
            key=len,
            reverse=True,
        )
    )
    noun_word = rf"(?!(?:{stop_alt})\b)[a-z0-9-]+"
    pattern = re.compile(
        rf"\b({color_alt})\s+((?:{noun_word}\s+){{0,3}}{noun_word})\b"
    )
    for match in pattern.finditer(lowered):
        color = match.group(1)
        noun_phrase = _trim_noun_phrase(match.group(2))
        if noun_phrase and noun_phrase not in PROTECTED_STYLE_TERMS:
            _add_color_constraint(colors, noun_phrase, color)
    return colors


def _extract_subjects(lowered: str, colors: Mapping[str, str]) -> list[str]:
    subjects = list(colors.keys())
    for pattern in (
        r"\ban?\s+small\s+([a-z0-9-]+(?:\s+[a-z0-9-]+){0,4})\b",
        r"\ban?\s+([a-z0-9-]+(?:\s+[a-z0-9-]+){0,4})\b",
        r"\bthe\s+([a-z0-9-]+(?:\s+[a-z0-9-]+){0,4})\b",
    ):
        for match in re.finditer(pattern, lowered):
            term = _trim_noun_phrase(match.group(1))
            if term and term not in subjects and term not in INVALID_SUBJECTS:
                subjects.append(term)
            for alias in _subject_aliases_from_attribute_phrase(term):
                if alias and alias not in subjects and alias not in INVALID_SUBJECTS:
                    subjects.append(alias)
    return subjects[:8]


def _extract_attribute_constraints(
    lowered: str,
    colors: Mapping[str, str],
    subjects: Sequence[str],
) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = {}
    known_subjects = list(dict.fromkeys([*colors.keys(), *subjects]))
    for subject in known_subjects:
        attrs = _attributes_from_subject_phrase(subject)
        object_name = _strip_leading_attributes(subject)
        if attrs and object_name:
            _add_attributes(attributes, object_name, attrs)
        if object_name and object_name != subject and subject in colors:
            _add_attributes(attributes, subject, attrs)
    for object_name in known_subjects:
        for attr in _attributes_near_object(lowered, object_name):
            _add_attributes(attributes, object_name, [attr])
    for object_name, attrs in list(attributes.items()):
        cleaned = [
            attr
            for attr in dict.fromkeys(attrs)
            if attr and attr not in COLOR_WORDS and attr not in NUMBER_WORDS_FOR_SUBJECTS
        ]
        if cleaned:
            attributes[object_name] = cleaned
        else:
            attributes.pop(object_name, None)
    return attributes


def _attributes_from_subject_phrase(subject: str) -> list[str]:
    parts = [part for part in _normalize_spacing(subject).lower().split() if part]
    if len(parts) < 2:
        return []
    return [part for part in parts[:-1] if part in ATTRIBUTE_WORDS]


def _subject_aliases_from_attribute_phrase(subject: str) -> list[str]:
    parts = [part for part in _normalize_spacing(subject).lower().split() if part]
    if len(parts) < 2:
        return []
    aliases: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part in MATERIAL_WORDS:
            aliases.append(" ".join(parts[index:]))
            break
    if not aliases and parts[0] in ATTRIBUTE_WORDS:
        aliases.append(parts[-1])
    return aliases


def _strip_leading_attributes(subject: str) -> str:
    parts = [part for part in _normalize_spacing(subject).lower().split() if part]
    while len(parts) > 1 and parts[0] in ATTRIBUTE_WORDS:
        parts.pop(0)
    return " ".join(parts)


def _attributes_near_object(lowered: str, object_name: str) -> list[str]:
    object_name = _normalize_spacing(object_name).lower()
    if not object_name:
        return []
    attrs: list[str] = []
    object_terms = _object_terms(object_name)
    attr_alt = "|".join(sorted(ATTRIBUTE_WORDS, key=len, reverse=True))
    for term in object_terms:
        term_pattern = re.escape(term)
        for match in re.finditer(
            rf"\b(?P<attrs>(?:(?:{attr_alt})\s+){{1,4}}){term_pattern}\b",
            lowered,
        ):
            attrs.extend(
                word
                for word in match.group("attrs").split()
                if word in ATTRIBUTE_WORDS
            )
    return attrs


def _add_attributes(
    attributes: dict[str, list[str]],
    object_name: str,
    values: Sequence[str],
) -> None:
    object_name = _normalize_spacing(object_name).lower()
    if not object_name:
        return
    bucket = attributes.setdefault(object_name, [])
    for value in values:
        value = _normalize_spacing(value).lower()
        if value and value not in bucket:
            bucket.append(value)


def _extract_count_constraints(lowered: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    count_alt = "|".join(sorted(COUNT_WORDS, key=len, reverse=True))
    stop_alt = "|".join(
        sorted(
            (
                SUBJECT_STOP_WORDS
                | RELATION_CONNECTOR_WORDS
                | set(COUNT_WORDS)
                | {"any", "extra", "both"}
            ),
            key=len,
            reverse=True,
        )
    )
    noun_word = rf"(?!(?:{stop_alt})\b)[a-z0-9-]+"
    pattern = re.compile(
        rf"\b(?P<count>\d+|{count_alt})\s+"
        rf"(?P<object>(?:{noun_word}\s+){{0,3}}{noun_word})\b"
    )
    for match in pattern.finditer(lowered):
        raw_count = match.group("count")
        count = int(raw_count) if raw_count.isdigit() else COUNT_WORDS.get(raw_count)
        if count is None:
            continue
        object_name = _trim_noun_phrase(match.group("object"))
        if not object_name or object_name in PROTECTED_STYLE_TERMS:
            continue
        counts[object_name] = count
    return counts


def _extract_relation_specs(
    lowered: str,
    subjects: Sequence[str],
    *,
    interaction_relations: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pattern in RELATION_PATTERNS:
        for match in re.finditer(pattern, lowered):
            phrase = match.group(0).strip()
            subject = _nearest_subject_before(lowered, match.start(), subjects)
            target = _nearest_subject_after(lowered, match.end(), subjects)
            subject = _display_container_subject_for_relation(
                lowered,
                match.start(),
                phrase,
                subject,
                interaction_relations,
            )
            key = (phrase, subject, target)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "phrase": phrase,
                    "subject": subject,
                    "object": target,
                    "type": "spatial",
                }
            )
    return relations


def _display_container_subject_for_relation(
    text: str,
    relation_index: int,
    phrase: str,
    subject: str,
    interaction_relations: Sequence[Mapping[str, str]],
) -> str:
    if not subject or not interaction_relations:
        return subject
    recent = _recent_sentence_prefix(text, relation_index)
    if not recent:
        return subject
    for relation in interaction_relations:
        action = relation.get("action", "").strip().lower()
        container = relation.get("subject", "").strip().lower()
        displayed = relation.get("object", "").strip().lower()
        if not (
            container
            and displayed
            and _is_display_action(action)
            and (_same_subject(subject, displayed) or _is_text_symbol_target(subject))
        ):
            continue
        if not _mentions_subject(recent, container):
            continue
        displayed_pos = _last_subject_mention(recent, displayed)
        if displayed_pos < 0 and _is_text_symbol_target(displayed):
            displayed_pos = _last_subject_mention(recent, "text")
        if displayed_pos < 0:
            continue
        trailing = recent[displayed_pos:]
        has_surface_cue = any(
            re.search(rf"\b{re.escape(word)}\b", trailing)
            for word in DISPLAY_SURFACE_WORDS
        )
        if phrase in DISPLAY_CONTAINER_RELATIONS or has_surface_cue:
            return container
        if _is_text_symbol_target(displayed) and phrase in {"above", "under", "left of", "right of"}:
            return container
    return subject


def _extract_interaction_relation_specs(
    lowered: str,
    subjects: Sequence[str],
    colors: Mapping[str, str],
) -> list[dict[str, str]]:
    action_alt = (
        "holding|holds?|gripping|grips?|grasping|grasps?|touching|touches?|"
        "attached to|attaches? to|carrying|carries?|wearing|wears?|"
        "showing|shows?|displaying|displays?|riding|rides?"
    )
    subject_alt = _subject_alternation(subjects)
    if subject_alt:
        pattern = re.compile(
            rf"\b(?:the\s+|a\s+|an\s+)?(?P<subject>{subject_alt})\s+"
            rf"(?:is\s+|are\s+|clearly\s+)?(?P<action>{action_alt})\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<object>(?:[a-z0-9-]+\s+){0,8}[a-z0-9-]+)\b"
        )
    else:
        pattern = re.compile(
            rf"\b(?P<subject>[a-z0-9-]+)\s+"
            rf"(?:is\s+|are\s+|clearly\s+)?(?P<action>{action_alt})\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<object>(?:[a-z0-9-]+\s+){0,8}[a-z0-9-]+)\b"
        )

    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in pattern.finditer(lowered):
        subject = _canonical_subject(match.group("subject"), subjects)
        action = match.group("action").strip()
        raw_target = match.group("object")
        target = _trim_interaction_target(raw_target)
        target = _expand_part_of_colored_object_target(raw_target, target, colors)
        if _is_display_action(action):
            target = _normalize_display_target(raw_target, target)
        if not subject or not action or not target:
            continue
        key = (subject, action, target)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            {
                "phrase": f"{subject} {action} {target}",
                "subject": subject,
                "action": action,
                "object": target,
                "type": "interaction",
            }
        )
        for chained in _extract_chained_interaction_specs(
            subject,
            raw_target,
            colors,
            action_alt,
        ):
            chain_key = (
                chained["subject"],
                chained["action"],
                chained["object"],
            )
            if chain_key in seen:
                continue
            seen.add(chain_key)
            relations.append(chained)
    return _dedupe_relation_specs(relations)


def _extract_occlusion_relation_specs(
    lowered: str,
    subjects: Sequence[str],
    colors: Mapping[str, str],
) -> list[dict[str, str]]:
    """Extract local occlusion constraints without creating pseudo-subjects.

    Prompts such as "a red screen hides the lower half of a green suitcase"
    are easy for noun-phrase regexes to misread as the colored object
    "screen hides lower".  Keeping them as typed relations lets VQA ask the
    right question and lets the repair planner target the occluder.
    """

    relations: list[dict[str, str]] = []
    object_phrase = r"(?:[a-z0-9-]+\s+){0,5}[a-z0-9-]+"
    hidden_part = (
        r"lower half|upper half|left half|right half|bottom half|top half|"
        r"lower part|upper part|left side|right side|part"
    )
    pattern = re.compile(
        rf"\b(?:the\s+|a\s+|an\s+)?(?P<occluder>{object_phrase})\s+"
        r"(?P<action>hides?|covers?|occludes?|occluding|covering)\s+"
        rf"(?:the\s+)?(?P<hidden_part>{hidden_part})\s+of\s+"
        rf"(?:the\s+|a\s+|an\s+)?(?P<target>{object_phrase})"
        rf"(?:,\s*while\s+(?P<visible_part>{object_phrase})\s+"
        r"(?:remains?|stays?|is|are)\s+(?:clearly\s+)?visible)?",
        re.I,
    )
    for match in pattern.finditer(lowered):
        raw_occluder = match.group("occluder")
        raw_target = match.group("target")
        hidden = _normalize_spacing(match.group("hidden_part")).lower()
        visible = _trim_noun_phrase(match.group("visible_part") or "")
        occluder = _canonical_colored_subject(raw_occluder, subjects, colors)
        target = _canonical_colored_subject(raw_target, subjects, colors)
        if not occluder or not target or occluder == target:
            continue
        relation: dict[str, str] = {
            "phrase": f"{occluder} {match.group('action').strip().lower()} {hidden} of {target}",
            "subject": occluder,
            "action": match.group("action").strip().lower(),
            "object": target,
            "type": "occlusion",
            "hidden_part": hidden,
        }
        if visible:
            visible = _expand_part_of_colored_object_target(
                match.group("visible_part") or "",
                visible,
                colors,
            )
            relation["visible_part"] = visible
        relations.append(relation)
    return _dedupe_relation_specs(relations)


def _canonical_colored_subject(
    value: str,
    subjects: Sequence[str],
    colors: Mapping[str, str],
) -> str:
    value = _trim_noun_phrase(_normalize_spacing(value).lower())
    if not value:
        return ""
    for subject in subjects:
        if _same_subject(value, subject):
            return subject
    value_without_color = _strip_leading_color(value)
    for object_name, color in colors.items():
        if _same_subject(value, object_name):
            return object_name
        if (
            value.startswith(f"{color} ")
            and _same_subject(value_without_color, object_name)
        ):
            return object_name
        if (
            object_name.startswith(f"{color} ")
            and _same_subject(value_without_color, _strip_leading_color(object_name))
        ):
            return object_name
    return value_without_color or value


def _strip_leading_color(value: str) -> str:
    parts = [part for part in _normalize_spacing(value).lower().split() if part]
    if len(parts) > 1 and parts[0] in COLOR_WORDS:
        parts = parts[1:]
    return " ".join(parts)


def _extract_chained_interaction_specs(
    subject: str,
    raw_target: str,
    colors: Mapping[str, str],
    action_alt: str,
) -> list[dict[str, str]]:
    pattern = re.compile(
        rf"\b(?P<action>{action_alt})\s+"
        r"(?:the\s+|a\s+|an\s+)?"
        r"(?P<object>(?:[a-z0-9-]+\s+){0,8}[a-z0-9-]+)\b"
    )
    relations: list[dict[str, str]] = []
    for match in pattern.finditer(raw_target):
        action = match.group("action").strip()
        raw_object = match.group("object")
        target = _trim_interaction_target(raw_object)
        target = _expand_part_of_colored_object_target(raw_object, target, colors)
        if _is_display_action(action):
            target = _normalize_display_target(raw_object, target)
        if not target:
            continue
        relations.append(
            {
                "phrase": f"{subject} {action} {target}",
                "subject": subject,
                "action": action,
                "object": target,
                "type": "interaction",
            }
        )
    return _dedupe_relation_specs(relations)


def _extract_negative_constraints(prompt: str) -> list[str]:
    constraints: list[str] = []
    for segment in re.split(r"[,;.]", prompt):
        text = _normalize_spacing(segment)
        lowered = text.lower()
        if not text:
            continue
        if re.search(r"\b(no|without|avoid|not)\b", lowered):
            constraints.append(text)
    return constraints


def _extract_style(prompt: str) -> str:
    phrases = _style_scene_phrases(prompt)
    return ", ".join(phrases[:2])


def _extract_background(prompt: str) -> str:
    for segment in re.split(r"[,;.]", prompt):
        text = _normalize_spacing(segment)
        lowered = text.lower()
        if re.search(r"\b(background|room|street|forest|garden|sky|table|scene)\b", lowered):
            return text
    return ""


def _extract_patterns(lowered: str, patterns: Iterable[str]) -> list[str]:
    result: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = match.group(0).strip()
            prefix = lowered[max(0, match.start() - 16) : match.start()]
            if re.search(r"\b(?:not|no|without)\s+$", prefix):
                continue
            if phrase not in result:
                result.append(phrase)
    return result


def _protected_phrases(
    prompt: str,
    colors: Mapping[str, str],
    actions: Sequence[str],
    relations: Sequence[str],
) -> list[str]:
    del actions, relations
    phrases: list[str] = []
    lowered = prompt.lower()
    for object_name, color in colors.items():
        phrase = _color_object_phrase(object_name, color)
        if phrase in lowered:
            phrases.append(phrase)
    for phrase in _style_scene_phrases(prompt):
        if phrase.lower() not in {item.lower() for item in phrases}:
            phrases.append(phrase)
    return phrases


def _style_scene_phrases(prompt: str) -> list[str]:
    phrases: list[str] = []
    for segment in re.split(r"[,;]", prompt):
        phrase = _normalize_spacing(segment)
        if not phrase:
            continue
        lowered = phrase.lower()
        term_count = sum(
            1
            for term in PROTECTED_STYLE_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered)
        )
        word_count = len(re.findall(r"[A-Za-z0-9-]+", phrase))
        if term_count >= 2 and word_count <= 8:
            phrases.append(phrase)
    return phrases


def _protect_color(prompt: str, *, object_name: str, color: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    result = prompt
    object_terms = _object_terms_for_color_guard(object_name, color)
    color_object = _color_object_phrase(object_name, color)
    if color_object.lower() not in result.lower():
        result = f"{color_object}, {result}"
        actions.append(f"restored user color: {color_object}")

    # Remove common generated contradictions such as a conflicting color
    # attached to a user-specified object.
    for other_color in COLOR_WORDS - {color}:
        for term in object_terms:
            pattern = re.compile(rf"\b{re.escape(other_color)}\s+{re.escape(term)}\b", re.I)
            if pattern.search(result):
                replacement = term if _starts_with_color(term, color) else f"{color} {term}"
                result = pattern.sub(replacement, result)
                actions.append(f"replaced conflicting color near {term}: {other_color}->{color}")
    for term in object_terms:
        if not _starts_with_color(term, color):
            continue
        duplicate_pattern = re.compile(rf"\b{re.escape(color)}\s+{re.escape(term)}\b", re.I)
        if duplicate_pattern.search(result):
            result = duplicate_pattern.sub(term, result)
            actions.append(f"collapsed duplicate color near {term}: {color} {term}->{term}")
    return result, actions


def _starts_with_color(object_name: str, color: str) -> bool:
    first = object_name.split()[0] if object_name.split() else ""
    return first.lower() == str(color).lower()


def _is_ambiguous_generic_color_binding(
    object_name: str,
    color: str,
    colors: Mapping[str, str],
) -> bool:
    """Avoid letting a generic class color overwrite specific colored instances.

    Example: a prompt with "black sign" and "blue sign" may also extract a
    generic "sign -> blue" binding. The generic binding must not rewrite
    "black sign" into "blue sign".
    """

    normalized = _normalize_spacing(object_name.lower())
    if not normalized:
        return False
    words = normalized.split()
    for other_name, other_color in colors.items():
        other = _normalize_spacing(str(other_name).lower())
        if other == normalized or str(other_color).lower() == str(color).lower():
            continue
        other_words = other.split()
        if len(other_words) <= len(words):
            continue
        if other_words[-len(words) :] == words:
            return True
    return False


def _conflicting_color_near_object(prompt: str, object_name: str, color: str) -> bool:
    for other_color in COLOR_WORDS - {color}:
        for term in _object_terms_for_color_guard(object_name, color):
            if re.search(rf"\b{re.escape(other_color)}\s+{re.escape(term)}\b", prompt):
                return True
    return False


def _add_color_constraint(colors: dict[str, str], object_name: str, color: str) -> None:
    """Preserve same-class objects with different colors as separate bindings."""

    existing = colors.get(object_name)
    if existing is None:
        colors[object_name] = color
        return
    if existing == color:
        return
    colors.pop(object_name, None)
    colors.setdefault(_colored_object_key(object_name, existing), existing)
    key = _colored_object_key(object_name, color)
    if key in colors and colors[key] != color:
        key = f"{key} {len(colors) + 1}"
    colors[key] = color


def _colored_object_key(object_name: str, color: str) -> str:
    first = object_name.split()[0] if object_name.split() else ""
    return object_name if first in COLOR_WORDS else f"{color} {object_name}".strip()


def _color_object_phrase(object_name: str, color: str) -> str:
    first = object_name.split()[0] if object_name.split() else ""
    return object_name if first == color else f"{color} {object_name}".strip()


def _object_terms_for_color_guard(object_name: str, color: str) -> list[str]:
    first = object_name.split()[0] if object_name.split() else ""
    if first == color:
        return [object_name]
    return _object_terms(object_name)


def _object_terms(object_name: str) -> list[str]:
    terms = [object_name]
    parts = [part for part in object_name.split() if part]
    if parts:
        terms.append(parts[-1])
    for term in list(terms):
        if term.endswith("ies") and len(term) > 3:
            terms.append(f"{term[:-3]}y")
        elif term.endswith("s") and len(term) > 3:
            terms.append(term[:-1])
        elif len(term) > 2:
            terms.append(f"{term}s")
    return list(dict.fromkeys(terms))


def _is_display_action(action: str) -> bool:
    return action.strip().lower() in DISPLAY_ACTIONS


def _normalize_display_target(raw_value: str, target: str) -> str:
    raw = _normalize_spacing(raw_value).lower()
    target = _normalize_spacing(target).lower()
    if re.search(r"\bexact\b.*\btext\b", raw) or target == "exact text":
        return "text"
    return target


def _is_text_symbol_target(value: str) -> bool:
    return bool(
        re.search(
            r"\b(text|symbol|logo|mark|letter|word|sign|star|moon|number)\b",
            str(value or "").lower(),
        )
    )


def _same_subject(left: str, right: str) -> bool:
    left = _normalize_spacing(left).lower()
    right = _normalize_spacing(right).lower()
    if not left or not right:
        return False
    if left == right:
        return True
    left_terms = set(_object_terms(left))
    right_terms = set(_object_terms(right))
    return bool(left_terms & right_terms)


def _mentions_subject(text: str, subject: str) -> bool:
    return _last_subject_mention(text, subject) >= 0


def _last_subject_mention(text: str, subject: str) -> int:
    best = -1
    for term in _object_terms(subject):
        matches = list(re.finditer(rf"\b{re.escape(term)}\b", text))
        if matches:
            best = max(best, matches[-1].start())
    return best


def _recent_sentence_prefix(text: str, index: int) -> str:
    prefix = text[:index]
    starts = [prefix.rfind(mark) for mark in ".;"]
    start = max(starts)
    if start >= 0:
        prefix = prefix[start + 1 :]
    return prefix[-160:]


def _nearest_subject_before(
    text: str,
    index: int,
    subjects: Sequence[str],
) -> str:
    best = ""
    best_index = -1
    prefix = text[:index]
    for subject in subjects:
        for term in _object_terms(subject):
            pos = prefix.rfind(term)
            if pos > best_index:
                best = subject
                best_index = pos
    return best


def _nearest_subject_after(
    text: str,
    index: int,
    subjects: Sequence[str],
) -> str:
    best = ""
    best_index: int | None = None
    suffix = text[index:]
    for subject in subjects:
        for term in _object_terms(subject):
            match = re.search(rf"\b{re.escape(term)}\b", suffix)
            if match and (
                best_index is None
                or match.start() < best_index
                or (match.start() == best_index and len(subject) > len(best))
            ):
                best = subject
                best_index = match.start()
    return best


def _subject_alternation(subjects: Sequence[str]) -> str:
    terms: list[str] = []
    for subject in subjects:
        terms.extend(_object_terms(subject))
    terms = sorted(set(terms), key=len, reverse=True)
    return "|".join(re.escape(term) for term in terms)


def _canonical_subject(raw_subject: str, subjects: Sequence[str]) -> str:
    raw = _normalize_spacing(raw_subject).lower()
    if not raw:
        return ""
    for subject in sorted(subjects, key=len, reverse=True):
        if _normalize_spacing(subject).lower() == raw:
            return subject
    trimmed = _trim_noun_phrase(raw)
    if not trimmed:
        return ""
    matches = [
        subject
        for subject in subjects
        if trimmed in _object_terms(subject) or subject in _object_terms(trimmed)
    ]
    if len(matches) == 1:
        return matches[0]
    return trimmed


def _trim_interaction_target(value: str) -> str:
    words = [word for word in value.split() if word]
    kept: list[str] = []
    for word in words:
        if word in SUBJECT_STOP_WORDS and kept:
            break
        if word in SUBJECT_STOP_WORDS or word in SUBJECT_DROP_WORDS:
            continue
        kept.append(word)
        if len(kept) >= 4:
            break
    cleaned = " ".join(kept)
    return "" if cleaned in INVALID_SUBJECTS else cleaned


def _expand_part_of_colored_object_target(
    raw_value: str,
    target: str,
    colors: Mapping[str, str],
) -> str:
    """Turn "part of a colored object" into a generic part target."""

    if not target:
        return target
    raw = _normalize_spacing(raw_value).lower()
    target_words = target.split()
    if not raw or not target_words:
        return target
    for object_name, color in sorted(colors.items(), key=lambda item: len(item[0]), reverse=True):
        if object_name == target or object_name in target.split():
            continue
        object_pattern = re.escape(object_name)
        color_pattern = re.escape(color)
        if re.search(
            rf"\bof\s+(?:the\s+|a\s+|an\s+)?(?:{color_pattern}\s+)?{object_pattern}\b",
            raw,
        ):
            return f"{object_name} {target}".strip()
    return target


def _is_part_of_known_object(
    target: str,
    colors: Mapping[str, str],
    subjects: Sequence[str],
) -> bool:
    target = _normalize_spacing(target).lower()
    target_words = target.split()
    if len(target_words) < 2:
        return False
    for object_name in [*colors.keys(), *subjects]:
        object_name = _normalize_spacing(object_name).lower()
        if not object_name or object_name == target:
            continue
        if target.startswith(f"{object_name} "):
            return True
    return False


def _remove_covered_part_subjects(
    subjects: Sequence[str],
    colors: Mapping[str, str],
    interaction_relations: Sequence[Mapping[str, str]],
) -> list[str]:
    covered_heads = {
        relation["object"].split()[-1]
        for relation in interaction_relations
        if relation.get("object")
        and _is_part_of_known_object(relation["object"], colors, subjects)
        and relation["object"].split()
    }
    result: list[str] = []
    for subject in subjects:
        if subject in covered_heads and subject not in colors:
            continue
        if _is_covered_generic_subject(subject, colors):
            continue
        if subject not in result:
            result.append(subject)
    return result


def _remove_occlusion_hidden_part_subjects(
    subjects: Sequence[str],
    interaction_relations: Sequence[Mapping[str, str]],
) -> list[str]:
    hidden_parts = {
        _normalize_spacing(relation.get("hidden_part", "")).lower()
        for relation in interaction_relations
        if relation.get("type") == "occlusion"
    }
    hidden_parts.discard("")
    if not hidden_parts:
        return list(subjects)
    result: list[str] = []
    for subject in subjects:
        if _normalize_spacing(subject).lower() in hidden_parts:
            continue
        if subject not in result:
            result.append(subject)
    return result


def _is_covered_generic_subject(
    subject: str,
    colors: Mapping[str, str],
) -> bool:
    subject = _normalize_spacing(subject).lower()
    if not subject or subject in colors:
        return False
    for object_name in colors:
        parts = object_name.split()
        if len(parts) > 1 and parts[-1] == subject:
            return True
    return False


def _dedupe_relation_specs(
    relations: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for relation in relations:
        item = {str(key): str(value) for key, value in relation.items()}
        key = (
            item.get("subject", ""),
            item.get("action", ""),
            item.get("phrase", ""),
            item.get("object", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _trim_noun_phrase(value: str) -> str:
    words = [word for word in value.split() if word]
    if not words:
        return ""
    kept: list[str] = []
    for word in words:
        if word.isdigit():
            continue
        if word in SUBJECT_STOP_WORDS and kept:
            break
        if word in SUBJECT_STOP_WORDS or word in SUBJECT_DROP_WORDS:
            continue
        kept.append(word)
    while len(kept) > 1 and kept[-1] in SUBJECT_TRAILING_ACTION_WORDS:
        kept.pop()
    cleaned = " ".join(kept[-3:])
    return "" if cleaned in INVALID_SUBJECTS else cleaned


def _remove_negative_clause_subjects(
    subjects: Sequence[str],
    prompt: str,
) -> list[str]:
    lowered = _normalize_spacing(prompt).lower()
    result: list[str] = []
    for subject in subjects:
        subject_norm = _normalize_spacing(subject).lower()
        if not subject_norm:
            continue
        if _is_pseudo_subject_fragment(subject_norm):
            continue
        if re.search(rf"\b{re.escape(subject_norm)}\s+must\s+not\b", lowered):
            continue
        if re.search(rf"\b{re.escape(subject_norm)}\s+(?:should|must)\s+not\b", lowered):
            continue
        if subject not in result:
            result.append(subject)
    return result


def _is_pseudo_subject_fragment(value: str) -> bool:
    value = _normalize_spacing(value).lower()
    if not value or value in INVALID_SUBJECTS:
        return True
    words = value.split()
    if not words:
        return True
    if words[0] in SUBJECT_STOP_WORDS:
        return True
    if words[-1] in {"has", "have", "reads", "read", "labeled", "labelled"}:
        return True
    if re.search(r"\b(?:has|have|reads?|label(?:ed|led))\b", value):
        return True
    if value in {
        "lower",
        "upper",
        "left",
        "right",
        "lower half",
        "upper half",
        "left half",
        "right half",
        "bottom half",
        "top half",
    }:
        return True
    return False


def _split_segments(prompt: str) -> list[str]:
    return [
        segment.strip(" ,;")
        for segment in re.split(r"[;,]", prompt)
        if segment.strip(" ,;")
    ]


def _remove_meta_constraint_segments(prompt: str) -> tuple[str, list[str]]:
    segments = _split_segments(prompt)
    kept: list[str] = []
    removed: list[str] = []
    for segment in segments:
        if _is_meta_constraint_segment(segment):
            removed.append(segment)
        else:
            kept.append(segment)
    if not removed:
        return prompt, []
    return _join_segments(kept) if kept else prompt, removed


def _is_meta_constraint_segment(segment: str) -> bool:
    lowered = segment.lower()
    patterns = (
        r"\bno\s+color\s+specification\b",
        r"\bno\s+specified\s+color\b",
        r"\bno\s+mention\s+of\b",
        r"\bdo\s+not\s+(?:mention|specify|include)\b",
        r"\bavoid\s+(?:mentioning|specifying|adding)\b",
        r"\bwithout\s+(?:specifying|mentioning|adding)\b",
        r"\bbeyond\s+the\s+user\b",
        r"\bbeyond\s+the\s+original\b",
        r"\bnot\s+part\s+of\s+the\s+original\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _partition_segments(
    segments: Sequence[str],
    constraints: PromptConstraints,
) -> tuple[list[str], list[str]]:
    protected: list[str] = []
    optional: list[str] = []
    protected_needles = _protected_needles(constraints)
    for segment in segments:
        lowered = segment.lower()
        if any(needle and needle.lower() in lowered for needle in protected_needles):
            protected.append(segment)
        elif any(term in lowered for term in LOW_PRIORITY_TERMS):
            optional.append(segment)
        else:
            optional.append(segment)
    protected = _prioritize_protected_segments(protected, constraints)
    return protected or [segments[0]], optional if protected else list(segments[1:])


def _prioritize_protected_segments(
    segments: Sequence[str],
    constraints: PromptConstraints,
) -> list[str]:
    indexed = list(enumerate(segments))
    indexed.sort(key=lambda item: (_protected_segment_priority(item[1], constraints), item[0]))
    return [segment for _, segment in indexed]


def _protected_segment_priority(segment: str, constraints: PromptConstraints) -> int:
    lowered = segment.lower()
    if any(
        phrase.lower() in lowered
        and sum(
            1
            for term in PROTECTED_STYLE_TERMS
            if re.search(rf"\b{re.escape(term)}\b", phrase.lower())
        )
        >= 2
        for phrase in constraints.protected_phrases
    ):
        return 0
    if any(action and action.lower() in lowered for action in constraints.actions):
        return 1
    if any(relation and relation.lower() in lowered for relation in constraints.relations):
        return 1
    if any(phrase and phrase.lower() in lowered for phrase in constraints.protected_phrases):
        return 2
    return 3


def _protected_needles(constraints: PromptConstraints) -> list[str]:
    needles: list[str] = []
    for value in [
        *constraints.protected_phrases,
        *constraints.subjects,
        *constraints.actions,
        *constraints.relations,
    ]:
        text = str(value or "").strip().lower()
        if not text:
            continue
        needles.append(text)
        for term in _object_terms(text):
            needles.append(term)
    return list(dict.fromkeys(needles))


def _join_segments(segments: Sequence[str]) -> str:
    return _normalize_spacing(", ".join(segment for segment in segments if segment))


def _normalize_spacing(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,;.])", r"\1", value)
    value = re.sub(r"([,;])\s*", r"\1 ", value)
    return value.strip(" ,;")


def _ensure_constraints(
    value: PromptConstraints | Mapping[str, Any] | str,
) -> PromptConstraints:
    if isinstance(value, PromptConstraints):
        return value
    if isinstance(value, str):
        return extract_constraints(value)
    if isinstance(value, Mapping):
        raw_intent = value.get("intent_spec")
        intent = (
            IntentSpec.from_dict(raw_intent)
            if isinstance(raw_intent, Mapping)
            else None
        )
        return PromptConstraints(
            original_prompt=str(value.get("original_prompt", "")),
            colors=dict(value.get("colors", {})),
            subjects=[str(item) for item in value.get("subjects", [])],
            actions=[str(item) for item in value.get("actions", [])],
            relations=[str(item) for item in value.get("relations", [])],
            protected_phrases=[
                str(item) for item in value.get("protected_phrases", [])
            ],
            intent_spec=intent,
        )
    raise TypeError("constraints must be PromptConstraints, mapping, or string")


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    value = value.strip()
    if not value:
        raise ValueError("prompt must not be empty")
    return value
