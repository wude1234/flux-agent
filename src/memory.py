"""Simple inspectable memory store for mock agent runs."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


class MemoryStore:
    """Small JSON-serializable memory store with deterministic substring search."""

    _ID_PATTERN = re.compile(r"^mem_(\d+)$")

    def __init__(self, records: Iterable[Mapping[str, Any]] | None = None) -> None:
        self._records: list[dict[str, Any]] = []
        self._next_index = 1
        for record in records or []:
            self._append_existing(record)

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError("memory record must be a mapping")
        item = deepcopy(dict(record))
        if "id" not in item:
            item["id"] = self._next_id()
        else:
            self._observe_id(item["id"])
        self._records.append(item)
        return deepcopy(item)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return deepcopy(list(reversed(self._records[-limit:])))

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(self._records):
            haystack = json.dumps(record, ensure_ascii=False, sort_keys=True).lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, index, record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [deepcopy(record) for _, _, record in scored[:limit]]

    def to_list(self) -> list[dict[str, Any]]:
        return deepcopy(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._next_index = 1

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "MemoryStore":
        input_path = Path(path)
        records = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("memory JSON must contain a list")
        return cls(records)

    def _append_existing(self, record: Mapping[str, Any]) -> None:
        item = deepcopy(dict(record))
        if "id" not in item:
            item["id"] = self._next_id()
        else:
            self._observe_id(item["id"])
        self._records.append(item)

    def _next_id(self) -> str:
        value = f"mem_{self._next_index:06d}"
        self._next_index += 1
        return value

    def _observe_id(self, value: object) -> None:
        if not isinstance(value, str):
            return
        match = self._ID_PATTERN.match(value)
        if not match:
            return
        self._next_index = max(self._next_index, int(match.group(1)) + 1)

