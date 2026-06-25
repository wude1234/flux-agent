from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import MemoryStore


def test_memory_append_copies_records_and_assigns_ids() -> None:
    store = MemoryStore()
    record = {"prompt": "red car on a rainy street", "feedback": "accepted"}

    stored = store.append(record)
    record["prompt"] = "mutated after append"

    assert stored["id"] == "mem_000001"
    assert store.to_list()[0]["prompt"] == "red car on a rainy street"


def test_memory_search_returns_relevant_records() -> None:
    store = MemoryStore()
    store.append({"prompt": "blue dragon in snow", "feedback": "wrong count"})
    store.append({"prompt": "red car on a rainy street", "feedback": "style ok"})
    store.append({"prompt": "red umbrella on a dry beach", "feedback": "background mismatch"})

    results = store.search("rainy car", limit=2)
    recent = store.search("", limit=1)

    assert [item["prompt"] for item in results] == ["red car on a rainy street"]
    assert recent[0]["prompt"] == "red umbrella on a dry beach"


def test_memory_json_roundtrip_preserves_records_and_next_id(tmp_path: Path) -> None:
    store = MemoryStore()
    store.append({"prompt": "first"})
    store.append({"prompt": "second"})
    path = tmp_path / "memory.json"

    store.save_json(path)
    loaded = MemoryStore.load_json(path)
    next_record = loaded.append({"prompt": "third"})

    assert loaded.to_list()[:2] == store.to_list()
    assert next_record["id"] == "mem_000003"

