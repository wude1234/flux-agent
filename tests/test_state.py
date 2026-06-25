from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state import AgentConfig, AgentState, CreativityLevel


def test_state_defaults_are_independent() -> None:
    first = AgentState("a red car")
    second = AgentState("a blue boat")

    first.add_images(["mock://image/0000"])
    first.add_feedback({"source": "mock", "message": "ok"})

    assert second.image_paths == []
    assert second.feedback == []


def test_state_candidate_selection_and_roundtrip() -> None:
    state = AgentState("a red car on a rainy street", creativity_level="high", seed=42)
    candidate = state.add_candidate(
        {
            "prompt": "cinematic red car on a rainy city street",
            "strategy": "mock_passthrough",
        }
    )

    selected = state.select_candidate(0)
    state.add_images(["mock://image/0000"])
    state.add_feedback({"source": "mock", "message": "no visual critique in M0"})
    state.remember({"prompt": selected, "image_count": 1})
    state.advance_round()

    restored = AgentState.from_dict(state.to_dict())

    assert candidate["prompt"] == "cinematic red car on a rainy city street"
    assert selected == "cinematic red car on a rainy city street"
    assert restored.active_prompt == selected
    assert restored.creativity_level is CreativityLevel.HIGH
    assert restored.seed == 42
    assert restored.round_index == 1
    assert restored.image_paths == ["mock://image/0000"]


def test_state_from_config_uses_t2i_copilot_subset() -> None:
    config = AgentConfig(
        human_in_loop=False,
        creativity_level=CreativityLevel.HIGH,
        n_images=2,
        max_rounds=1,
        seed=7,
    )

    state = AgentState.from_config("a small cabin in snow", config)

    assert state.human_in_loop is False
    assert state.creativity_level is CreativityLevel.HIGH
    assert state.seed == 7
    assert config.to_dict()["creativity_level"] == "high"


def test_state_rejects_invalid_updates() -> None:
    with pytest.raises(ValueError):
        AgentState("   ")

    state = AgentState("a red car")
    with pytest.raises(KeyError):
        state.apply_update({"unknown": "value"})

