"""M0 mock entrypoint for the fused multimodal T2I agent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .image_generator import MockImageGenerator
from .memory import MemoryStore
from .state import AgentConfig, AgentState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


def run_mock_agent(
    prompt: str,
    *,
    n_images: int = 1,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    """Run one mock generation round and write a structured JSON log."""

    config = AgentConfig(n_images=n_images, max_rounds=1)
    state = AgentState.from_config(prompt, config)
    memory = MemoryStore()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / run_id
    image_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=False)

    events: list[dict[str, Any]] = [
        {
            "type": "state_initialized",
            "round": state.round_index,
            "user_prompt": state.user_prompt,
        }
    ]

    candidate = state.add_candidate(
        {
            "prompt": state.user_prompt,
            "strategy": "mock_passthrough",
            "source": "m0",
            "reason": "M0 validates state flow before real model adapters.",
        }
    )
    state.select_candidate(0)
    events.append({"type": "candidate_selected", "round": 0, "candidate": candidate})

    generator = MockImageGenerator(
        placeholder_dir=image_dir,
        create_placeholders=True,
    )
    image_paths = generator.generate(state.active_prompt, n=config.n_images)
    state.add_images(image_paths)
    events.append(
        {
            "type": "mock_generation_completed",
            "round": 0,
            "prompt": state.active_prompt,
            "image_paths": image_paths,
        }
    )

    memory_record = memory.append(
        {
            "round": state.round_index,
            "user_prompt": state.user_prompt,
            "selected_prompt": state.active_prompt,
            "image_paths": image_paths,
            "mode": "mock",
        }
    )
    state.memory = memory.to_list()
    events.append({"type": "memory_appended", "round": 0, "record": memory_record})

    log = {
        "run_id": run_id,
        "mode": "mock",
        "config": config.to_dict(),
        "state": state.to_dict(),
        "events": events,
    }
    log_path = run_dir / "run.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "log": log,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M0 mock multimodal T2I agent")
    parser.add_argument("--prompt", required=True, help="User prompt to pass through M0")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock clients and mock image generation. Required in M0.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=1,
        help="Number of mock placeholder image paths to create.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory where the structured mock run log will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.mock:
        parser.error("M0 supports mock runs only; pass --mock.")

    result = run_mock_agent(
        args.prompt,
        n_images=args.n_images,
        runs_dir=args.runs_dir,
    )
    print(json.dumps({"run_dir": result["run_dir"], "log_path": result["log_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

