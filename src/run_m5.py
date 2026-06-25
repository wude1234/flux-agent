"""Command-line runner for the M5 layout planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clients import MockLLMClient, client_from_env
from .layout_planner import (
    LayoutPlanner,
    build_mock_layout_response,
    layout_to_enriched_prompt,
    layout_to_prompt_package,
)
from .logging_utils import DEFAULT_RUNS_DIR, create_run_dir, write_json
from .run_m4 import DEFAULT_DASHSCOPE_BASE_URL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M5 layout planner")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--canvas-width", type=int, default=1024)
    parser.add_argument("--canvas-height", type=int, default=1024)
    parser.add_argument("--llm", choices=["mock", "api"], default="mock")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-base-url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--llm-model", default="qwen-plus")
    parser.add_argument(
        "--strict-background",
        action="store_true",
        help="Fail if the background repeats foreground object terms.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canvas_size = (args.canvas_width, args.canvas_height)
    llm = _build_llm(args, canvas_size)
    planner = LayoutPlanner(llm, strict_background=args.strict_background)
    layout = planner.plan(args.prompt, canvas_size=canvas_size)
    package = layout_to_prompt_package(layout, user_prompt=args.prompt)
    enriched_prompt = layout_to_enriched_prompt(args.prompt, package)

    run_dir = create_run_dir(args.runs_dir, run_id=args.run_id)
    payload = {
        "run_id": run_dir.name,
        "mode": args.llm,
        "user_prompt": args.prompt,
        "layout": _strip_runtime_fields(layout),
        "prompt_package": package,
        "enriched_prompt": enriched_prompt,
    }
    write_json(run_dir / "layout.json", payload)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "layout": str(run_dir / "layout.json"),
                "objects": len(package["objects"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _build_llm(args: argparse.Namespace, canvas_size: tuple[int, int]):
    if args.llm == "api":
        return client_from_env(
            kind="llm",
            model=args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
        )
    return MockLLMClient(
        responses=[build_mock_layout_response(args.prompt, canvas_size=canvas_size)]
    )


def _strip_runtime_fields(layout: dict) -> dict:
    return {
        key: value
        for key, value in layout.items()
        if key not in {"request", "raw_response"}
    }


if __name__ == "__main__":
    raise SystemExit(main())
