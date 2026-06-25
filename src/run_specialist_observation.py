"""CLI for one structured VLM observation plus local specialist analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .clients import MockVLMClient, VLMClient, client_from_env
from .specialist_agents import analyze_specialist_observation, run_specialist_observation


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one VLM observation and local specialist-agent analysis."
    )
    parser.add_argument("--prompt", required=True, help="Original user prompt")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument(
        "--generated-prompt",
        default="",
        help="Prompt actually sent to the generator, if different.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--input-report",
        type=Path,
        default=None,
        help="Reuse an existing specialist report raw_response/observation without another VLM call.",
    )
    parser.add_argument("--vlm", choices=["mock", "api"], default="mock")
    parser.add_argument(
        "--mock-response",
        default="",
        help="JSON response for mock VLM tests.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-base-url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_report is not None:
        previous = json.loads(args.input_report.read_text(encoding="utf-8"))
        result = analyze_specialist_observation(
            user_prompt=args.prompt,
            image_path=str(args.image_path),
            generated_prompt=args.generated_prompt,
            raw_response=str(previous.get("raw_response") or ""),
            observation=(
                previous.get("observation")
                if isinstance(previous.get("observation"), dict)
                else None
            ),
            request=str(previous.get("request") or ""),
            api_call_count=0,
        )
    else:
        vlm = _build_vlm(args)
        result = run_specialist_observation(
            vlm=vlm,
            user_prompt=args.prompt,
            image_path=str(args.image_path),
            generated_prompt=args.generated_prompt,
        )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _build_vlm(args: argparse.Namespace) -> VLMClient:
    if args.vlm == "mock":
        responses = [args.mock_response] if args.mock_response else ()
        return MockVLMClient(responses=responses)
    return client_from_env(
        kind="vlm",
        model=args.vlm_model,
        api_key_env=args.api_key_env,
        base_url=args.api_base_url,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    raise SystemExit(main())
