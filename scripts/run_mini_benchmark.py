from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binding_strategy import build_negative_prompt
from src.clients import MockLLMClient, MockVLMClient, client_from_env
from src.evaluators import VLMJudgeEvaluator
from src.image_generator import MockImageGenerator, _merge_flux_negative_prompt
from src.logging_utils import atomic_write_text
from src.orchestrator import OrchestratorAgent
from src.prompt_constraints import extract_constraints
from src.state import AgentConfig

DEFAULT_PYTHON = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/project/.conda-m0/bin/python"
)
DEFAULT_POWERPAINT_PYTHON = "/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_FLUX_PYTHON = "/mnt/ssd1/conda/envs/flux-dev/bin/python"
DEFAULT_FLUX_HF_HOME = "/mnt/ssd3/zrr/hf_cache"
DEFAULT_MGRAG_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small FLUX-agent hard-prompt benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "hard_prompts_mini.json",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--one-per-category", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Run all cases whose category matches this value. Can be repeated.",
    )
    parser.add_argument("--generator", choices=["flux", "fusion", "sdxl"], default="flux")
    parser.add_argument("--llm", choices=["mock", "api"], default="api")
    parser.add_argument("--vlm", choices=["mock", "api"], default="api")
    parser.add_argument("--llm-model", default="qwen-plus")
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-base-url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help=(
            "Forward run_m4 --decision-only: generate/evaluate/plan once and "
            "skip typed action candidates, SAM, PowerPaint, and local repairs."
        ),
    )
    parser.add_argument("--n-images", type=int, default=1)
    parser.add_argument("--auto-n-images-for-hard-prompts", action="store_true")
    parser.add_argument("--hard-prompt-n-images", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=7100)
    parser.add_argument(
        "--seed-policy",
        choices=["case-id", "sequential"],
        default="case-id",
        help=(
            "How to assign seeds. case-id keeps the same case stable across "
            "different subsets; sequential keeps the older seed_base+index behavior."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.85)
    parser.add_argument("--flux-timeout-seconds", type=int, default=900)
    parser.add_argument("--flux-python", default=DEFAULT_FLUX_PYTHON)
    parser.add_argument("--flux-repo", default="/home/zrr/flux")
    parser.add_argument("--flux-hf-home", default=DEFAULT_FLUX_HF_HOME)
    parser.add_argument("--flux-model-path", default=None)
    parser.add_argument("--flux-ae-path", default=None)
    parser.add_argument("--flux-attn-mode", choices=["mgrag", "baseline"], default="mgrag")
    parser.add_argument("--mgrag-delta-scale", type=float, default=1.3)
    parser.add_argument("--mgrag-bias-scale", type=float, default=1.0)
    parser.add_argument("--mgrag-intervene-steps", type=int, default=20)
    parser.add_argument("--mgrag-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--mgrag-image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--mgrag-model-id", default=None)
    parser.add_argument("--mgrag-online", action="store_true")
    parser.add_argument("--subprocess-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--retry-on-infra-failure",
        type=int,
        default=0,
        help=(
            "Retry a case this many times when the failure is infrastructure-like "
            "(process_killed, gpu_oom, or subprocess_timeout)."
        ),
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
        help="Delay between infrastructure retries.",
    )
    parser.add_argument("--use-layout-planner", action="store_true")
    parser.add_argument("--layout-llm", choices=["mock", "api"], default=None)
    parser.add_argument("--layout-llm-model", default=None)
    parser.add_argument("--layout-canvas-width", type=int, default=None)
    parser.add_argument("--layout-canvas-height", type=int, default=None)
    parser.add_argument("--enable-local-repair", action="store_true")
    parser.add_argument("--enable-vlm-target-locator", action="store_true")
    parser.add_argument("--local-editor", choices=["recolor"], default=None)
    parser.add_argument("--mask-refiner", choices=["bbox", "mock", "sam-v1", "none"], default=None)
    parser.add_argument("--sam-checkpoint-path", default=None)
    parser.add_argument("--sam-model-type", default=None)
    parser.add_argument("--sam-device", default=None)
    parser.add_argument("--enable-relation-repair", action="store_true")
    parser.add_argument("--enable-object-insertion-repair", action="store_true")
    parser.add_argument("--enable-efficient-repair-agent", action="store_true")
    parser.add_argument(
        "--auto-efficient-repair-for-categories",
        action="store_true",
        help=(
            "Automatically enable the efficient editing agent for benchmark "
            "categories where local edits are expected to be useful."
        ),
    )
    parser.add_argument(
        "--efficient-repair-category",
        action="append",
        default=[],
        help=(
            "Category that should receive efficient editing flags when "
            "--auto-efficient-repair-for-categories is set. Can be repeated."
        ),
    )
    parser.add_argument("--enable-editing-mask-agent", action="store_true")
    parser.add_argument("--editing-mask-mode", choices=["auto", "grounded-sam2", "bbox"], default=None)
    parser.add_argument("--editing-mask-text", default=None)
    parser.add_argument("--editing-mask-dilation-kernel-size", type=int, default=None)
    parser.add_argument("--editing-min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--allow-editing-bbox-fallback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grounded-sam2-python", default=None)
    parser.add_argument("--grounded-sam2-dir", default=None)
    parser.add_argument("--grounded-sam2-timeout-seconds", type=int, default=None)
    parser.add_argument("--grounded-sam2-cuda-visible-devices", default=None)
    parser.add_argument("--grounded-sam2-hf-home", default=None)
    parser.add_argument(
        "--relation-editor",
        choices=["mock", "sd15-inpaint", "sd15-subprocess-inpaint", "powerpaint-subprocess"],
        default=None,
    )
    parser.add_argument("--relation-candidates", type=int, default=None)
    parser.add_argument("--relation-pass-threshold", type=float, default=None)
    parser.add_argument("--relation-inpaint-model-path", default=None)
    parser.add_argument("--relation-inpaint-python", default=None)
    parser.add_argument("--relation-inpaint-timeout-seconds", type=int, default=None)
    parser.add_argument("--relation-inpaint-cuda-visible-devices", default=None)
    parser.add_argument("--powerpaint-python", default=DEFAULT_POWERPAINT_PYTHON)
    parser.add_argument("--powerpaint-dir", default=None)
    parser.add_argument("--powerpaint-checkpoint-dir", default=None)
    parser.add_argument("--powerpaint-timeout-seconds", type=int, default=None)
    parser.add_argument("--relation-steps", type=int, default=None)
    parser.add_argument("--relation-guidance-scale", type=float, default=None)
    parser.add_argument("--relation-strength", type=float, default=None)
    parser.add_argument("--enable-typed-action-backend", action="store_true")
    parser.add_argument("--disable-typed-action-backend", action="store_true")
    parser.add_argument("--typed-action-candidates", type=int, default=None)
    parser.add_argument("--typed-action-max-candidates", type=int, default=None)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument(
        "--batch-decision",
        action="store_true",
        help=(
            "Decision-only accelerator: generate all selected FLUX/M-GRAG "
            "round-0 images in one subprocess, then run per-case VLM "
            "evaluation and repair planning with pregenerated images. This "
            "avoids one FLUX cold start per case."
        ),
    )
    parser.add_argument(
        "--batch-decision-chunk-size",
        type=int,
        default=5,
        help=(
            "Number of prompts per FLUX subprocess in --batch-decision. "
            "Use a small micro-batch to reduce cold starts without keeping "
            "CPU-offloaded FLUX alive for the whole benchmark."
        ),
    )
    parser.add_argument(
        "--batch-generation-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Timeout per FLUX micro-batch. Defaults to a per-case scaled "
            "timeout based on --flux-timeout-seconds."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    cases = _select_cases(benchmark["cases"], args)
    run_prefix = args.run_prefix or datetime.now().strftime("mini-flux-%Y%m%d-%H%M%S")
    runs_dir = args.runs_dir or PROJECT_ROOT / "runs_mini_benchmark" / run_prefix
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_decision:
        args.decision_only = True
        results = _run_batch_decision(args, benchmark, cases, runs_dir, run_prefix)
        _write_summary(runs_dir, benchmark, args, results)
        print(json.dumps({"runs_dir": str(runs_dir), "cases": len(results)}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        run_id = f"{run_prefix}-{case['id']}"
        seed = _seed_for_case(args, case, index)
        command = _build_command(args, case, runs_dir, run_id, seed)
        print(f"[{index + 1}/{len(cases)}] {case['id']} {case['category']}", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "prompt": case["prompt"],
                    "focus": list(case.get("focus", []) or []),
                    "seed": seed,
                    "dry_run_command": command,
                }
            )
            continue
        results.append(
            _run_case_with_retries(
                case,
                command,
                seed,
                timeout=args.subprocess_timeout_seconds,
                max_retries=args.retry_on_infra_failure,
                retry_delay_seconds=args.retry_delay_seconds,
            )
        )
        _write_summary(runs_dir, benchmark, args, results)

    _write_summary(runs_dir, benchmark, args, results)
    print(json.dumps({"runs_dir": str(runs_dir), "cases": len(results)}, indent=2))
    return 0


def _select_cases(cases: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [dict(case) for case in cases]
    if args.category:
        wanted_categories = {str(item) for item in args.category}
        selected = [
            case for case in selected if str(case.get("category")) in wanted_categories
        ]
    if args.case_id:
        wanted = set(args.case_id)
        selected = [case for case in selected if case["id"] in wanted]
    if args.one_per_category:
        by_category: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for case in selected:
            by_category.setdefault(str(case["category"]), case)
        selected = list(by_category.values())
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    if not selected:
        raise SystemExit("no benchmark cases selected")
    return selected


def _seed_for_case(
    args: argparse.Namespace,
    case: Mapping[str, Any],
    index: int,
) -> int:
    if "seed" in case and case["seed"] is not None:
        return int(case["seed"])
    if getattr(args, "seed_policy", "case-id") == "sequential":
        return int(args.seed_base) + int(index)
    case_id = str(case.get("id") or case.get("prompt") or index)
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 1_000_000
    return int(args.seed_base) + offset


def _run_batch_decision(
    args: argparse.Namespace,
    benchmark: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    runs_dir: Path,
    run_prefix: str,
) -> list[dict[str, Any]]:
    """Generate all round-0 FLUX images once, then run decision-only agents."""

    del benchmark
    if args.generator != "flux":
        raise SystemExit("--batch-decision currently supports --generator flux only")
    if args.flux_attn_mode != "mgrag":
        raise SystemExit("--batch-decision currently supports --flux-attn-mode mgrag only")

    seeds = [_seed_for_case(args, case, index) for index, case in enumerate(cases)]
    image_dir = runs_dir / "_batch_decision_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        _batch_flux_prompt(str(case["prompt"]), negative_prompt=None)
        for case in cases
    ]
    chunks = _batch_chunks(len(cases), max(1, int(args.batch_decision_chunk_size)))
    commands = [
        _build_batch_mgrag_command(
            args,
            prompts[start:end],
            seeds[start:end],
            image_dir,
            start_index=start,
        )
        for start, end in chunks
    ]
    atomic_write_text(
        runs_dir / "batch_decision_generation_command.json",
        json.dumps(
            {
                "mode": "batch_decision",
                "case_count": len(cases),
                "chunk_size": max(1, int(args.batch_decision_chunk_size)),
                "commands": commands,
                "seeds": seeds,
                "image_dir": str(image_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.dry_run:
        return [
            {
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "focus": list(case.get("focus", []) or []),
                "seed": seed,
                "dry_run_command": commands[0] if commands else [],
                "dry_run_commands": commands,
            }
            for case, seed in zip(cases, seeds)
        ]

    failed_records: dict[int, dict[str, Any]] = {}
    env = _batch_flux_env(args)
    for chunk_index, ((start, end), command) in enumerate(zip(chunks, commands)):
        timeout = _batch_generation_timeout(args, end - start)
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            atomic_write_text(
                runs_dir / f"batch_decision_generation_chunk_{chunk_index:03d}_stdout.log",
                stdout,
                encoding="utf-8",
            )
            atomic_write_text(
                runs_dir / f"batch_decision_generation_chunk_{chunk_index:03d}_stderr.log",
                stderr,
                encoding="utf-8",
            )
            for local_index, case in enumerate(cases[start:end]):
                global_index = start + local_index
                failed_records[global_index] = {
                    "status": "subprocess_timeout",
                    "returncode": None,
                    "failure_category": "infrastructure",
                    "failure_reason": f"FLUX micro-batch timed out after {timeout}s",
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-4000:],
                }
            continue
        atomic_write_text(
            runs_dir / f"batch_decision_generation_chunk_{chunk_index:03d}_stdout.log",
            completed.stdout,
            encoding="utf-8",
        )
        atomic_write_text(
            runs_dir / f"batch_decision_generation_chunk_{chunk_index:03d}_stderr.log",
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            details = _classify_process_failure(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
            for local_index, case in enumerate(cases[start:end]):
                global_index = start + local_index
                failed_records[global_index] = {
                    "status": details.get("status", "batch_generation_failed"),
                    "returncode": completed.returncode,
                    "failure_category": details.get("failure_category", "infrastructure"),
                    "failure_reason": details.get("failure_reason"),
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-4000:],
                }

    results: list[dict[str, Any]] = []
    for index, (case, seed) in enumerate(zip(cases, seeds)):
        image_path = image_dir / f"img_{index:04d}.{args.mgrag_image_format}"
        run_id = f"{run_prefix}-{case['id']}"
        record = {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "focus": list(case.get("focus", []) or []),
            "seed": seed,
            "batch_decision": True,
            "pregenerated_image": str(image_path),
        }
        if index in failed_records:
            record.update(failed_records[index])
            results.append(record)
            continue
        if not image_path.exists():
            record.update(
                {
                    "status": "missing_pregenerated_image",
                    "failure_category": "infrastructure",
                    "failure_reason": f"Expected batch image does not exist: {image_path}",
                }
            )
            results.append(record)
            continue
        try:
            run_dir = _run_decision_agent_on_image(
                args,
                case,
                image_path,
                runs_dir,
                run_id,
                seed,
            )
        except Exception as exc:
            record.update(
                {
                    "status": "decision_agent_failed",
                    "failure_category": "agent",
                    "failure_reason": str(exc),
                }
            )
            results.append(record)
            continue
        record["run_dir"] = str(run_dir)
        record.update(_summarize_run_dir(run_dir))
        results.append(record)
    return results


def _build_batch_mgrag_command(
    args: argparse.Namespace,
    prompts: Sequence[str],
    seeds: Sequence[int],
    image_dir: Path,
    *,
    start_index: int = 0,
) -> list[str]:
    effective_intervene_steps = min(int(args.mgrag_intervene_steps), int(args.steps))
    command = [
        args.flux_python,
        str(PROJECT_ROOT / "infer_mgrag_flux.py"),
        "--output_dir",
        str(image_dir),
        "--output_prefix",
        "img",
        "--start_index",
        str(int(start_index)),
        "--image_format",
        args.mgrag_image_format,
        "--model_id",
        args.mgrag_model_id or DEFAULT_MGRAG_FLUX_MODEL_ID,
        "--dtype",
        args.mgrag_dtype,
        "--delta_list",
        str(args.mgrag_delta_scale),
        "--bias_list",
        str(args.mgrag_bias_scale),
        "--intervene_steps",
        str(effective_intervene_steps),
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(int(seeds[0]) if seeds else int(args.seed_base)),
        "--device",
        args.device or "cuda",
        "--local_files_only",
    ]
    for prompt, seed in zip(prompts, seeds):
        command.extend(["--prompt", prompt, "--prompt_seed", str(int(seed))])
    return command


def _batch_chunks(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(int(total), start + int(chunk_size)))
        for start in range(0, int(total), int(chunk_size))
    ]


def _batch_generation_timeout(args: argparse.Namespace, chunk_count: int) -> int:
    if args.batch_generation_timeout_seconds is not None:
        return int(args.batch_generation_timeout_seconds)
    per_case = int(args.flux_timeout_seconds or args.subprocess_timeout_seconds or 900)
    return max(int(args.subprocess_timeout_seconds), per_case * max(1, int(chunk_count)))


def _run_decision_agent_on_image(
    args: argparse.Namespace,
    case: Mapping[str, Any],
    image_path: Path,
    runs_dir: Path,
    run_id: str,
    seed: int,
) -> Path:
    prompt = str(case["prompt"])
    vlm = _decision_vlm(args)
    agent = OrchestratorAgent(
        llm=_decision_llm(args, prompt),
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[image_path]),
        config=AgentConfig(n_images=1, max_rounds=1, seed=seed, creativity_level="high"),
        runs_dir=runs_dir,
        mode="batch_decision",
        score_threshold=args.score_threshold,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=True,
        enable_evaluator=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_typed_action_backend=False,
        enable_local_repair=False,
        enable_vlm_target_locator=False,
        enable_relation_repair=False,
        enable_object_insertion_repair=False,
        enable_efficient_repair_agent=False,
        auto_negative_prompt=False,
    )
    result = agent.run(prompt, run_id=run_id)
    return Path(result.run_dir)


def _decision_llm(args: argparse.Namespace, prompt: str):
    if args.llm == "api":
        return client_from_env(
            kind="llm",
            model=args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
        )
    response = json.dumps({"prompts": [prompt]}, ensure_ascii=False)
    return MockLLMClient(responses=[response], default_response=response)


def _decision_vlm(args: argparse.Namespace):
    if args.vlm == "api":
        return client_from_env(
            kind="vlm",
            model=args.vlm_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
        )
    return MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.8}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["Mock VLM accepted the generated image."],
                    "revision_hint": "No mock revision needed.",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.9,
                    "checks": [],
                    "errors": [],
                    "revision_hint": "Mock constraints pass.",
                },
                ensure_ascii=False,
            ),
        ],
        default_response=json.dumps(
            {
                "score": 0.9,
                "errors": [],
                "strengths": ["Mock VLM accepted the generated image."],
                "revision_hint": "No mock revision needed.",
            },
            ensure_ascii=False,
        ),
    )


def _batch_flux_prompt(prompt: str, *, negative_prompt: str | None) -> str:
    constraints = extract_constraints(prompt)
    automatic_negative = build_negative_prompt(constraints)
    merged_negative = _merge_optional_prompts(automatic_negative, negative_prompt)
    return _merge_flux_negative_prompt(prompt, merged_negative)


def _merge_optional_prompts(*values: str | None) -> str | None:
    parts = [str(value).strip(" .") for value in values if str(value or "").strip()]
    return ". ".join(parts) if parts else None


def _batch_flux_env(args: argparse.Namespace) -> dict[str, str]:
    import os

    env = os.environ.copy()
    src_path = str(Path(args.flux_repo) / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.flux_hf_home:
        hf_home = Path(args.flux_hf_home)
        env["HF_HOME"] = str(hf_home)
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def _build_command(
    args: argparse.Namespace,
    case: Mapping[str, Any],
    runs_dir: Path,
    run_id: str,
    seed: int,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "src.run_m4",
        "--prompt",
        str(case["prompt"]),
        "--runs-dir",
        str(runs_dir),
        "--run-id",
        run_id,
        "--generator",
        args.generator,
        "--llm",
        args.llm,
        "--vlm",
        args.vlm,
        "--api-key-env",
        args.api_key_env,
        "--llm-model",
        args.llm_model,
        "--vlm-model",
        args.vlm_model,
        "--disable-clarifier",
        "--enable-evaluator",
        "--evaluator-vlm",
        args.vlm,
        "--evaluator-vlm-model",
        args.vlm_model,
        "--max-rounds",
        str(args.max_rounds),
        "--n-images",
        str(args.n_images),
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--guidance-scale",
        str(args.guidance_scale),
        "--seed",
        str(seed),
        "--score-threshold",
        str(args.score_threshold),
    ]
    if args.decision_only:
        command.append("--decision-only")
    if args.device:
        command.extend(["--device", args.device])
    if args.auto_n_images_for_hard_prompts:
        command.append("--auto-n-images-for-hard-prompts")
        command.extend(["--hard-prompt-n-images", str(args.hard_prompt_n_images)])
    if args.generator in {"flux", "fusion"}:
        command.extend(
            [
                "--cuda-visible-devices",
                args.cuda_visible_devices,
                "--flux-python",
                args.flux_python,
                "--flux-repo",
                args.flux_repo,
                "--flux-timeout-seconds",
                str(args.flux_timeout_seconds),
                "--flux-attn-mode",
                args.flux_attn_mode,
                "--mgrag-delta-scale",
                str(args.mgrag_delta_scale),
                "--mgrag-bias-scale",
                str(args.mgrag_bias_scale),
                "--mgrag-intervene-steps",
                str(args.mgrag_intervene_steps),
                "--mgrag-dtype",
                args.mgrag_dtype,
                "--mgrag-image-format",
                args.mgrag_image_format,
            ]
        )
        if args.flux_hf_home:
            command.extend(["--flux-hf-home", args.flux_hf_home])
        if args.flux_model_path:
            command.extend(["--flux-model-path", args.flux_model_path])
        if args.flux_ae_path:
            command.extend(["--flux-ae-path", args.flux_ae_path])
        if args.mgrag_model_id:
            command.extend(["--mgrag-model-id", args.mgrag_model_id])
        if args.mgrag_online:
            command.append("--mgrag-online")
    if args.use_layout_planner:
        command.append("--use-layout-planner")
        if args.layout_llm:
            command.extend(["--layout-llm", args.layout_llm])
        if args.layout_llm_model:
            command.extend(["--layout-llm-model", args.layout_llm_model])
        if args.layout_canvas_width is not None:
            command.extend(["--layout-canvas-width", str(args.layout_canvas_width)])
        if args.layout_canvas_height is not None:
            command.extend(["--layout-canvas-height", str(args.layout_canvas_height)])
    _append_repair_args_for_case(
        command,
        args,
        case_category=str(case.get("category") or ""),
    )
    return command


def _append_repair_args(command: list[str], args: argparse.Namespace) -> None:
    _append_repair_args_for_case(command, args, case_category=None)


DEFAULT_EFFICIENT_REPAIR_CATEGORIES = {
    "occlusion_visibility",
    "occlusion",
    "text_symbol",
    "text_symbol_layout",
    "negation_absence",
    "multi_compositional",
}


def _append_repair_args_for_case(
    command: list[str],
    args: argparse.Namespace,
    *,
    case_category: str | None,
) -> None:
    if getattr(args, "decision_only", False):
        return
    auto_efficient = _auto_enable_efficient_repair(args, case_category)
    enable_object_insertion_repair = bool(
        args.enable_object_insertion_repair or auto_efficient
    )
    enable_relation_repair = bool(args.enable_relation_repair or auto_efficient)
    enable_efficient_repair_agent = bool(args.enable_efficient_repair_agent or auto_efficient)
    # D2 fix: the bbox-based editing routes (existing_object_inpaint /
    # bbox_shape_inpaint / shape_overlay) are rejected by the efficient-repair
    # gate when the plan has no localized bbox. The locator is what fills that
    # bbox in, so it must follow whenever the efficient repair agent is on —
    # otherwise enabling the agent alone silently produces 0 edits (the
    # "efficient inpaint route requires an explicit localized bbox" skip).
    enable_vlm_target_locator = bool(
        args.enable_vlm_target_locator or auto_efficient or enable_efficient_repair_agent
    )
    enable_editing_mask_agent = bool(args.enable_editing_mask_agent or auto_efficient)
    relation_editor = args.relation_editor or (
        "powerpaint-subprocess" if auto_efficient else None
    )
    relation_candidates = args.relation_candidates
    if relation_candidates is None and auto_efficient:
        relation_candidates = 1
    editing_mask_mode = args.editing_mask_mode or ("auto" if auto_efficient else None)
    editing_mask_dilation_kernel_size = args.editing_mask_dilation_kernel_size
    if editing_mask_dilation_kernel_size is None and auto_efficient:
        editing_mask_dilation_kernel_size = 31
    allow_editing_bbox_fallback = args.allow_editing_bbox_fallback
    if allow_editing_bbox_fallback is None and auto_efficient:
        allow_editing_bbox_fallback = True
    relation_steps = args.relation_steps
    if relation_steps is None and auto_efficient:
        relation_steps = 20
    relation_guidance_scale = args.relation_guidance_scale
    if relation_guidance_scale is None and auto_efficient:
        relation_guidance_scale = 7.5
    relation_strength = args.relation_strength
    if relation_strength is None and auto_efficient:
        relation_strength = 1.0
    relation_inpaint_cuda_visible_devices = args.relation_inpaint_cuda_visible_devices
    if relation_inpaint_cuda_visible_devices is None and auto_efficient:
        relation_inpaint_cuda_visible_devices = args.cuda_visible_devices
    grounded_sam2_cuda_visible_devices = args.grounded_sam2_cuda_visible_devices
    if grounded_sam2_cuda_visible_devices is None and auto_efficient:
        grounded_sam2_cuda_visible_devices = args.cuda_visible_devices
    grounded_sam2_python = args.grounded_sam2_python or (
        "/mnt/ssd1/conda/envs/tweediemix/bin/python" if auto_efficient else None
    )
    grounded_sam2_hf_home = args.grounded_sam2_hf_home or (
        "/mnt/ssd1/powerpaint_envs/hf-cache" if auto_efficient else None
    )
    powerpaint_timeout_seconds = args.powerpaint_timeout_seconds
    if powerpaint_timeout_seconds is None and auto_efficient:
        powerpaint_timeout_seconds = 1800
    if args.enable_local_repair:
        command.append("--enable-local-repair")
    if enable_vlm_target_locator:
        command.append("--enable-vlm-target-locator")
    if args.local_editor:
        command.extend(["--local-editor", args.local_editor])
    if args.mask_refiner:
        command.extend(["--mask-refiner", args.mask_refiner])
    if args.sam_checkpoint_path:
        command.extend(["--sam-checkpoint-path", args.sam_checkpoint_path])
    if args.sam_model_type:
        command.extend(["--sam-model-type", args.sam_model_type])
    if args.sam_device:
        command.extend(["--sam-device", args.sam_device])
    if enable_relation_repair:
        command.append("--enable-relation-repair")
    if enable_object_insertion_repair:
        command.append("--enable-object-insertion-repair")
    if enable_efficient_repair_agent:
        command.append("--enable-efficient-repair-agent")
    if enable_editing_mask_agent:
        command.append("--enable-editing-mask-agent")
    if editing_mask_mode:
        command.extend(["--editing-mask-mode", editing_mask_mode])
    if args.editing_mask_text:
        command.extend(["--editing-mask-text", args.editing_mask_text])
    if editing_mask_dilation_kernel_size is not None:
        command.extend(
            [
                "--editing-mask-dilation-kernel-size",
                str(editing_mask_dilation_kernel_size),
            ]
        )
    if args.editing_min_mask_area_ratio is not None:
        command.extend(["--editing-min-mask-area-ratio", str(args.editing_min_mask_area_ratio)])
    if allow_editing_bbox_fallback is not None:
        command.append(
            "--allow-editing-bbox-fallback"
            if allow_editing_bbox_fallback
            else "--no-allow-editing-bbox-fallback"
        )
    if grounded_sam2_python:
        command.extend(["--grounded-sam2-python", grounded_sam2_python])
    if args.grounded_sam2_dir:
        command.extend(["--grounded-sam2-dir", args.grounded_sam2_dir])
    if args.grounded_sam2_timeout_seconds is not None:
        command.extend(
            ["--grounded-sam2-timeout-seconds", str(args.grounded_sam2_timeout_seconds)]
        )
    if grounded_sam2_cuda_visible_devices:
        command.extend(
            ["--grounded-sam2-cuda-visible-devices", grounded_sam2_cuda_visible_devices]
        )
    if grounded_sam2_hf_home:
        command.extend(["--grounded-sam2-hf-home", grounded_sam2_hf_home])
    if relation_editor:
        command.extend(["--relation-editor", relation_editor])
    if relation_candidates is not None:
        command.extend(["--relation-candidates", str(relation_candidates)])
    if args.relation_pass_threshold is not None:
        command.extend(["--relation-pass-threshold", str(args.relation_pass_threshold)])
    if args.relation_inpaint_model_path:
        command.extend(["--relation-inpaint-model-path", args.relation_inpaint_model_path])
    if args.relation_inpaint_python:
        command.extend(["--relation-inpaint-python", args.relation_inpaint_python])
    if args.relation_inpaint_timeout_seconds is not None:
        command.extend(
            [
                "--relation-inpaint-timeout-seconds",
                str(args.relation_inpaint_timeout_seconds),
            ]
        )
    if relation_inpaint_cuda_visible_devices:
        command.extend(
            [
                "--relation-inpaint-cuda-visible-devices",
                relation_inpaint_cuda_visible_devices,
            ]
        )
    if args.powerpaint_python:
        command.extend(["--powerpaint-python", args.powerpaint_python])
    if args.powerpaint_dir:
        command.extend(["--powerpaint-dir", args.powerpaint_dir])
    if args.powerpaint_checkpoint_dir:
        command.extend(["--powerpaint-checkpoint-dir", args.powerpaint_checkpoint_dir])
    if powerpaint_timeout_seconds is not None:
        command.extend(["--powerpaint-timeout-seconds", str(powerpaint_timeout_seconds)])
    if relation_steps is not None:
        command.extend(["--relation-steps", str(relation_steps)])
    if relation_guidance_scale is not None:
        command.extend(["--relation-guidance-scale", str(relation_guidance_scale)])
    if relation_strength is not None:
        command.extend(["--relation-strength", str(relation_strength)])
    if args.enable_typed_action_backend:
        command.append("--enable-typed-action-backend")
    if args.disable_typed_action_backend:
        command.append("--disable-typed-action-backend")
    if args.typed_action_candidates is not None:
        command.extend(["--typed-action-candidates", str(args.typed_action_candidates)])
    if args.typed_action_max_candidates is not None:
        command.extend(["--typed-action-max-candidates", str(args.typed_action_max_candidates)])


def _auto_enable_efficient_repair(
    args: argparse.Namespace,
    case_category: str | None,
) -> bool:
    if getattr(args, "decision_only", False):
        return False
    if not args.auto_efficient_repair_for_categories:
        return False
    category = str(case_category or "").strip()
    wanted = (
        {str(item).strip() for item in args.efficient_repair_category if str(item).strip()}
        or DEFAULT_EFFICIENT_REPAIR_CATEGORIES
    )
    return category in wanted


RETRYABLE_INFRASTRUCTURE_STATUSES = {
    "gpu_oom",
    "process_killed",
    "subprocess_timeout",
}


def _run_case_with_retries(
    case: Mapping[str, Any],
    command: Sequence[str],
    seed: int,
    *,
    timeout: int,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    retry_count = max(0, int(max_retries))
    for attempt_index in range(retry_count + 1):
        attempt_command = _command_for_attempt(command, attempt_index)
        record = _run_case_once(
            case,
            attempt_command,
            seed,
            timeout=timeout,
            attempt_index=attempt_index,
        )
        attempts.append(_compact_attempt_record(record))
        status = str(record.get("status") or "")
        if status not in RETRYABLE_INFRASTRUCTURE_STATUSES or attempt_index >= retry_count:
            if attempt_index:
                record["attempt"] = attempt_index
            record["attempts"] = attempts
            return record
        if retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)
    # The loop always returns; this fallback keeps type checkers happy.
    record["attempts"] = attempts
    return record


def _command_for_attempt(command: Sequence[str], attempt_index: int) -> list[str]:
    attempt_command = list(command)
    if attempt_index <= 0:
        return attempt_command
    try:
        run_id_index = attempt_command.index("--run-id") + 1
    except ValueError:
        return attempt_command
    if run_id_index >= len(attempt_command):
        return attempt_command
    attempt_command[run_id_index] = f"{attempt_command[run_id_index]}-retry{attempt_index}"
    return attempt_command


def _run_case_once(
    case: Mapping[str, Any],
    command: Sequence[str],
    seed: int,
    *,
    timeout: int,
    attempt_index: int = 0,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "focus": list(case.get("focus", []) or []),
        "seed": seed,
        "command": list(command),
        "attempt": attempt_index,
    }
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "status": "subprocess_timeout",
                "returncode": None,
                "stdout_tail": (exc.stdout or "")[-2000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
        )
        return record

    record["returncode"] = completed.returncode
    record["stdout_tail"] = completed.stdout[-2000:]
    record["stderr_tail"] = completed.stderr[-4000:]
    run_info = _parse_run_info(completed.stdout)
    record.update(run_info)
    run_dir_value = str(run_info.get("run_dir") or "").strip()
    run_dir = Path(run_dir_value) if run_dir_value else None
    if run_dir and run_dir.exists():
        record.update(_summarize_run_dir(run_dir))
    elif completed.returncode == 0:
        record.setdefault("status", "missing_run_dir")
    else:
        record.update(_classify_process_failure(completed.returncode, completed.stdout, completed.stderr))
    return record


def _compact_attempt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt": record.get("attempt", 0),
        "status": record.get("status"),
        "returncode": record.get("returncode"),
        "failure_category": record.get("failure_category"),
        "failure_reason": record.get("failure_reason"),
    }


def _parse_run_info(stdout: str) -> dict[str, Any]:
    text = str(stdout or "")
    for start in [match.start() for match in __import__("re").finditer(r"\{", text)][::-1]:
        try:
            parsed = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _summarize_run_dir(run_dir: Path) -> dict[str, Any]:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return {"run_dir": str(run_dir), "status": "missing_run_json"}
    data = json.loads(run_json.read_text(encoding="utf-8"))
    rounds = data.get("round_records") or []
    last_round = rounds[-1] if rounds else {}
    final_selection = data.get("final_selection")
    final_selection = final_selection if isinstance(final_selection, Mapping) else {}
    if final_selection:
        selected_round = int(final_selection.get("round", len(rounds) - 1))
        if 0 <= selected_round < len(rounds):
            last_round = rounds[selected_round]
    feedback = last_round.get("feedback") if isinstance(last_round, Mapping) else {}
    constraint = feedback.get("constraint_check") if isinstance(feedback, Mapping) else {}
    evaluation = feedback.get("evaluation") if isinstance(feedback, Mapping) else {}
    gate = feedback.get("completion_gate") if isinstance(feedback, Mapping) else {}
    errors = feedback.get("errors") if isinstance(feedback, Mapping) else []
    repair_summary = _summarize_repair_plans(run_dir)
    efficient_summary = _summarize_efficient_repairs(run_dir)
    typed_action_summary = _summarize_typed_actions(run_dir)
    return {
        "run_dir": str(run_dir),
        "status": data.get("status"),
        "rounds": len(rounds),
        "selected_image": final_selection.get("selected_image") or last_round.get("selected_image"),
        "final_selected_round": final_selection.get("round"),
        "final_selection_reason": final_selection.get("reason"),
        "revised_prompt": last_round.get("revised_prompt"),
        "constraint_passed": constraint.get("passed") if isinstance(constraint, Mapping) else None,
        "constraint_score": constraint.get("score") if isinstance(constraint, Mapping) else None,
        "evaluation_passed": evaluation.get("passed") if isinstance(evaluation, Mapping) else None,
        "evaluation_score": evaluation.get("score") if isinstance(evaluation, Mapping) else None,
        "completion_passed": gate.get("passed") if isinstance(gate, Mapping) else None,
        "completion_score": gate.get("score") if isinstance(gate, Mapping) else None,
        "error_count": len(errors) if isinstance(errors, list) else 0,
        "errors": _compact_errors(errors),
        **repair_summary,
        **efficient_summary,
        **typed_action_summary,
    }


def _summarize_repair_plans(run_dir: Path) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    typed_routes: list[str] = []
    primary_actions: list[str] = []
    for path in sorted(run_dir.glob("repair_plan_round_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        route = str(payload.get("typed_route") or "none").strip() or "none"
        action = str(payload.get("primary_action") or "none").strip() or "none"
        typed_routes.append(route)
        primary_actions.append(action)
        plans.append(
            {
                "path": str(path),
                "round": _round_index_from_path(path),
                "typed_route": route,
                "primary_action": action,
                "target_object": payload.get("target_object"),
                "target_attribute": payload.get("target_attribute"),
                "repairable": payload.get("repairable"),
            }
        )
    return {
        "repair_plan_count": len(plans),
        "repair_plans": plans,
        "typed_routes": typed_routes,
        "repair_actions": primary_actions,
        "route_none_count": sum(1 for route in typed_routes if route == "none"),
    }


def _summarize_efficient_repairs(run_dir: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("efficient_repair_round_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        attempts.append(
            {
                "path": str(path),
                "round": payload.get("round", _round_index_from_path(path)),
                "route": payload.get("route"),
                "accepted": bool(payload.get("accepted")),
                "ok": payload.get("ok"),
                "gpu_used": payload.get("gpu_used"),
                "sam2_used": payload.get("sam2_used"),
                "powerpaint_used": payload.get("powerpaint_used"),
                "error": payload.get("error"),
            }
        )
    return {
        "efficient_edit_attempts": len(attempts),
        "accepted_edit_count": sum(1 for item in attempts if item.get("accepted")),
        "efficient_repairs": attempts,
    }


def _summarize_typed_actions(run_dir: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("typed_action_round_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        attempts.append(
            {
                "path": str(path),
                "round": payload.get("round", _round_index_from_path(path)),
                "route": payload.get("route"),
                "accepted": bool(payload.get("accepted")),
                "candidate_count": len(payload.get("candidate_checks", []) or []),
                "selected_index": payload.get("selected_index"),
                "error": payload.get("error"),
            }
        )
    return {
        "typed_action_attempts": len(attempts),
        "typed_action_accepted": sum(1 for item in attempts if item.get("accepted")),
        "typed_actions": attempts,
    }


def _round_index_from_path(path: Path) -> int | None:
    import re

    match = re.search(r"round_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _compact_errors(errors: Any) -> list[dict[str, str]]:
    if not isinstance(errors, list):
        return []
    result: list[dict[str, str]] = []
    for item in errors[:5]:
        if not isinstance(item, Mapping):
            result.append({"type": "unknown", "evidence": str(item)[:240]})
            continue
        result.append(
            {
                "type": str(item.get("type") or item.get("error_type") or ""),
                "target": str(item.get("target") or item.get("prompt_span") or ""),
                "evidence": str(item.get("evidence") or item.get("description") or "")[:240],
            }
        )
    return result


INFRASTRUCTURE_STATUSES = {
    "api_auth_error",
    "api_connection_error",
    "api_dns_error",
    "api_http_error",
    "api_rate_limit",
    "api_timeout",
    "environment_error",
    "gpu_oom",
    "process_killed",
    "subprocess_timeout",
}


def _classify_process_failure(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    text = f"{stdout or ''}\n{stderr or ''}".lower()
    if "temporary failure in name resolution" in text or "socket.gaierror" in text:
        status = "api_dns_error"
    elif "timeoutexpired" in text or "subprocess.timeoutexpired" in text:
        status = "subprocess_timeout"
    elif "api connection error" in text:
        if "timed out" in text or "timeout" in text:
            status = "api_timeout"
        else:
            status = "api_connection_error"
    elif "api http error 401" in text or "unauthorized" in text or "invalid api key" in text:
        status = "api_auth_error"
    elif "api http error 429" in text or "rate limit" in text:
        status = "api_rate_limit"
    elif "api http error" in text:
        status = "api_http_error"
    elif "cuda out of memory" in text or "outofmemoryerror" in text:
        status = "gpu_oom"
    elif returncode in {-9, 137} or "exit code -9" in text or "exit code 137" in text:
        status = "process_killed"
    elif (
        "no module named" in text
        or "modulenotfounderror" in text
        or "localentrynotfounderror" in text
        or "cannot find an appropriate cached snapshot folder" in text
    ):
        status = "environment_error"
    else:
        status = "subprocess_failed"
    return {
        "status": status,
        "failure_category": "infrastructure"
        if status in INFRASTRUCTURE_STATUSES
        else "subprocess",
        "failure_reason": _failure_reason(stdout, stderr),
        "returncode": returncode,
    }


def _failure_reason(stdout: str, stderr: str) -> str:
    for line in reversed((stderr or "").splitlines()):
        text = line.strip()
        if text:
            return text[:300]
    for line in reversed((stdout or "").splitlines()):
        text = line.strip()
        if text:
            return text[:300]
    return ""


def _write_summary(
    runs_dir: Path,
    benchmark: Mapping[str, Any],
    args: argparse.Namespace,
    results: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "benchmark_version": benchmark.get("version"),
        "config": {
            "generator": args.generator,
            "llm": args.llm,
            "vlm": args.vlm,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "max_rounds": args.max_rounds,
            "decision_only": args.decision_only,
            "n_images": args.n_images,
            "auto_n_images_for_hard_prompts": args.auto_n_images_for_hard_prompts,
            "hard_prompt_n_images": args.hard_prompt_n_images,
            "seed_base": args.seed_base,
            "seed_policy": args.seed_policy,
            "cuda_visible_devices": args.cuda_visible_devices,
            "use_layout_planner": args.use_layout_planner,
            "layout_llm": args.layout_llm,
            "retry_on_infra_failure": args.retry_on_infra_failure,
            "retry_delay_seconds": args.retry_delay_seconds,
            "enable_typed_action_backend": args.enable_typed_action_backend,
            "disable_typed_action_backend": args.disable_typed_action_backend,
            "typed_action_candidates": args.typed_action_candidates,
            "typed_action_max_candidates": args.typed_action_max_candidates,
        },
        "results": [dict(item) for item in results],
        "case_metadata": {
            str(case.get("id")): {
                "focus": list(case.get("focus", []) or []),
                "category": case.get("category"),
            }
            for case in benchmark.get("cases", [])
            if isinstance(case, Mapping) and case.get("id")
        },
    }
    for item in payload["results"]:
        if isinstance(item, dict):
            item["failure_layer"] = _determine_failure_layer(item)
    payload["aggregate"] = _aggregate_results(payload["results"])
    atomic_write_text(
        runs_dir / "summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atomic_write_text(runs_dir / "summary.md", _summary_markdown(payload), encoding="utf-8")


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    lines = [
        "# Mini Hard Prompt Benchmark",
        "",
        f"Benchmark: `{payload.get('benchmark_version')}`",
        "",
        "## Aggregate",
        "",
        f"- total: {aggregate.get('total', 0)}",
        f"- evaluable_cases: {aggregate.get('evaluable_cases', 0)}",
        f"- completion_passed: {aggregate.get('completion_passed', 0)}",
        f"- infrastructure_failures: {aggregate.get('infrastructure_failures', 0)}",
        f"- typed_route_counts: {json.dumps(aggregate.get('typed_route_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- route_none_count: {aggregate.get('route_none_count', 0)}",
        f"- efficient_edit_attempts: {aggregate.get('efficient_edit_attempts', 0)}",
        f"- accepted_edit_count: {aggregate.get('accepted_edit_count', 0)}",
        f"- typed_action_attempts: {aggregate.get('typed_action_attempts', 0)}",
        f"- typed_action_accepted: {aggregate.get('typed_action_accepted', 0)}",
        f"- typed_action_route_counts: {json.dumps(aggregate.get('typed_action_route_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- unverifiable_or_clarify_count: {aggregate.get('unverifiable_or_clarify_count', 0)}",
        f"- needs_clarification_count: {aggregate.get('needs_clarification_count', 0)}",
        f"- non_edit_route_skipped_powerpaint_count: {aggregate.get('non_edit_route_skipped_powerpaint_count', 0)}",
        f"- false_pass_blocked_count: {aggregate.get('false_pass_blocked_count', 0)}",
        f"- status_counts: {json.dumps(aggregate.get('status_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- focus_counts: {json.dumps(aggregate.get('focus_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- focus_completion_passed: {json.dumps(aggregate.get('focus_completion_passed', {}), ensure_ascii=False, sort_keys=True)}",
        "",
        "| id | category | status | hard VQA | eval | image |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        hard = _score_cell(item.get("constraint_passed"), item.get("constraint_score"))
        eval_cell = _score_cell(item.get("evaluation_passed"), item.get("evaluation_score"))
        image = item.get("selected_image") or ""
        lines.append(
            "| {id} | {cat} | {status} | {hard} | {eval} | {image} |".format(
                id=item.get("id", ""),
                cat=item.get("category", ""),
                status=item.get("status", ""),
                hard=hard,
                eval=eval_cell,
                image=image,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _determine_failure_layer(result: Mapping[str, Any]) -> str:
    """
    Determine which layer failed for a given case result.

    The whole point of this attribution is to NOT misattribute across layers.
    Critical rule: a present judgment (constraint_passed is not None) means L2
    *ran and produced a verdict*. If that verdict is False, L2 did its job by
    flagging the failure — the fault is downstream in L3, never L2.

    Returns:
        "L1_generation":    Generation backend failed (infra, subprocess, no images)
        "L2_judgment":      Judgment missing when it should have run, OR a false pass
                            (L2 said constraint_passed=True but the case is still wrong)
        "L3_repair":        L2 flagged a failure, L3 fired a repair, but it didn't fix it
        "L3_not_triggered": L2 flagged a failure, but L3 never fired any repair
                            (routing/orchestration gap — e.g. efficient_edit_attempts=0)
        "none":             Success (completion_passed=True)
        "unclear":          Cannot determine failure layer
    """
    status = str(result.get("status") or "unknown")

    # Success case
    if result.get("completion_passed") is True:
        return "none"

    # Prompt issue, not a system failure
    if status == "needs_clarification":
        return "unclear"

    # --- L1 failures: infrastructure or generation problems ---
    if status in INFRASTRUCTURE_STATUSES:
        return "L1_generation"
    if status == "subprocess_failed":
        return "L1_generation"
    if status == "missing_pregenerated_image":
        return "L1_generation"

    # No image generated at all
    rounds = result.get("rounds", 0)
    selected_image = result.get("selected_image")
    if rounds == 0 or not selected_image:
        return "L1_generation"

    # --- L1 succeeded (images generated), now diagnose L2/L3 ---
    constraint_passed = result.get("constraint_passed")
    evaluation_passed = result.get("evaluation_passed")
    completion_passed = result.get("completion_passed")

    typed_routes = result.get("typed_routes") or []
    efficient_edit_attempts = result.get("efficient_edit_attempts", 0) or 0
    typed_action_attempts = result.get("typed_action_attempts", 0) or 0
    # A "none" route is the planner explicitly declining to repair — it is NOT a
    # real repair attempt. Counting it as one would misattribute an L3 routing gap
    # (planner never picked an actionable route) to "L3 fired and failed". Only
    # non-"none" routes, or an actual edit/action attempt, count as L3 firing.
    actionable_routes = [
        route
        for route in typed_routes
        if str(route or "none").strip() not in {"", "none"}
    ]
    has_repair_attempts = (
        len(actionable_routes) > 0
        or efficient_edit_attempts > 0
        or typed_action_attempts > 0
    )

    # L2 never produced a judgment though the run completed -> L2 did not run.
    if constraint_passed is None and evaluation_passed is None:
        if status == "completed":
            return "L2_judgment"
        return "unclear"

    # L2 produced a judgment. If it correctly flagged a hard-constraint failure,
    # the fault lies in L3 — either it fired and failed, or it never triggered.
    if constraint_passed is False:
        if has_repair_attempts:
            return "L3_repair"
        return "L3_not_triggered"

    # L2 said the hard constraint passed but completion still failed: this is a
    # false pass / soft-eval disagreement attributable to the judgment layer.
    if constraint_passed is True and completion_passed is False:
        return "L2_judgment"

    # constraint_passed is None but evaluation exists, and completion failed.
    if completion_passed is False:
        if has_repair_attempts:
            return "L3_repair"
        return "L2_judgment"

    # Default: unclear
    return "unclear"


def _aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_completion_passed: Counter[str] = Counter()
    focus_counts: Counter[str] = Counter()
    focus_completion_passed: Counter[str] = Counter()
    evaluable_cases = 0
    completion_passed = 0
    hard_passed = 0
    evaluation_passed = 0
    infrastructure_failures = 0
    typed_route_counts: Counter[str] = Counter()
    route_none_count = 0
    efficient_edit_attempts = 0
    accepted_edit_count = 0
    typed_action_attempts = 0
    typed_action_accepted = 0
    typed_action_route_counts: Counter[str] = Counter()
    unverifiable_or_clarify_count = 0
    needs_clarification_count = 0
    non_edit_route_skipped_powerpaint_count = 0
    false_pass_blocked_count = 0
    failure_layer_counts: Counter[str] = Counter()
    # B3: per-route hit and post-route success counts. A route "hits" when it
    # appears in a case's typed_routes; it "succeeds" when that case ends
    # completion_passed=True. The ratio answers "once we pick route X, does it
    # actually fix the case?" — which is how we tell a useless route apart from
    # a route that never fires.
    route_hit_counts: Counter[str] = Counter()
    route_success_counts: Counter[str] = Counter()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "unknown")
        category = str(item.get("category") or "unknown")
        status_counts[status] += 1
        category_counts[category] += 1
        focus_values = [
            str(value)
            for value in item.get("focus", []) or []
            if str(value).strip()
        ]
        for focus in focus_values:
            focus_counts[focus] += 1
        if status in INFRASTRUCTURE_STATUSES:
            infrastructure_failures += 1
        if status == "needs_clarification":
            needs_clarification_count += 1
        if item.get("completion_passed") is not None:
            evaluable_cases += 1
        if item.get("completion_passed") is True:
            completion_passed += 1
            category_completion_passed[category] += 1
            for focus in focus_values:
                focus_completion_passed[focus] += 1
        if item.get("constraint_passed") is True:
            hard_passed += 1
        if item.get("evaluation_passed") is True:
            evaluation_passed += 1
        for route in item.get("typed_routes", []) or []:
            route_text = str(route or "none").strip() or "none"
            typed_route_counts[route_text] += 1
        case_completed = item.get("completion_passed") is True
        for route_text in {
            str(route or "none").strip() or "none"
            for route in item.get("typed_routes", []) or []
        }:
            route_hit_counts[route_text] += 1
            if case_completed:
                route_success_counts[route_text] += 1
        route_none_count += int(item.get("route_none_count") or 0)
        efficient_edit_attempts += int(item.get("efficient_edit_attempts") or 0)
        accepted_edit_count += int(item.get("accepted_edit_count") or 0)
        typed_action_attempts += int(item.get("typed_action_attempts") or 0)
        typed_action_accepted += int(item.get("typed_action_accepted") or 0)
        for action in item.get("typed_actions", []) or []:
            if not isinstance(action, Mapping):
                continue
            route = str(action.get("route") or "none").strip() or "none"
            typed_action_route_counts[route] += 1
        if any(
            route in {"lexical_grounding_regeneration", "unverifiable_rare_word_or_clarify"}
            for route in item.get("typed_routes", []) or []
        ):
            unverifiable_or_clarify_count += 1
        for route in item.get("typed_routes", []) or []:
            if route in {
                "count_aware_regeneration",
                "comparative_count_rerank",
                "layout_guided_regeneration",
                "multi_constraint_decompose",
                "comparative_attribute_binding",
                "role_action_binding_regeneration",
                "lexical_grounding_regeneration",
                "relation_focused_regeneration",
            }:
                non_edit_route_skipped_powerpaint_count += 1
        if (
            item.get("constraint_passed") is True
            and item.get("completion_passed") is False
        ):
            false_pass_blocked_count += 1
        failure_layer = _determine_failure_layer(item)
        failure_layer_counts[failure_layer] += 1

    def _rate(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return round(numerator / denominator, 4)

    # B3: agent self-evaluation derived rates. These are ratios over the raw
    # counts above, kept separate from the counts so a reader can answer
    # "is the agent actually helping?" without recomputing by hand.
    self_eval_metrics = {
        # L3 local-edit acceptance: of the efficient edits L3 attempted, how
        # many were accepted (kept over rollback). attempts==0 -> null, which
        # is itself the signal that L3 never fired (see L3_not_triggered).
        "efficient_edit_accept_rate": _rate(accepted_edit_count, efficient_edit_attempts),
        # Typed-action acceptance: same idea for the typed_action backend.
        "typed_action_accept_rate": _rate(typed_action_accepted, typed_action_attempts),
        # False-pass rate: hard VQA said pass but completion gate still failed,
        # over the cases that produced a completion verdict. High -> L2 judgment
        # is letting wrong images through.
        "false_pass_rate": _rate(false_pass_blocked_count, evaluable_cases),
        # Hard-constraint pass rate over evaluable cases (L2 hard verdict).
        "hard_pass_rate": _rate(hard_passed, evaluable_cases),
        # End-to-end completion rate over evaluable cases.
        "completion_rate": _rate(completion_passed, evaluable_cases),
        # Fraction of cases blocked at L1 (no usable image) over all cases.
        "infrastructure_failure_rate": _rate(infrastructure_failures, len(results)),
        # Per-route success rate: of the cases where a given typed route fired,
        # how many ended in completion_passed. This is the core B3 question
        # "does hitting route X actually help?" — e.g. count_aware_regeneration
        # firing repeatedly with a near-zero success rate is the smoking gun
        # that the route is a no-op for that failure class.
        "route_success_rate": {
            route: _rate(route_success_counts.get(route, 0), hits)
            for route, hits in sorted(route_hit_counts.items())
        },
    }

    return {
        "total": len(results),
        "evaluable_cases": evaluable_cases,
        "completion_passed": completion_passed,
        "hard_passed": hard_passed,
        "evaluation_passed": evaluation_passed,
        "infrastructure_failures": infrastructure_failures,
        "typed_route_counts": dict(sorted(typed_route_counts.items())),
        "route_none_count": route_none_count,
        "efficient_edit_attempts": efficient_edit_attempts,
        "accepted_edit_count": accepted_edit_count,
        "typed_action_attempts": typed_action_attempts,
        "typed_action_accepted": typed_action_accepted,
        "typed_action_route_counts": dict(sorted(typed_action_route_counts.items())),
        "unverifiable_or_clarify_count": unverifiable_or_clarify_count,
        "needs_clarification_count": needs_clarification_count,
        "non_edit_route_skipped_powerpaint_count": non_edit_route_skipped_powerpaint_count,
        "false_pass_blocked_count": false_pass_blocked_count,
        "failure_layer_counts": dict(sorted(failure_layer_counts.items())),
        "route_hit_counts": dict(sorted(route_hit_counts.items())),
        "route_success_counts": dict(sorted(route_success_counts.items())),
        "self_eval_metrics": self_eval_metrics,
        "status_counts": dict(sorted(status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "category_completion_passed": dict(sorted(category_completion_passed.items())),
        "focus_counts": dict(sorted(focus_counts.items())),
        "focus_completion_passed": dict(sorted(focus_completion_passed.items())),
    }


def _score_cell(passed: Any, score: Any) -> str:
    if passed is None and score is None:
        return ""
    return f"{passed} / {score}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
