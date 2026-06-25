"""Command-line runner for the M4 orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clients import MockLLMClient, MockVLMClient, client_from_env
from .evaluators import VLMJudgeEvaluator
from .factuality_qa import FactualityQAEvaluator
from .image_generator import (
    DiffusersSDXLGenerator,
    FluxCLIImageGenerator,
    FusionImageGenerator,
    MockImageGenerator,
)
from .layout_planner import LayoutPlanner, build_mock_layout_response
from .logging_utils import DEFAULT_RUNS_DIR
from .orchestrator import OrchestratorAgent
from .editing_agent import (
    DEFAULT_GROUNDED_SAM2_DIR,
    DEFAULT_GROUNDED_SAM2_HF_HOME,
    DEFAULT_GROUNDED_SAM2_PYTHON,
    EfficientRepairAgent,
    GroundedSAM2SubprocessMasker,
    MaskGeneratingInpaintEditor,
)
from .local_editor import (
    DiffusersInpaintEditor,
    MockInpaintEditor,
    PowerPaintSubprocessEditor,
    SubprocessInpaintEditor,
)
from .mask_refiner import BBoxMaskRefiner, MockMaskRefiner, SamV1MaskRefiner
from .relation_repair import RelationActionRepairer
from .reward_reranker import MockRewardBackend, RewardReranker, VLMRewardBackend
from .state import AgentConfig


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_FLUX_REPO = "/home/zrr/flux"
DEFAULT_FLUX_PYTHON = "/mnt/ssd1/conda/envs/flux-dev/bin/python"
DEFAULT_FLUX_MODEL_PATH = "/home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/flux1-dev.safetensors"
DEFAULT_FLUX_AE_PATH = "/home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/ae.safetensors"
DEFAULT_FLUX_HF_HOME = "/mnt/ssd3/zrr/hf_cache"
DEFAULT_MGRAG_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_SDXL_MODEL_PATH = "/mnt/ssd1/models/stable-diffusion-xl-base-1.0"
DEFAULT_SD15_INPAINT_PATH = "/mnt/hdd2/lwt/huggingface/runwayml/stable-diffusion-inpainting"
DEFAULT_SDXL_PYTHON = "/mnt/ssd1/conda/envs/sdxl/bin/python"
DEFAULT_POWERPAINT_CHECKPOINT_DIR = "/mnt/ssd1/models/PowerPaint/ppt-v2-1"
DEFAULT_POWERPAINT_PYTHON = "/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python"
DEFAULT_POWERPAINT_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/PowerPaint"
)
DEFAULT_SAM_V1_PATH = "/mnt/hdd2/lwt/sam_vit_l_0b3195.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FLUX multimodal T2I agent")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help=(
            "Run one generation/evaluation/planning pass and stop before any "
            "typed-action candidate generation, local repair, SAM, or editor call. "
            "This is for cheap route/repair-direction validation."
        ),
    )
    parser.add_argument("--n-images", type=int, default=1)
    parser.add_argument(
        "--auto-n-images-for-hard-prompts",
        action="store_true",
        help=(
            "Raise n_images for compositional prompts that usually benefit from "
            "candidate reranking, while keeping simple prompts at --n-images."
        ),
    )
    parser.add_argument(
        "--hard-prompt-n-images",
        type=int,
        default=2,
        help="Candidate count used when --auto-n-images-for-hard-prompts triggers.",
    )
    parser.add_argument("--creativity-level", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--human-in-loop", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.85)
    parser.add_argument("--clip-token-budget", type=int, default=70)
    parser.add_argument(
        "--disable-constraint-check",
        action="store_true",
        help="Skip the dedicated VLM check for user color/action/relation constraints.",
    )
    parser.add_argument(
        "--disable-specialist-reports",
        action="store_true",
        help="Do not write no-extra-API specialist_reports_round_<n>.json files.",
    )
    parser.add_argument(
        "--enable-specialist-vlm-observation",
        action="store_true",
        help=(
            "When existing feedback is insufficient, call the VLM once for a "
            "structured specialist observation before local specialist analysis."
        ),
    )
    parser.add_argument(
        "--disable-auto-negative-prompt",
        action="store_true",
        help="Skip automatic negative prompts for user-grounded binding conflicts.",
    )
    parser.add_argument(
        "--use-binding-variants",
        action="store_true",
        help="Generate SDXL prompt variants for color/relation binding and let the VLM select among them.",
    )
    parser.add_argument(
        "--disable-clarifier",
        action="store_true",
        help="Skip proactive clarification before generation.",
    )
    parser.add_argument(
        "--auto-merge-clarification",
        default=None,
        help="Answer to merge automatically if the clarifier asks a question.",
    )

    parser.add_argument("--llm", choices=["mock", "api"], default="mock")
    parser.add_argument("--vlm", choices=["mock", "api"], default="mock")
    parser.add_argument("--generator", choices=["mock", "flux", "sdxl", "fusion"], default="mock")

    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-base-url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--llm-model", default="qwen-plus")
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument(
        "--enable-m6-evaluator",
        "--enable-evaluator",
        dest="enable_m6_evaluator",
        action="store_true",
        help="Run VLMJudgeEvaluator after image selection and log evaluation_round_<n>.json.",
    )
    parser.add_argument(
        "--evaluator-vlm",
        choices=["mock", "api"],
        default=None,
        help="Optional VLM backend for M6 evaluator; defaults to the main --vlm backend.",
    )
    parser.add_argument(
        "--evaluator-vlm-model",
        default=None,
        help="Optional API model for the M6 evaluator VLM.",
    )
    parser.add_argument(
        "--enable-factuality-qa",
        action="store_true",
        help="Run I-HallA-style factual QA for factual prompts after the VLM judge.",
    )
    parser.add_argument(
        "--enable-api-reward-reranker",
        "--enable-reward-rerank",
        dest="enable_api_reward_reranker",
        action="store_true",
        help="Use the VLM API as a LLaVA-Reward-style reward proxy to rerank images.",
    )
    parser.add_argument(
        "--reward-backend",
        choices=["mock", "api"],
        default=None,
        help="Reward reranker backend; defaults to api when --vlm api is used, otherwise mock.",
    )
    parser.add_argument(
        "--reward-vlm-model",
        default=None,
        help="Optional API model for the reward reranker VLM proxy.",
    )
    parser.add_argument(
        "--disable-reward-selection-override",
        action="store_true",
        help="Log reward ranking but keep the VLM selector's image choice.",
    )
    parser.add_argument(
        "--enable-typed-action-backend",
        action="store_true",
        help="After typed repair routes fail, generate route-specific candidates and VLM-rerank them.",
    )
    parser.add_argument(
        "--disable-typed-action-backend",
        action="store_true",
        help="Disable the P5.5 typed action backend even when evaluator retry is enabled.",
    )
    parser.add_argument("--typed-action-candidates", type=int, default=3)
    parser.add_argument("--typed-action-max-candidates", type=int, default=4)
    parser.add_argument(
        "--reward-aspects",
        default="overall",
        help="Comma-separated reward aspects: alignment,fidelity,safety,overall.",
    )
    parser.add_argument(
        "--enable-local-repair",
        action="store_true",
        help="When M6/M4 detects color binding failure, try M5.4 recolor repair after selection.",
    )
    parser.add_argument(
        "--enable-vlm-target-locator",
        action="store_true",
        help="Before local recolor, ask the VLM to locate the visible target object/part bbox in the generated image.",
    )
    parser.add_argument(
        "--local-editor",
        choices=["recolor"],
        default="recolor",
        help="Local repair backend. M6.5 currently supports deterministic recolor repair.",
    )
    parser.add_argument(
        "--mask-refiner",
        choices=["bbox", "mock", "sam-v1", "none"],
        default="bbox",
        help=(
            "Mask refinement backend for local repair evidence. bbox is a CPU "
            "fallback; sam-v1 loads SAM only when explicitly selected."
        ),
    )
    parser.add_argument("--sam-checkpoint-path", default=DEFAULT_SAM_V1_PATH)
    parser.add_argument("--sam-model-type", default="vit_l")
    parser.add_argument("--sam-device", default=None)
    parser.add_argument(
        "--enable-relation-repair",
        action="store_true",
        help="When grip/handle/contact relation fails, run local action/contact repair and VLM verification.",
    )
    parser.add_argument(
        "--enable-object-insertion-repair",
        action="store_true",
        help="When required objects are missing or under-counted, run local object insertion and verify constraints.",
    )
    parser.add_argument(
        "--relation-editor",
        choices=["mock", "sd15-inpaint", "sd15-subprocess-inpaint", "powerpaint-subprocess"],
        default="mock",
        help="Local editor backend for relation repair.",
    )
    parser.add_argument("--relation-candidates", type=int, default=3)
    parser.add_argument("--relation-pass-threshold", type=float, default=0.82)
    parser.add_argument("--relation-inpaint-model-path", default=DEFAULT_SD15_INPAINT_PATH)
    parser.add_argument("--relation-inpaint-python", default=DEFAULT_SDXL_PYTHON)
    parser.add_argument("--relation-inpaint-timeout-seconds", type=int, default=900)
    parser.add_argument("--relation-inpaint-cuda-visible-devices", default=None)
    parser.add_argument("--powerpaint-python", default=DEFAULT_POWERPAINT_PYTHON)
    parser.add_argument("--powerpaint-dir", default=DEFAULT_POWERPAINT_DIR)
    parser.add_argument("--powerpaint-checkpoint-dir", default=DEFAULT_POWERPAINT_CHECKPOINT_DIR)
    parser.add_argument("--powerpaint-timeout-seconds", type=int, default=1800)
    parser.add_argument("--relation-steps", type=int, default=20)
    parser.add_argument("--relation-guidance-scale", type=float, default=7.5)
    parser.add_argument("--relation-strength", type=float, default=0.82)
    parser.add_argument(
        "--enable-editing-mask-agent",
        action="store_true",
        help=(
            "Wrap the relation/object editor with a text-mask agent: optional "
            "Grounded-SAM2 mask, mask dilation, then PowerPaint/subprocess edit."
        ),
    )
    parser.add_argument(
        "--editing-mask-mode",
        choices=["auto", "grounded-sam2", "bbox"],
        default="auto",
        help="Mask source for --enable-editing-mask-agent.",
    )
    parser.add_argument(
        "--editing-mask-text",
        default=None,
        help="Optional fixed mask text. Defaults to each repair region name.",
    )
    parser.add_argument(
        "--editing-mask-dilation-kernel-size",
        type=int,
        default=51,
        help="Odd MaxFilter kernel size used to dilate SAM/bbox masks before editing.",
    )
    parser.add_argument("--editing-min-mask-area-ratio", type=float, default=0.0005)
    parser.add_argument(
        "--enable-efficient-repair-agent",
        action="store_true",
        help=(
            "Enable cheap typed repairs before heavy editors: deterministic text/"
            "symbol overlay and bbox-first insertion routing."
        ),
    )
    parser.add_argument(
        "--allow-editing-bbox-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bbox masks when Grounded-SAM2 is unavailable or returns no usable mask.",
    )
    parser.add_argument("--grounded-sam2-python", default=DEFAULT_GROUNDED_SAM2_PYTHON)
    parser.add_argument("--grounded-sam2-dir", default=DEFAULT_GROUNDED_SAM2_DIR)
    parser.add_argument("--grounded-sam2-timeout-seconds", type=int, default=900)
    parser.add_argument("--grounded-sam2-cuda-visible-devices", default=None)
    parser.add_argument("--grounded-sam2-hf-home", default=DEFAULT_GROUNDED_SAM2_HF_HOME)
    parser.add_argument(
        "--use-layout-planner",
        action="store_true",
        help="Plan a LayerCraft-style layout and prepend compact layout guidance to generation prompts.",
    )
    parser.add_argument("--layout-llm", choices=["mock", "api"], default=None)
    parser.add_argument("--layout-llm-model", default=None)
    parser.add_argument("--layout-canvas-width", type=int, default=1024)
    parser.add_argument("--layout-canvas-height", type=int, default=1024)
    parser.add_argument(
        "--strict-layout-background",
        action="store_true",
        help="Fail if the layout background repeats foreground object terms.",
    )

    parser.add_argument("--flux-repo", default=DEFAULT_FLUX_REPO)
    parser.add_argument("--flux-python", default=DEFAULT_FLUX_PYTHON)
    parser.add_argument("--flux-name", default="flux-dev")
    parser.add_argument("--flux-model-path", default=DEFAULT_FLUX_MODEL_PATH)
    parser.add_argument("--flux-ae-path", default=DEFAULT_FLUX_AE_PATH)
    parser.add_argument("--flux-hf-home", default=DEFAULT_FLUX_HF_HOME)
    parser.add_argument(
        "--flux-attn-mode",
        choices=["mgrag", "baseline"],
        default="mgrag",
        help="Use M-GRAG attention intervention by default, or baseline for original FLUX.",
    )
    parser.add_argument("--mgrag-model-id", default=DEFAULT_MGRAG_FLUX_MODEL_ID)
    parser.add_argument("--mgrag-delta-scale", type=float, default=1.3)
    parser.add_argument("--mgrag-bias-scale", type=float, default=1.0)
    parser.add_argument("--mgrag-intervene-steps", type=int, default=20)
    parser.add_argument("--mgrag-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--mgrag-image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--mgrag-online",
        action="store_true",
        help="Allow M-GRAG diffusers FLUX loading to contact Hugging Face instead of local cache only.",
    )
    parser.add_argument(
        "--flux-online",
        action="store_true",
        help="Allow the FLUX backend to contact Hugging Face instead of forcing local cache/offline mode.",
    )
    parser.add_argument(
        "--flux-no-offload",
        action="store_true",
        help="Keep FLUX modules on GPU instead of CPU offloading between stages.",
    )
    parser.add_argument(
        "--mgrag-cpu-offload-mode",
        choices=["model", "sequential", "none"],
        default="model",
        help=(
            "CPU offload strategy for the M-GRAG diffusers backend. "
            "model=enable_model_cpu_offload (default, peak ~20G VRAM); "
            "sequential=enable_sequential_cpu_offload (slower, peak ~12G VRAM, "
            "survives GPU contention); none=no offload (fastest, needs ~24G VRAM)."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for local FLUX generation.",
    )
    parser.add_argument(
        "--flux-timeout-seconds",
        type=int,
        default=None,
        help="Timeout for the FLUX subprocess.",
    )
    parser.add_argument("--sdxl-model-path", default=DEFAULT_SDXL_MODEL_PATH)
    parser.add_argument("--sdxl-single-file", default=None)
    parser.add_argument("--sdxl-variant", default="fp16")
    parser.add_argument("--sdxl-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--sdxl-device", default=None)
    parser.add_argument(
        "--sdxl-python",
        default=None,
        help="Reserved compatibility option; SDXL diffusers runs in the current Python process.",
    )
    parser.add_argument(
        "--sdxl-env",
        default=None,
        help="Reserved compatibility option for documenting the intended SDXL environment.",
    )
    parser.add_argument(
        "--fusion-policy",
        choices=["parallel", "flux-first", "sdxl-repair"],
        default="parallel",
        help="FLUX-first candidate policy when --generator fusion is used.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Extra negative prompt appended to the automatic M4.2 negative prompt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_n_images, n_images_reason = _effective_n_images(args)
    effective_max_rounds = 1 if args.decision_only else args.max_rounds
    action_backends_enabled = not args.decision_only
    config = AgentConfig(
        human_in_loop=args.human_in_loop,
        creativity_level=args.creativity_level,
        n_images=effective_n_images,
        max_rounds=effective_max_rounds,
        seed=args.seed,
    )
    llm = _build_llm(args)
    vlm = _build_vlm(args)
    generator = _build_generator(args)
    layout_planner = _build_layout_planner(args)
    evaluator = _build_evaluator(args, vlm)
    factuality_evaluator = _build_factuality_evaluator(args, vlm, llm)
    reward_reranker = _build_reward_reranker(args, vlm)
    relation_repairer = _build_relation_repairer(args, vlm) if action_backends_enabled else None
    efficient_repair_agent = (
        _build_efficient_repair_agent(args, relation_repairer)
        if action_backends_enabled
        else None
    )
    mask_refiner = _build_mask_refiner(args) if action_backends_enabled else None
    mode = "mock" if args.generator == "mock" and args.llm == "mock" and args.vlm == "mock" else "local"
    enable_typed_action_backend = (
        args.enable_typed_action_backend
        and not args.disable_typed_action_backend
        and action_backends_enabled
    )

    agent = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=config,
        runs_dir=args.runs_dir,
        mode=mode,
        score_threshold=args.score_threshold,
        prompt_candidates_per_round=1,
        enable_clarifier=not args.disable_clarifier,
        auto_merge_clarification=args.auto_merge_clarification,
        clip_token_budget=args.clip_token_budget,
        enable_constraint_check=not args.disable_constraint_check,
        enable_specialist_reports=not args.disable_specialist_reports,
        enable_specialist_vlm_observation=args.enable_specialist_vlm_observation,
        auto_negative_prompt=not args.disable_auto_negative_prompt,
        negative_prompt=args.negative_prompt,
        enable_layout_planner=args.use_layout_planner,
        layout_planner=layout_planner,
        layout_canvas_size=(args.layout_canvas_width, args.layout_canvas_height),
        enable_binding_variants=(
            args.use_binding_variants
            or (
                args.auto_n_images_for_hard_prompts
                and n_images_reason == "auto_hard_compositional_prompt"
            )
        ),
        evaluator=evaluator,
        enable_evaluator=args.enable_m6_evaluator,
        factuality_evaluator=factuality_evaluator,
        enable_factuality_qa=args.enable_factuality_qa,
        reward_reranker=reward_reranker,
        enable_reward_reranker=args.enable_api_reward_reranker,
        reward_rerank_override=not args.disable_reward_selection_override,
        enable_local_repair=args.enable_local_repair and action_backends_enabled,
        enable_vlm_target_locator=args.enable_vlm_target_locator and action_backends_enabled,
        relation_repairer=relation_repairer,
        enable_relation_repair=args.enable_relation_repair and action_backends_enabled,
        enable_object_insertion_repair=(
            args.enable_object_insertion_repair and action_backends_enabled
        ),
        efficient_repair_agent=efficient_repair_agent,
        enable_efficient_repair_agent=(
            args.enable_efficient_repair_agent and action_backends_enabled
        ),
        mask_refiner=mask_refiner,
        enable_mask_refiner=mask_refiner is not None and action_backends_enabled,
        enable_typed_action_backend=enable_typed_action_backend,
        typed_action_candidates=args.typed_action_candidates,
        typed_action_max_candidates=args.typed_action_max_candidates,
    )
    result = agent.run(args.prompt, run_id=args.run_id)
    print(
        json.dumps(
            {
                "run_dir": result.run_dir,
                "status": result.status,
                "rounds": len(result.round_records),
                "final_report": result.final_report_path,
                "n_images": effective_n_images,
                "n_images_reason": n_images_reason,
                "decision_only": args.decision_only,
                "effective_max_rounds": effective_max_rounds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _effective_n_images(args: argparse.Namespace) -> tuple[int, str]:
    base = max(1, int(args.n_images))
    if not args.auto_n_images_for_hard_prompts:
        return base, "explicit_n_images"
    hard_count = max(base, int(args.hard_prompt_n_images))
    if _is_hard_compositional_prompt(args.prompt):
        return hard_count, "auto_hard_compositional_prompt"
    return base, "simple_or_unclassified_prompt"


def _is_hard_compositional_prompt(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    hard_terms = (
        "left of",
        "right of",
        "under",
        "above",
        "behind",
        "in front of",
        "next to",
        "inside",
        "contains",
        "without",
        " no ",
        "not attached",
        "holding",
        "holds",
        "carries",
        "touches",
        "gripping",
        "fully visible",
        "partially covers",
        "hides",
        "exact",
        "text",
        "symbol",
        "plain",
        "separate",
    )
    number_terms = (
        "two",
        "three",
        "four",
        "five",
        "exactly",
        "one ",
        "1 ",
        "2 ",
        "3 ",
    )
    color_hits = sum(
        1
        for color in (
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "silver",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "teal",
        )
        if f"{color} " in lowered
    )
    relation_hit = any(term in f" {lowered} " for term in hard_terms)
    count_hit = any(term in f" {lowered} " for term in number_terms)
    multi_object_hit = lowered.count(",") >= 1 or " and " in lowered or ";" in lowered
    return relation_hit or (count_hit and multi_object_hit) or color_hits >= 3


def _build_llm(args: argparse.Namespace):
    if args.llm == "api":
        return client_from_env(
            kind="llm",
            model=args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
        )
    responses: list[str] = []
    if not args.disable_clarifier:
        responses.append("not json belief")
    responses.append(json.dumps({"prompts": [args.prompt]}, ensure_ascii=False))
    return MockLLMClient(
        responses=responses,
        default_response=json.dumps({"prompts": [args.prompt]}, ensure_ascii=False),
    )


def _build_vlm(args: argparse.Namespace):
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


def _build_generator(args: argparse.Namespace):
    if args.generator == "flux":
        return _build_flux_generator(args)
    if args.generator == "sdxl":
        return _build_sdxl_generator(args)
    if args.generator == "fusion":
        return FusionImageGenerator(
            flux=_build_flux_generator(args),
            sdxl=_build_sdxl_generator(args),
            policy=args.fusion_policy,
            negative_prompt=args.negative_prompt,
        )
    return MockImageGenerator()


def _build_flux_generator(args: argparse.Namespace) -> FluxCLIImageGenerator:
    return FluxCLIImageGenerator(
        flux_repo=args.flux_repo,
        python=args.flux_python,
        attention_mode=args.flux_attn_mode,
        mgrag_model_id=args.mgrag_model_id,
        mgrag_delta_scale=args.mgrag_delta_scale,
        mgrag_bias_scale=args.mgrag_bias_scale,
        mgrag_intervene_steps=args.mgrag_intervene_steps,
        mgrag_local_files_only=not args.mgrag_online,
        mgrag_dtype=args.mgrag_dtype,
        mgrag_image_format=args.mgrag_image_format,
        mgrag_cpu_offload_mode=args.mgrag_cpu_offload_mode,
        model_name=args.flux_name,
        model_path=args.flux_model_path,
        ae_path=args.flux_ae_path,
        hf_home=args.flux_hf_home,
        output_dir=args.runs_dir,
        device=args.device,
        offload=not args.flux_no_offload,
        offline=not args.flux_online,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        timeout_seconds=args.flux_timeout_seconds,
        cuda_visible_devices=args.cuda_visible_devices,
    )


def _build_sdxl_generator(args: argparse.Namespace) -> DiffusersSDXLGenerator:
    return DiffusersSDXLGenerator(
        model_path=args.sdxl_model_path,
        single_file=args.sdxl_single_file,
        variant=args.sdxl_variant or None,
        dtype=args.sdxl_dtype,
        output_dir=args.runs_dir,
        device=args.sdxl_device or args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )


def _build_layout_planner(args: argparse.Namespace):
    if not args.use_layout_planner:
        return None
    layout_llm = args.layout_llm or args.llm
    if layout_llm == "api":
        client = client_from_env(
            kind="llm",
            model=args.layout_llm_model or args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
        )
    else:
        client = MockLLMClient(
            responses=[
                build_mock_layout_response(
                    args.prompt,
                    canvas_size=(args.layout_canvas_width, args.layout_canvas_height),
                )
            ]
        )
    return LayoutPlanner(client, strict_background=args.strict_layout_background)


def _build_evaluator(args: argparse.Namespace, vlm):
    if not args.enable_m6_evaluator:
        return None
    evaluator_vlm = _build_optional_vlm(
        args,
        backend=args.evaluator_vlm,
        model=args.evaluator_vlm_model,
        fallback=vlm,
    )
    return VLMJudgeEvaluator(evaluator_vlm)


def _build_factuality_evaluator(args: argparse.Namespace, vlm, llm):
    if not args.enable_factuality_qa:
        return None
    return FactualityQAEvaluator(vlm, llm=llm)


def _build_reward_reranker(args: argparse.Namespace, vlm):
    if not args.enable_api_reward_reranker:
        return None
    reward_backend = args.reward_backend or ("api" if args.vlm == "api" else "mock")
    if reward_backend == "api":
        reward_vlm = _build_optional_vlm(
            args,
            backend="api",
            model=args.reward_vlm_model,
            fallback=vlm,
        )
        backend = VLMRewardBackend(reward_vlm)
    else:
        backend = MockRewardBackend(default_score=0.8)
    aspects = [part.strip() for part in args.reward_aspects.split(",") if part.strip()]
    return RewardReranker(backend, aspects=aspects or ("overall",))


def _build_optional_vlm(
    args: argparse.Namespace,
    *,
    backend: str | None,
    model: str | None,
    fallback,
):
    if backend is None:
        if model and args.vlm == "api" and model != args.vlm_model:
            return _build_api_vlm(args, model)
        return fallback
    if backend == "api":
        chosen_model = model or args.vlm_model
        if args.vlm == "api" and chosen_model == args.vlm_model:
            return fallback
        return _build_api_vlm(args, chosen_model)
    return MockVLMClient(
        default_response=json.dumps(
            {
                "score": 0.85,
                "passed": True,
                "criteria_scores": {},
                "errors": [],
                "strengths": ["Mock optional VLM accepted the image."],
                "revision_hint": "",
            },
            ensure_ascii=False,
        )
    )


def _build_api_vlm(args: argparse.Namespace, model: str):
    return client_from_env(
        kind="vlm",
        model=model,
        api_key_env=args.api_key_env,
        base_url=args.api_base_url,
    )


def _build_relation_repairer(args: argparse.Namespace, vlm):
    if not args.enable_relation_repair and not args.enable_object_insertion_repair:
        return None
    if args.relation_editor == "powerpaint-subprocess":
        editor = PowerPaintSubprocessEditor(
            checkpoint_dir=args.powerpaint_checkpoint_dir,
            python=args.powerpaint_python,
            powerpaint_dir=args.powerpaint_dir,
            device=args.device,
            dtype="float16",
            guidance_scale=args.relation_guidance_scale,
            num_inference_steps=args.relation_steps,
            strength=args.relation_strength,
            seed=args.seed,
            prefix="relation_powerpaint",
            task="text-guided",
            timeout_seconds=args.powerpaint_timeout_seconds,
            cuda_visible_devices=args.relation_inpaint_cuda_visible_devices,
            local_files_only=True,
        )
    elif args.relation_editor == "sd15-subprocess-inpaint":
        editor = SubprocessInpaintEditor(
            model_path=args.relation_inpaint_model_path,
            python=args.relation_inpaint_python,
            device=args.device,
            dtype="float16",
            guidance_scale=args.relation_guidance_scale,
            num_inference_steps=args.relation_steps,
            strength=args.relation_strength,
            seed=args.seed,
            prefix="relation_sd15_subprocess_inpaint",
            pipeline="sd15",
            timeout_seconds=args.relation_inpaint_timeout_seconds,
            cuda_visible_devices=args.relation_inpaint_cuda_visible_devices,
        )
    elif args.relation_editor == "sd15-inpaint":
        editor = DiffusersInpaintEditor(
            model_path=args.relation_inpaint_model_path,
            device=args.device,
            dtype="float16",
            guidance_scale=args.relation_guidance_scale,
            num_inference_steps=args.relation_steps,
            strength=args.relation_strength,
            seed=args.seed,
            prefix="relation_sd15_inpaint",
        )
    else:
        editor = MockInpaintEditor(prefix="relation_mock_inpaint")
    if args.enable_editing_mask_agent:
        mask_generator = None
        if args.editing_mask_mode in {"auto", "grounded-sam2"}:
            mask_generator = GroundedSAM2SubprocessMasker(
                python=args.grounded_sam2_python,
                grounded_sam2_dir=args.grounded_sam2_dir,
                timeout_seconds=args.grounded_sam2_timeout_seconds,
                cuda_visible_devices=args.grounded_sam2_cuda_visible_devices,
                local_files_only=True,
                hf_home=args.grounded_sam2_hf_home,
            )
        editor = MaskGeneratingInpaintEditor(
            base_editor=editor,
            mask_generator=mask_generator,
            mask_mode=args.editing_mask_mode,
            mask_text=args.editing_mask_text,
            allow_bbox_fallback=args.allow_editing_bbox_fallback,
            dilation_kernel_size=args.editing_mask_dilation_kernel_size,
            min_mask_area_ratio=args.editing_min_mask_area_ratio,
            prefix="relation_editing_agent",
        )
    return RelationActionRepairer(
        vlm,
        editor,
        candidates=args.relation_candidates,
        pass_threshold=args.relation_pass_threshold,
    )


def _build_efficient_repair_agent(args: argparse.Namespace, relation_repairer):
    if not args.enable_efficient_repair_agent:
        return None
    inpaint_agent = None
    editor = getattr(relation_repairer, "editor", None) if relation_repairer is not None else None
    if editor is not None:
        mask_generator = None
        if args.editing_mask_mode in {"auto", "grounded-sam2"}:
            mask_generator = GroundedSAM2SubprocessMasker(
                python=args.grounded_sam2_python,
                grounded_sam2_dir=args.grounded_sam2_dir,
                timeout_seconds=args.grounded_sam2_timeout_seconds,
                cuda_visible_devices=args.grounded_sam2_cuda_visible_devices,
                local_files_only=True,
                hf_home=args.grounded_sam2_hf_home,
            )
        from .editing_agent import GroundedSAM2PowerPaintEditingAgent

        inpaint_agent = GroundedSAM2PowerPaintEditingAgent(
            editor=editor,
            mask_generator=mask_generator,
            mask_mode=args.editing_mask_mode,
            allow_bbox_fallback=args.allow_editing_bbox_fallback,
            dilation_kernel_size=args.editing_mask_dilation_kernel_size,
            min_mask_area_ratio=args.editing_min_mask_area_ratio,
        )
    return EfficientRepairAgent(inpaint_agent=inpaint_agent)


def _build_mask_refiner(args: argparse.Namespace):
    if not args.enable_local_repair or args.mask_refiner == "none":
        return None
    if args.mask_refiner == "mock":
        return MockMaskRefiner()
    if args.mask_refiner == "sam-v1":
        return SamV1MaskRefiner(
            checkpoint_path=args.sam_checkpoint_path,
            model_type=args.sam_model_type,
            device=args.sam_device or args.device,
        )
    return BBoxMaskRefiner()


if __name__ == "__main__":
    raise SystemExit(main())
