from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.editing_agent import (
    DEFAULT_GROUNDED_SAM2_DIR,
    DEFAULT_GROUNDED_SAM2_HF_HOME,
    DEFAULT_GROUNDED_SAM2_PYTHON,
    EfficientRepairAgent,
    EfficientRepairRequest,
    GroundedSAM2PowerPaintEditingAgent,
    GroundedSAM2SubprocessMasker,
)
from src.local_editor import MockInpaintEditor, PowerPaintSubprocessEditor


DEFAULT_POWERPAINT_CHECKPOINT_DIR = "/mnt/ssd1/models/PowerPaint/ppt-v2-1"
DEFAULT_POWERPAINT_PYTHON = "/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python"
DEFAULT_POWERPAINT_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/PowerPaint"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Grounded-SAM2/bbox mask + dilation + PowerPaint edit on one image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--prompt", required=True, help="PowerPaint edit prompt.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--repair-kind",
        choices=[
            "text_overlay",
            "symbol_overlay",
            "shape_overlay",
            "bbox_shape_inpaint",
            "existing_object_inpaint",
            "layout_regenerate",
            "count_rerank",
        ],
        default=None,
        help=(
            "Use the efficient typed repair router. Omit this to keep the legacy "
            "Grounded-SAM2/bbox mask + inpaint path."
        ),
    )
    parser.add_argument("--text", default="", help="Exact text for text_overlay repairs.")
    parser.add_argument("--symbol", default="", help="Simple symbol for symbol_overlay repairs.")
    parser.add_argument("--fill-color", default="", help="Background/fill color for deterministic overlay repairs.")
    parser.add_argument("--text-color", default="", help="Text/symbol color for deterministic overlay repairs.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="Fallback/source bbox in source-image pixels unless --canvas-size is set.",
    )
    parser.add_argument("--canvas-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    parser.add_argument("--mask-text", default=None)
    parser.add_argument("--mask-mode", choices=["auto", "grounded-sam2", "bbox"], default="auto")
    parser.add_argument("--allow-bbox-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dilation-kernel-size", type=int, default=51)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0005)
    parser.add_argument("--editor", choices=["powerpaint-subprocess", "mock"], default="powerpaint-subprocess")
    parser.add_argument("--grounded-sam2-python", default=DEFAULT_GROUNDED_SAM2_PYTHON)
    parser.add_argument("--grounded-sam2-dir", default=DEFAULT_GROUNDED_SAM2_DIR)
    parser.add_argument("--grounded-sam2-timeout-seconds", type=int, default=900)
    parser.add_argument("--grounded-sam2-cuda-visible-devices", default=None)
    parser.add_argument("--grounded-sam2-hf-home", default=DEFAULT_GROUNDED_SAM2_HF_HOME)
    parser.add_argument("--powerpaint-python", default=DEFAULT_POWERPAINT_PYTHON)
    parser.add_argument("--powerpaint-dir", default=DEFAULT_POWERPAINT_DIR)
    parser.add_argument("--powerpaint-checkpoint-dir", default=DEFAULT_POWERPAINT_CHECKPOINT_DIR)
    parser.add_argument("--powerpaint-timeout-seconds", type=int, default=1800)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--task", choices=["text-guided", "object-removal"], default="text-guided")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "editing_agent_result.json"

    try:
        with Image.open(image_path) as image:
            canvas_size = list(args.canvas_size or image.size)
        inpaint_agent = None
        if args.repair_kind is None or args.repair_kind in {"bbox_shape_inpaint", "existing_object_inpaint"}:
            mask_generator = None
            if args.mask_mode in {"auto", "grounded-sam2"}:
                mask_generator = GroundedSAM2SubprocessMasker(
                    python=args.grounded_sam2_python,
                    grounded_sam2_dir=args.grounded_sam2_dir,
                    timeout_seconds=args.grounded_sam2_timeout_seconds,
                    cuda_visible_devices=args.grounded_sam2_cuda_visible_devices,
                    local_files_only=bool(args.local_files_only),
                    hf_home=args.grounded_sam2_hf_home,
                )
            if args.editor == "mock":
                editor = MockInpaintEditor(prefix="editing_agent_mock")
            else:
                editor = PowerPaintSubprocessEditor(
                    checkpoint_dir=args.powerpaint_checkpoint_dir,
                    python=args.powerpaint_python,
                    powerpaint_dir=args.powerpaint_dir,
                    dtype=args.dtype,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps,
                    strength=args.strength,
                    seed=args.seed,
                    prefix="editing_agent_powerpaint",
                    task=args.task,
                    timeout_seconds=args.powerpaint_timeout_seconds,
                    cuda_visible_devices=args.cuda_visible_devices,
                    local_files_only=bool(args.local_files_only),
                )
            inpaint_agent = GroundedSAM2PowerPaintEditingAgent(
                editor=editor,
                mask_generator=mask_generator,
                mask_mode=args.mask_mode,
                allow_bbox_fallback=bool(args.allow_bbox_fallback),
                dilation_kernel_size=args.dilation_kernel_size,
                min_mask_area_ratio=args.min_mask_area_ratio,
            )
        if args.repair_kind is not None:
            agent = EfficientRepairAgent(inpaint_agent=inpaint_agent)
            payload = agent.repair(
                EfficientRepairRequest(
                    repair_kind=args.repair_kind,
                    image_path=str(image_path),
                    output_dir=output_dir,
                    target_object=args.target_object,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    bbox=list(args.bbox),
                    canvas_size=canvas_size,
                    mask_text=args.mask_text,
                    text=args.text,
                    symbol=args.symbol,
                    fill_color=args.fill_color,
                    text_color=args.text_color,
                    reason="run_editing_agent_smoke efficient route",
                )
            )
        else:
            assert inpaint_agent is not None
            payload = inpaint_agent.edit(
                image_path=str(image_path),
                output_dir=output_dir,
                target_object=args.target_object,
                edit_prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                bbox=list(args.bbox),
                canvas_size=canvas_size,
                mask_text=args.mask_text,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        payload: dict[str, Any] = {
            "ok": False,
            "type": "grounded_sam2_powerpaint_editing_agent",
            "source_image": str(image_path),
            "error": str(exc),
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
