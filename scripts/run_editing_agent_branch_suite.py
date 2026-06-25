from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.editing_agent import (  # noqa: E402
    EfficientRepairAgent,
    EfficientRepairRequest,
    GroundedSAM2PowerPaintEditingAgent,
)
from src.local_editor import MockInpaintEditor, PowerPaintSubprocessEditor  # noqa: E402
from src.logging_utils import write_json  # noqa: E402


DEFAULT_POWERPAINT_CHECKPOINT_DIR = "/mnt/ssd1/models/PowerPaint/ppt-v2-1"
DEFAULT_POWERPAINT_PYTHON = "/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python"
DEFAULT_POWERPAINT_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/PowerPaint"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local branch suite for EfficientRepairAgent. The default mock "
            "mode covers every route without loading GPU models."
        )
    )
    parser.add_argument("--image", default="", help="Optional source image. A synthetic test card is created if omitted.")
    parser.add_argument("--output-dir", default="runs_edit_smoke/local_branch_suite")
    parser.add_argument("--editor", choices=["mock", "powerpaint-subprocess"], default="mock")
    parser.add_argument(
        "--include-real-inpaint",
        action="store_true",
        help="Run bbox/existing-object inpaint with the selected real editor. Text/symbol still stay CPU-only.",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--powerpaint-python", default=DEFAULT_POWERPAINT_PYTHON)
    parser.add_argument("--powerpaint-dir", default=DEFAULT_POWERPAINT_DIR)
    parser.add_argument("--powerpaint-checkpoint-dir", default=DEFAULT_POWERPAINT_CHECKPOINT_DIR)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = _prepare_source_image(args.image, output_dir)

    inpaint_agent = _build_inpaint_agent(args)
    agent = EfficientRepairAgent(inpaint_agent=inpaint_agent)
    cases = _suite_cases(source_image)

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_dir = output_dir / f"{index:02d}_{case['repair_kind']}"
        request = EfficientRepairRequest(
            repair_kind=case["repair_kind"],
            image_path=source_image,
            output_dir=case_dir,
            bbox=case["bbox"],
            target_object=case.get("target_object", ""),
            prompt=case.get("prompt", ""),
            text=case.get("text", ""),
            symbol=case.get("symbol", ""),
            fill_color=case.get("fill_color", ""),
            text_color=case.get("text_color", ""),
            mask_text=case.get("mask_text"),
            canvas_size=[512, 512],
            reason=case.get("reason", "local branch suite"),
        )
        if case["repair_kind"] in {"bbox_shape_inpaint", "existing_object_inpaint"}:
            if args.editor != "mock" and not args.include_real_inpaint:
                result = {
                    "ok": False,
                    "accepted": False,
                    "route": case["repair_kind"],
                    "skipped": True,
                    "reason": "real inpaint branch skipped; pass --include-real-inpaint to run PowerPaint",
                    "gpu_used": False,
                    "sam2_used": False,
                    "powerpaint_used": False,
                }
                write_json(case_dir / "efficient_repair_result.json", result)
            else:
                result = agent.repair(request)
        else:
            result = agent.repair(request)
        if args.editor == "mock" and case["repair_kind"] in {"bbox_shape_inpaint", "existing_object_inpaint"}:
            _write_mock_visual(source_image, case, case_dir, result)
        results.append(
            {
                "case_id": case["case_id"],
                "repair_kind": case["repair_kind"],
                "output_dir": str(case_dir),
                "ok": result.get("ok"),
                "accepted": result.get("accepted"),
                "route": result.get("route"),
                "gpu_used": result.get("gpu_used"),
                "sam2_used": result.get("sam2_used"),
                "powerpaint_used": result.get("powerpaint_used"),
                "edited_image": result.get("edited_image") or _nested_edited_image(result),
                "mask_path": result.get("mask_path") or _nested_mask_path(result),
                "contact_sheet": result.get("contact_sheet"),
                "reason": result.get("reason"),
                "error": result.get("error"),
                "skipped": result.get("skipped", False),
            }
        )

    summary = {
        "source_image": str(source_image),
        "output_dir": str(output_dir),
        "editor": args.editor,
        "include_real_inpaint": bool(args.include_real_inpaint),
        "cases": results,
    }
    write_json(output_dir / "branch_suite_summary.json", summary)
    grid_path = _write_result_grid(output_dir, results)
    summary["grid_path"] = str(grid_path) if grid_path else None
    write_json(output_dir / "branch_suite_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_inpaint_agent(args: argparse.Namespace) -> GroundedSAM2PowerPaintEditingAgent:
    if args.editor == "mock":
        editor = MockInpaintEditor(prefix="branch_suite_mock")
    else:
        editor = PowerPaintSubprocessEditor(
            checkpoint_dir=args.powerpaint_checkpoint_dir,
            python=args.powerpaint_python,
            powerpaint_dir=args.powerpaint_dir,
            dtype="float16",
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            strength=args.strength,
            seed=args.seed,
            prefix="branch_suite_powerpaint",
            timeout_seconds=args.timeout_seconds,
            cuda_visible_devices=args.cuda_visible_devices,
            local_files_only=True,
        )
    return GroundedSAM2PowerPaintEditingAgent(
        editor=editor,
        mask_generator=None,
        mask_mode="bbox",
        allow_bbox_fallback=True,
        dilation_kernel_size=1,
    )


def _prepare_source_image(image_arg: str, output_dir: Path) -> Path:
    if image_arg:
        return Path(image_arg).expanduser().resolve()
    path = output_dir / "synthetic_source.png"
    image = Image.new("RGB", (512, 512), (232, 235, 229))
    draw = ImageDraw.Draw(image)
    draw.rectangle([38, 36, 232, 176], fill=(30, 80, 170), outline=(10, 10, 10), width=3)
    draw.text((88, 82), "OLD", fill=(245, 245, 245), font=_font(38))
    draw.rectangle([280, 42, 468, 178], fill=(245, 156, 35), outline=(10, 10, 10), width=3)
    draw.ellipse([92, 276, 242, 426], fill=(40, 150, 90), outline=(10, 10, 10), width=3)
    draw.rectangle([310, 288, 452, 420], fill=(154, 78, 185), outline=(10, 10, 10), width=3)
    draw.text((62, 202), "local editing agent branch test", fill=(40, 40, 40), font=_font(24))
    image.save(path)
    return path


def _suite_cases(source_image: Path) -> list[dict[str, Any]]:
    del source_image
    return [
        {
            "case_id": "text_overlay_exact",
            "repair_kind": "text_overlay",
            "bbox": [38, 36, 194, 140],
            "target_object": "left blue sign",
            "prompt": "Render exact yellow text 'NO' on a black sign.",
            "text": "NO",
            "fill_color": "black",
            "text_color": "yellow",
            "reason": "exact text repair should be CPU-only and OCR-verifiable",
        },
        {
            "case_id": "symbol_overlay_triangle",
            "repair_kind": "symbol_overlay",
            "bbox": [280, 42, 188, 136],
            "target_object": "right orange panel",
            "prompt": "Draw a white triangle symbol on a purple panel.",
            "symbol": "triangle",
            "fill_color": "purple",
            "text_color": "white",
            "reason": "simple symbol repair should be CPU-only",
        },
        {
            "case_id": "bbox_shape_inpaint_new_occluder",
            "repair_kind": "shape_overlay",
            "bbox": [98, 338, 140, 88],
            "target_object": "flat red screen",
            "prompt": "Add a flat red screen covering the lower half of the green circle.",
            "fill_color": "red",
            "reason": "simple planar occluder should use deterministic bbox overlay",
        },
        {
            "case_id": "bbox_shape_inpaint_new_complex_object",
            "repair_kind": "bbox_shape_inpaint",
            "bbox": [70, 260, 170, 120],
            "target_object": "small toy car",
            "prompt": "Add a small glossy toy car on the floor.",
            "reason": "new non-primitive object should use inpaint, not shape overlay",
        },
        {
            "case_id": "existing_object_inpaint_recolor",
            "repair_kind": "existing_object_inpaint",
            "bbox": [310, 288, 142, 132],
            "target_object": "purple rectangle",
            "mask_text": "purple rectangle",
            "prompt": "Repaint the rectangle as a clean silver metal block.",
            "reason": "existing object local edit path",
        },
        {
            "case_id": "layout_regenerate_no_edit",
            "repair_kind": "layout_regenerate",
            "bbox": [0, 0, 512, 512],
            "target_object": "left right spatial relation",
            "prompt": "Regenerate because left/right layout is reversed.",
            "reason": "complex layout should not waste local editing",
        },
        {
            "case_id": "count_rerank_no_edit",
            "repair_kind": "count_rerank",
            "bbox": [0, 0, 512, 512],
            "target_object": "exact count",
            "prompt": "Rerank candidates because the exact object count is wrong.",
            "reason": "count failures should route to rerank/regeneration",
        },
    ]


def _write_mock_visual(
    source_image: Path,
    case: dict[str, Any],
    case_dir: Path,
    result: dict[str, Any],
) -> None:
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    x, y, w, h = [int(value) for value in case["bbox"]]
    color = (220, 30, 30, 150) if case["repair_kind"] == "bbox_shape_inpaint" else (210, 210, 210, 170)
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 255, 255, 255), width=4)
    draw.text((x + 8, y + 8), case["repair_kind"], fill=(0, 0, 0, 255), font=_font(18))
    visual_path = case_dir / "mock_visual_edit.png"
    image.save(visual_path)
    result["edited_image"] = str(visual_path)
    result["mock_visual_note"] = "Visualization only; actual mock inpaint artifact is recorded under result.edited_image."
    write_json(case_dir / "efficient_repair_result.json", result)


def _write_result_grid(output_dir: Path, results: list[dict[str, Any]]) -> Path | None:
    panels: list[Image.Image] = []
    for result in results:
        image_path = result.get("edited_image")
        if not image_path or not Path(str(image_path)).exists():
            image_path = result.get("contact_sheet")
        if not image_path or not Path(str(image_path)).exists():
            panels.append(_status_panel(result))
            continue
        try:
            panel = Image.open(str(image_path)).convert("RGB")
        except Exception:
            panel = _status_panel(result)
        panels.append(_label_panel(panel, result))
    if not panels:
        return None
    cell_w, cell_h = 360, 300
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), (245, 245, 245))
    for index, panel in enumerate(panels):
        panel.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = (index % cols) * cell_w + (cell_w - panel.width) // 2
        y = (index // cols) * cell_h + (cell_h - panel.height) // 2
        grid.paste(panel, (x, y))
    grid_path = output_dir / "branch_suite_grid.jpg"
    grid.save(grid_path, quality=92)
    return grid_path


def _label_panel(image: Image.Image, result: dict[str, Any]) -> Image.Image:
    label_h = 58
    panel = Image.new("RGB", (image.width, image.height + label_h), (245, 245, 245))
    panel.paste(image, (0, label_h))
    draw = ImageDraw.Draw(panel)
    status = "OK" if result.get("ok") else "NO_EDIT"
    label = f"{result['case_id']} | {status} | {result.get('route')}"
    draw.text((8, 8), label[:90], fill=(20, 20, 20), font=_font(18))
    flags = f"gpu={result.get('gpu_used')} sam2={result.get('sam2_used')} pp={result.get('powerpaint_used')}"
    draw.text((8, 32), flags, fill=(70, 70, 70), font=_font(15))
    return panel


def _status_panel(result: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (360, 240), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    draw.text((18, 28), result["case_id"], fill=(20, 20, 20), font=_font(22))
    draw.text((18, 64), str(result.get("route")), fill=(70, 70, 70), font=_font(18))
    draw.text((18, 96), str(result.get("reason") or result.get("error") or "")[:70], fill=(80, 80, 80), font=_font(15))
    return image


def _nested_edited_image(result: dict[str, Any]) -> str | None:
    nested = result.get("result")
    if isinstance(nested, dict):
        value = nested.get("edited_image")
        return str(value) if value else None
    return None


def _nested_mask_path(result: dict[str, Any]) -> str | None:
    nested = result.get("result")
    if isinstance(nested, dict):
        value = nested.get("mask_path")
        return str(value) if value else None
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
