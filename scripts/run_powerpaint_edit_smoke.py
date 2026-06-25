from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_editor import InpaintRegion, PowerPaintSubprocessEditor


DEFAULT_POWERPAINT_CHECKPOINT_DIR = "/mnt/ssd1/models/PowerPaint/ppt-v2-1"
DEFAULT_POWERPAINT_PYTHON = "/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python"
DEFAULT_POWERPAINT_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/PowerPaint"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Force one PowerPaint edit on an existing image for backend inspection."
    )
    parser.add_argument("--image", required=True, help="Source image to edit.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Edit bbox in source-image pixel coordinates unless --canvas-size is set.",
    )
    parser.add_argument(
        "--canvas-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("W", "H"),
        help="Coordinate system for --bbox. Defaults to the source image size.",
    )
    parser.add_argument("--name", default="forced edit region")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional precomputed binary mask. If set, bbox is only used for logging/fallback.",
    )
    parser.add_argument("--reason", default="forced PowerPaint backend smoke test")
    parser.add_argument("--task", choices=["text-guided", "object-removal"], default="text-guided")
    parser.add_argument("--powerpaint-python", default=DEFAULT_POWERPAINT_PYTHON)
    parser.add_argument("--powerpaint-dir", default=DEFAULT_POWERPAINT_DIR)
    parser.add_argument("--powerpaint-checkpoint-dir", default=DEFAULT_POWERPAINT_CHECKPOINT_DIR)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "edit_result.json"

    source = Image.open(image_path).convert("RGB")
    canvas_size = list(args.canvas_size or source.size)
    region = InpaintRegion(
        name=args.name,
        bbox=list(args.bbox),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        reason=args.reason,
        canvas_size=canvas_size,
        mask_path=str(Path(args.mask).expanduser().resolve()) if args.mask else None,
    )
    editor = PowerPaintSubprocessEditor(
        checkpoint_dir=args.powerpaint_checkpoint_dir,
        python=args.powerpaint_python,
        powerpaint_dir=args.powerpaint_dir,
        dtype=args.dtype,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        strength=args.strength,
        seed=args.seed,
        prefix="forced_powerpaint",
        task=args.task,
        timeout_seconds=args.timeout_seconds,
        cuda_visible_devices=args.cuda_visible_devices,
        local_files_only=bool(args.local_files_only),
    )

    try:
        result = editor.edit(str(image_path), region, output_dir)
        contact_sheet = _write_contact_sheet(
            source_image=source,
            mask_path=Path(result["mask_path"]),
            edited_image_path=Path(result["edited_image"]),
            output_path=output_dir / "before_mask_after.jpg",
            bbox=list(args.bbox),
        )
        payload: dict[str, Any] = {
            "ok": True,
            "source_image": str(image_path),
            "edited_image": result["edited_image"],
            "mask_path": result["mask_path"],
            "contact_sheet": str(contact_sheet),
            "region": region.to_dict(),
            "edit_result": result,
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "source_image": str(image_path),
            "region": region.to_dict(),
            "error": str(exc),
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


def _write_contact_sheet(
    *,
    source_image: Image.Image,
    mask_path: Path,
    edited_image_path: Path,
    output_path: Path,
    bbox: list[int],
) -> Path:
    source_panel = source_image.copy()
    draw = ImageDraw.Draw(source_panel)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=max(2, source_panel.width // 180))

    mask = Image.open(mask_path).convert("RGB").resize(source_image.size)
    edited = Image.open(edited_image_path).convert("RGB").resize(source_image.size)
    panels = [source_panel, mask, edited]
    label_h = max(28, source_image.height // 18)
    sheet = Image.new(
        "RGB",
        (source_image.width * len(panels), source_image.height + label_h),
        (245, 245, 245),
    )
    labels = ["before + bbox", "mask", "after"]
    for index, panel in enumerate(panels):
        x0 = index * source_image.width
        sheet.paste(panel, (x0, label_h))
        ImageDraw.Draw(sheet).text((x0 + 8, 8), labels[index], fill=(20, 20, 20))
    sheet.save(output_path, quality=92)
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
