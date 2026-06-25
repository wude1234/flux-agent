"""Standalone runner for M5.3 local inpainting/editing experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .local_editor import (
    ColorRecolorEditor,
    DiffusersInpaintEditor,
    InpaintEditor,
    MockInpaintEditor,
    detect_color_region_from_layout,
    plan_inpaint_region_from_layout,
)
from .logging_utils import DEFAULT_RUNS_DIR, create_run_dir, write_json


DEFAULT_SD15_INPAINT_PATH = "/mnt/hdd2/lwt/huggingface/runwayml/stable-diffusion-inpainting"
DEFAULT_EDIT_PROMPT = (
    "paint only the umbrella canopy vivid cobalt blue, keep the small red robot "
    "unchanged, keep the robot hand gripping the dark umbrella handle, rainy street photo"
)
DEFAULT_NEGATIVE_PROMPT = (
    "red umbrella, red canopy, wrong umbrella color, blue robot, changed robot, "
    "extra umbrella, floating umbrella, hidden handle, missing grip"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M5.3 local inpainting on an existing image")
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Existing M4/M5 run directory containing run.json and layout.json.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Image to edit. Defaults to the last selected image in source run.json.",
    )
    parser.add_argument(
        "--layout-path",
        type=Path,
        default=None,
        help="Layout JSON. Defaults to <source-run-dir>/layout.json.",
    )
    parser.add_argument("--target-object", default="umbrella")
    parser.add_argument("--edit-prompt", default=DEFAULT_EDIT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--expand", type=float, default=0.12)
    parser.add_argument(
        "--manual-bbox",
        default=None,
        help="Override layout target with x,y,width,height coordinates.",
    )
    parser.add_argument(
        "--manual-bbox-size",
        choices=["layout", "image"],
        default="image",
        help="Coordinate system for --manual-bbox. Image coordinates use the selected image size.",
    )
    parser.add_argument(
        "--auto-color-bbox",
        action="store_true",
        help=(
            "Detect the target object/part edit mask near the layout box. "
            "Color components are kept as diagnostics/fallback."
        ),
    )
    parser.add_argument(
        "--auto-target-mask",
        action="store_true",
        dest="auto_color_bbox",
        help="Alias for --auto-color-bbox with the object/part-first behavior.",
    )
    parser.add_argument(
        "--target-region",
        choices=["full", "object", "upper", "upper_half", "canopy", "lower", "lower_half"],
        default="full",
        help="Object part to edit when --auto-color-bbox is used.",
    )
    parser.add_argument(
        "--subtract-other-objects",
        action="store_true",
        help="Remove other layout objects from the local edit mask.",
    )
    parser.add_argument(
        "--disable-object-region-mask",
        action="store_true",
        help="Use legacy source-color component selection instead of object/part-first masking.",
    )
    parser.add_argument("--auto-search-expand", type=float, default=0.65)
    parser.add_argument("--auto-component-padding", type=int, default=8)
    parser.add_argument("--auto-min-component-area", type=int, default=128)
    parser.add_argument(
        "--auto-selection-strategy",
        choices=["largest", "layout_overlap"],
        default="largest",
    )
    parser.add_argument(
        "--disable-border-component-reject",
        action="store_true",
        help="Allow auto color bbox detection to select components touching the image border.",
    )

    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--editor", choices=["mock", "sd15-inpaint", "recolor"], default="mock")
    parser.add_argument("--inpaint-model-path", default=DEFAULT_SD15_INPAINT_PATH)
    parser.add_argument("--target-color", default="#1d63d9")
    parser.add_argument(
        "--source-color",
        choices=[
            "red",
            "green",
            "blue",
            "any",
            "low_saturation",
            "transparent",
            "translucent",
            "clear",
            "silver",
            "gray",
            "grey",
            "white",
        ],
        default="red",
    )
    parser.add_argument("--saturation-threshold", type=int, default=70)
    parser.add_argument("--value-threshold", type=int, default=35)
    parser.add_argument(
        "--disable-largest-component",
        action="store_true",
        help="Keep all selected color pixels instead of only the largest connected component.",
    )
    parser.add_argument("--feather-radius", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_run_dir = _resolve_optional_path(args.source_run_dir)
    image_path = _resolve_image_path(args.image_path, source_run_dir)
    layout_path = _resolve_layout_path(args.layout_path, source_run_dir)
    layout_context = _read_json(layout_path)

    run_dir = create_run_dir(args.runs_dir, run_id=args.run_id)
    region, detection = _build_region(args, layout_context, image_path, mask_output_dir=run_dir)
    editor = _build_editor(args, detection=detection)
    result = editor.edit(str(image_path), region, run_dir)

    payload = {
        "mode": "m5.3-local-edit",
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "image_path": str(image_path),
        "layout_path": str(layout_path),
        "target_object": args.target_object,
        "region": region.to_dict(),
        "detection": detection,
        "editor": args.editor,
        "result": result,
        "config": {
            "expand": args.expand,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "inpaint_model_path": args.inpaint_model_path,
            "target_color": args.target_color,
            "source_color": args.source_color,
            "saturation_threshold": args.saturation_threshold,
            "value_threshold": args.value_threshold,
            "keep_largest_component": not args.disable_largest_component,
            "feather_radius": args.feather_radius,
            "auto_color_bbox": args.auto_color_bbox,
            "auto_search_expand": args.auto_search_expand,
            "auto_component_padding": args.auto_component_padding,
            "auto_min_component_area": args.auto_min_component_area,
            "auto_selection_strategy": args.auto_selection_strategy,
            "reject_border_components": not args.disable_border_component_reject,
            "target_region": args.target_region,
            "subtract_other_objects": args.subtract_other_objects,
            "prefer_object_mask": not args.disable_object_region_mask,
        },
    }
    edit_path = write_json(run_dir / "edit.json", payload)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "edit_log": str(edit_path),
                "edited_image": result["edited_image"],
                "mask_path": result["mask_path"],
                "mode": result["mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _build_editor(
    args: argparse.Namespace,
    *,
    detection: Mapping[str, Any] | None = None,
) -> InpaintEditor:
    if args.editor == "sd15-inpaint":
        return DiffusersInpaintEditor(
            model_path=args.inpaint_model_path,
            device=args.device,
            dtype=args.dtype,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            strength=args.strength,
            seed=args.seed,
        )
    if args.editor == "recolor":
        return ColorRecolorEditor(
            target_color=args.target_color,
            source_color=args.source_color,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            keep_largest_component=not args.disable_largest_component,
            feather_radius=args.feather_radius,
            precomputed_mask_path=(
                str(detection.get("precomputed_mask_path"))
                if isinstance(detection, Mapping) and detection.get("precomputed_mask_path")
                else None
            ),
        )
    return MockInpaintEditor()


def _build_region(
    args: argparse.Namespace,
    layout_context: Mapping[str, Any],
    image_path: Path,
    *,
    mask_output_dir: Path | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    if not args.manual_bbox:
        if args.auto_color_bbox:
            return detect_color_region_from_layout(
                image_path,
                layout_context,
                args.target_object,
                prompt=args.edit_prompt,
                negative_prompt=args.negative_prompt,
                source_color=args.source_color,
                saturation_threshold=args.saturation_threshold,
                value_threshold=args.value_threshold,
                search_expand=args.auto_search_expand,
                component_padding=args.auto_component_padding,
                min_component_area=args.auto_min_component_area,
                selection_strategy=args.auto_selection_strategy,
                reject_border_components=not args.disable_border_component_reject,
                target_region=args.target_region,
                subtract_other_objects=args.subtract_other_objects,
                prefer_object_mask=not args.disable_object_region_mask,
                mask_output_dir=mask_output_dir,
            )
        region = plan_inpaint_region_from_layout(
            layout_context,
            args.target_object,
            prompt=args.edit_prompt,
            negative_prompt=args.negative_prompt,
            expand=args.expand,
        )
        return region, None

    from PIL import Image

    from .local_editor import InpaintRegion

    bbox = _parse_bbox(args.manual_bbox)
    if args.manual_bbox_size == "image":
        with Image.open(image_path) as image:
            canvas_size = [int(image.size[0]), int(image.size[1])]
    else:
        canvas_size = _layout_canvas_size(layout_context)
    region = InpaintRegion(
        name=args.target_object,
        bbox=bbox,
        prompt=args.edit_prompt,
        negative_prompt=args.negative_prompt,
        reason=f"manual {args.manual_bbox_size}-coordinate edit bbox",
        canvas_size=canvas_size,
    )
    return region, {"method": "manual_bbox", "manual_bbox_size": args.manual_bbox_size}


def _parse_bbox(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--manual-bbox must be x,y,width,height")
    bbox = [int(part) for part in parts]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("--manual-bbox width and height must be positive")
    return bbox


def _layout_canvas_size(layout_context: Mapping[str, Any]) -> list[int]:
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return [1024, 1024]
    size = layout.get("canvas_size", [1024, 1024])
    if not isinstance(size, list) or len(size) != 2:
        return [1024, 1024]
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return [1024, 1024]
    return [width, height]


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {resolved}")
    return resolved


def _resolve_image_path(image_path: Path | None, source_run_dir: Path | None) -> Path:
    if image_path is not None:
        resolved = image_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"image path does not exist: {resolved}")
        return resolved
    if source_run_dir is None:
        raise ValueError("--image-path is required when --source-run-dir is not provided")
    selected = _last_selected_image(source_run_dir / "run.json")
    if selected:
        resolved = Path(selected).expanduser().resolve()
        if resolved.exists():
            return resolved
    candidates = sorted((source_run_dir / "images").glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"no selectable image found in source run: {source_run_dir}")
    return candidates[-1].resolve()


def _resolve_layout_path(layout_path: Path | None, source_run_dir: Path | None) -> Path:
    if layout_path is not None:
        resolved = layout_path.expanduser().resolve()
    elif source_run_dir is not None:
        resolved = source_run_dir / "layout.json"
    else:
        raise ValueError("--layout-path is required when --source-run-dir is not provided")
    if not resolved.exists():
        raise FileNotFoundError(f"layout path does not exist: {resolved}")
    return resolved.resolve()


def _last_selected_image(run_json_path: Path) -> str | None:
    if not run_json_path.exists():
        return None
    payload = _read_json(run_json_path)
    records = payload.get("round_records", [])
    if not isinstance(records, list):
        return None
    for record in reversed(records):
        if isinstance(record, Mapping) and record.get("selected_image"):
            return str(record["selected_image"])
    return None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
