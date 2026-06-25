from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from PIL import Image


DEFAULT_GROUNDED_SAM2_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/Grounded_SAM2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Grounded-SAM2 referring segmentation.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grounded-sam2-dir", default=DEFAULT_GROUNDED_SAM2_DIR)
    parser.add_argument(
        "--backend",
        choices=["lightweight", "t2i-copilot"],
        default="lightweight",
        help="lightweight skips T2I-Copilot's supervision visualization dependency.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE so missing weights fail fast.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    grounded_sam2_dir = Path(args.grounded_sam2_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "ok": False,
        "method": "grounded_sam2_referring_expression_segmentation",
        "image": str(image_path),
        "text": args.text,
        "output_dir": str(output_dir),
        "grounded_sam2_dir": str(grounded_sam2_dir),
    }
    try:
        if args.local_files_only:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if not image_path.exists():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        if not grounded_sam2_dir.exists():
            raise FileNotFoundError(f"Grounded-SAM2 dir does not exist: {grounded_sam2_dir}")
        if args.backend == "t2i-copilot":
            mask_path = _run_t2i_copilot_backend(
                image_path=image_path,
                text=args.text,
                output_dir=output_dir,
                grounded_sam2_dir=grounded_sam2_dir,
            )
        else:
            mask_path = _run_lightweight_backend(
                image_path=image_path,
                text=args.text,
                output_dir=output_dir,
                grounded_sam2_dir=grounded_sam2_dir,
            )
        if not mask_path:
            raise RuntimeError("referring_expression_segmentation returned no mask path")
        mask_path = Path(mask_path).expanduser().resolve()
        if not mask_path.exists():
            raise RuntimeError(f"Grounded-SAM2 mask path was not created: {mask_path}")
        payload.update({"ok": True, "mask_path": str(mask_path)})
        (output_dir / "grounded_sam2_mask_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        (output_dir / "grounded_sam2_mask_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1


def _run_t2i_copilot_backend(
    *,
    image_path: Path,
    text: str,
    output_dir: Path,
    grounded_sam2_dir: Path,
) -> Path:
    sys.path.insert(0, str(grounded_sam2_dir))
    os.chdir(grounded_sam2_dir)
    from test_REF import referring_expression_segmentation

    return Path(
        referring_expression_segmentation(
            str(image_path),
            text_input=str(text),
            output_dir=str(output_dir),
        )
    )


def _run_lightweight_backend(
    *,
    image_path: Path,
    text: str,
    output_dir: Path,
    grounded_sam2_dir: Path,
) -> Path:
    """Run Florence-2 referring segmentation + SAM2 without visualization deps."""

    sys.path.insert(0, str(grounded_sam2_dir))
    os.chdir(grounded_sam2_dir)
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = torch.float16 if device == "cuda" else torch.float32
    image = Image.open(image_path).convert("RGB")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        local_files_only=True,
    )
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task_prompt + str(text)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        use_cache=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    result = parsed.get(task_prompt) if isinstance(parsed, dict) else None
    if not isinstance(result, dict) or not result.get("polygons"):
        raise RuntimeError(f"Florence-2 returned no polygons for text={text!r}: {parsed!r}")
    polygons = _normalize_florence_polygons(result["polygons"])
    if not polygons:
        raise RuntimeError(f"Florence-2 returned no valid polygons: {result['polygons']!r}")
    polygon_arrays = [np.asarray(polygon, dtype=np.int32) for polygon in polygons]
    all_points = np.concatenate(polygon_arrays, axis=0)
    x_min = int(np.min(all_points[:, 0]))
    y_min = int(np.min(all_points[:, 1]))
    x_max = int(np.max(all_points[:, 0]))
    y_max = int(np.max(all_points[:, 1]))
    input_boxes = np.array([[x_min, y_min, x_max, y_max]])

    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "./checkpoints/sam2.1_hiera_large.pt",
        device=device,
        current_dir=str(grounded_sam2_dir),
    )
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(np.array(image))
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if masks.size == 0:
        raise RuntimeError("SAM2 returned no masks")
    binary_mask = (masks[0] * 255).astype(np.uint8)
    mask_path = output_dir / f"{image_path.stem}_grounded_sam2_mask.png"
    cv2.imwrite(str(mask_path), binary_mask)
    diagnostics = {
        "backend": "lightweight",
        "florence_polygon_count": len(polygons),
        "florence_polygon_points": polygons,
        "florence_bbox_xyxy": [x_min, y_min, x_max, y_max],
        "sam_score": float(np.asarray(scores).reshape(-1)[0]) if np.asarray(scores).size else None,
        "generated_text": generated_text[-1000:],
    }
    (output_dir / "grounded_sam2_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mask_path


def _normalize_florence_polygons(polygons: Any) -> list[list[list[int]]]:
    """Normalize Florence polygon outputs across single/multi-object formats."""

    normalized: list[list[list[int]]] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            pair_points = _as_point_pairs(item)
            if pair_points and len(pair_points) >= 3:
                normalized.append(pair_points)
                return
            flat_points = _as_flat_polygon(item)
            if flat_points and len(flat_points) >= 3:
                normalized.append(flat_points)
                return
            for child in item:
                visit(child)

    visit(polygons)
    return normalized


def _as_point_pairs(item: Any) -> list[list[int]] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    points: list[list[int]] = []
    for point in item:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        points.append([int(round(x)), int(round(y))])
    return points


def _as_flat_polygon(item: Any) -> list[list[int]] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 6 or len(item) % 2 != 0:
        return None
    if not all(isinstance(value, (int, float)) for value in item):
        return None
    values = [int(round(value)) for value in item]
    return [[values[index], values[index + 1]] for index in range(0, len(values), 2)]


if __name__ == "__main__":
    raise SystemExit(main())
