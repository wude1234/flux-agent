from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project/.conda-m0/bin/python"
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT
    / "runs_mini_benchmark"
    / "real-flux-mgrag-exact-fail8-512-30-gpu0-r2"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs_edit_smoke" / "fail6_editing_agent"


FAIL6_CASES: list[dict[str, Any]] = [
    {
        "case_id": "holdout_spatial_001",
        "category": "spatial_layout",
        "editable_risk": "low",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-holdout_spatial_001/images/img_0000.jpg",
        "target_object": "yellow pyramid and red cylinder spatial relation",
        "mask_text": "yellow pyramid and red cylinder",
        "bbox": [90, 90, 340, 340],
        "prompt": (
            "Repair the local layout so a yellow pyramid is clearly to the right of a red cylinder, "
            "and the red cylinder is clearly above a blue cube. Keep all three simple geometric objects visible."
        ),
        "negative_prompt": "extra object, missing cube, merged objects, text, realistic clutter",
        "reason": "spatial relation was reversed/collapsed; this is low-editability and may need regeneration",
    },
    {
        "case_id": "holdout_color_002",
        "category": "multi_object_color_material",
        "editable_risk": "medium",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-holdout_color_002/images/img_0000.jpg",
        "target_object": "wooden chair glass lamp paper fan on black rug",
        "mask_text": "chair lamp fan rug",
        "bbox": [35, 105, 445, 325],
        "prompt": (
            "Correct the object attributes inside this region: one turquoise wooden chair, "
            "one crimson glass lamp, and one silver paper fan sitting on a black rug. "
            "Avoid duplicating the chair and keep the fan on the rug."
        ),
        "negative_prompt": "extra chair, duplicated chair, missing fan, off rug, wrong colors, text",
        "reason": "multi-object rare color/material binding failure",
    },
    {
        "case_id": "compact_dev_single_spatial_001",
        "category": "single_spatial",
        "editable_risk": "low",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-compact_dev_single_spatial_001/images/img_0000.jpg",
        "target_object": "yellow pyramid right of red cylinder",
        "mask_text": "yellow pyramid and red cylinder",
        "bbox": [80, 105, 350, 315],
        "prompt": (
            "Move or redraw only these two simple objects so the yellow pyramid is clearly on the right "
            "and the red cylinder is clearly on the left. Only these two objects should be visible."
        ),
        "negative_prompt": "extra object, cube, text, merged object, wrong direction",
        "reason": "direction relation reversed; likely better handled by layout-guided regeneration",
    },
    {
        "case_id": "compact_dev_single_occlusion_002",
        "category": "occlusion",
        "editable_risk": "high",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-compact_dev_single_occlusion_002/images/img_0000.jpg",
        "target_object": "lower half of green suitcase",
        "mask_text": "green suitcase",
        "bbox": [120, 250, 270, 210],
        "prompt": (
            "Add a flat opaque red screen in front of the lower half of the green suitcase. "
            "The suitcase handle must remain clearly visible above the screen."
        ),
        "negative_prompt": "new suitcase, changed handle, hidden handle, text, extra luggage, brown panel",
        "reason": "occlusion insertion is a good local-edit target",
    },
    {
        "case_id": "compact_dev_scene_001",
        "category": "count_spatial_multi_constraint",
        "editable_risk": "medium",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-compact_dev_scene_001/images/img_0000.jpg",
        "target_object": "two cyan mugs left of orange tray and purple spoon under tray",
        "mask_text": "mugs tray spoon",
        "bbox": [50, 120, 420, 330],
        "prompt": (
            "Repair the tabletop objects so exactly two cyan ceramic mugs are to the left of one orange wooden tray, "
            "and one purple spoon lies under the tray. Do not add extra mugs or forks."
        ),
        "negative_prompt": "extra mug, extra fork, missing spoon, wrong side, spoon above tray, wrong color",
        "reason": "mixed count/spatial/object attribute failure; partial edit may help but regeneration may be needed",
    },
    {
        "case_id": "compact_dev_scene_003",
        "category": "symbol_text_color",
        "editable_risk": "high",
        "image_rel": "real-flux-mgrag-exact-fail8-512-30-gpu0-r2-compact_dev_scene_003/images/img_0000.jpg",
        "target_object": "top sign with NO text",
        "mask_text": "top sign",
        "bbox": [105, 45, 305, 210],
        "prompt": (
            "Repaint only the upper sign as a black sign displaying the exact yellow text 'NO'. "
            "Keep the lower sign plain blue with no text and keep the pink ball to the right."
        ),
        "negative_prompt": "blue upper sign, missing NO text, wrong text, text on lower sign, changed ball",
        "reason": "localized sign color/text repair is a good edit target, although exact text can still be hard",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the editing agent on the six recent failed cases.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--grounded-sam2-cuda-visible-devices", default=None)
    parser.add_argument("--mask-mode", choices=["auto", "grounded-sam2", "bbox"], default="auto")
    parser.add_argument("--dilation-kernel-size", type=int, default=31)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--editor", choices=["powerpaint-subprocess", "mock"], default="powerpaint-subprocess")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [
        case for case in FAIL6_CASES if not args.case_id or case["case_id"] in set(args.case_id)
    ]
    manifest = {
        "cases": selected_cases,
        "command_defaults": {
            "mask_mode": args.mask_mode,
            "dilation_kernel_size": args.dilation_kernel_size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "editor": args.editor,
        },
    }
    (output_dir / "editing_batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results = []
    for index, case in enumerate(selected_cases, start=1):
        case_dir = output_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = (args.runs_root / case["image_rel"]).resolve()
        case_payload = {**case, "image_path": str(image_path), "output_dir": str(case_dir)}
        (case_dir / "editing_instruction.json").write_text(
            json.dumps(case_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            args.python,
            "scripts/run_editing_agent_smoke.py",
            "--image",
            str(image_path),
            "--output-dir",
            str(case_dir),
            "--target-object",
            case["target_object"],
            "--prompt",
            case["prompt"],
            "--negative-prompt",
            case["negative_prompt"],
            "--bbox",
            *[str(value) for value in case["bbox"]],
            "--mask-text",
            case["mask_text"],
            "--mask-mode",
            args.mask_mode,
            "--dilation-kernel-size",
            str(args.dilation_kernel_size),
            "--editor",
            args.editor,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--strength",
            str(args.strength),
            "--powerpaint-timeout-seconds",
            str(args.timeout_seconds),
        ]
        if args.grounded_sam2_cuda_visible_devices is not None:
            command.extend(
                ["--grounded-sam2-cuda-visible-devices", args.grounded_sam2_cuda_visible_devices]
            )
        print(f"[{index}/{len(selected_cases)}] {case['case_id']} {case['category']}")
        print(" ".join(command))
        if args.dry_run:
            results.append({"case_id": case["case_id"], "skipped": True, "command": command})
            continue
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds + 300,
        )
        result = {
            "case_id": case["case_id"],
            "returncode": completed.returncode,
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
            "result_json": str(case_dir / "editing_agent_result.json"),
            "contact_sheet": str(case_dir / "before_rawmask_dilatedmask_after.jpg"),
        }
        results.append(result)
        (case_dir / "subprocess_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            print(completed.stderr[-1200:], file=sys.stderr)
    summary = {"output_dir": str(output_dir), "results": results}
    (output_dir / "editing_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_contact_sheet_grid(output_dir, selected_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item.get("returncode", 0) == 0 for item in results) else 1


def _write_contact_sheet_grid(output_dir: Path, cases: list[dict[str, Any]]) -> Path | None:
    panels = []
    for case in cases:
        sheet_path = output_dir / case["case_id"] / "before_rawmask_dilatedmask_after.jpg"
        if sheet_path.exists():
            panels.append((case["case_id"], sheet_path))
    if not panels:
        return None
    thumbs = []
    thumb_w = 1024
    for case_id, path in panels:
        image = Image.open(path).convert("RGB")
        ratio = thumb_w / image.width
        thumb = image.resize((thumb_w, int(image.height * ratio)))
        label_h = 28
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (245, 245, 245))
        canvas.paste(thumb, (0, label_h))
        ImageDraw.Draw(canvas).text((8, 7), case_id, fill=(20, 20, 20))
        thumbs.append(canvas)
    width = max(item.width for item in thumbs)
    height = sum(item.height for item in thumbs)
    grid = Image.new("RGB", (width, height), (245, 245, 245))
    y = 0
    for thumb in thumbs:
        grid.paste(thumb, (0, y))
        y += thumb.height
    output_path = output_dir / "fail6_editing_contact_sheets.jpg"
    grid.save(output_path, quality=92)
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
