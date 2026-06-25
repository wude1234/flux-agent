from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_BASE_DIR = Path("runs_edit_smoke/fail6_editing_agent_real_sam_g0_6cases")
DEFAULT_COLOR002_DIR = Path("runs_edit_smoke/fail6_editing_agent_real_sam_g0_color002_polyfix2")
CASE_IDS = [
    "holdout_spatial_001",
    "holdout_color_002",
    "compact_dev_single_spatial_001",
    "compact_dev_single_occlusion_002",
    "compact_dev_scene_001",
    "compact_dev_scene_003",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final contact-sheet grid for the fail6 edit run.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--color002-dir", type=Path, default=DEFAULT_COLOR002_DIR)
    parser.add_argument("--output", type=Path, default=Path("reports/fail6_editing_agent_final_grid.jpg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    panels = []
    for case_id in CASE_IDS:
        root = args.color002_dir if case_id == "holdout_color_002" else args.base_dir
        path = root / case_id / "before_rawmask_dilatedmask_after.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        target_w = 1024
        image = image.resize((target_w, int(image.height * target_w / image.width)))
        label_h = 30
        panel = Image.new("RGB", (image.width, image.height + label_h), (245, 245, 245))
        panel.paste(image, (0, label_h))
        ImageDraw.Draw(panel).text((8, 8), case_id, fill=(20, 20, 20))
        panels.append(panel)
    output = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), (245, 245, 245))
    y = 0
    for panel in panels:
        output.paste(panel, (0, y))
        y += panel.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output, quality=92)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
