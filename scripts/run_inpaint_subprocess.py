from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a diffusers inpaint edit in a subprocess.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pipeline", choices=["sd15", "sdxl"], default="sd15")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    pipeline_cls = (
        StableDiffusionXLInpaintPipeline
        if args.pipeline == "sdxl"
        else StableDiffusionInpaintPipeline
    )
    pipe = pipeline_cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=bool(args.local_files_only),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(int(args.seed))
    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image": image,
        "mask_image": mask,
        "num_inference_steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "strength": float(args.strength),
        "generator": generator,
    }
    result = pipe(**kwargs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output)

    metadata = {
        "model_path": args.model_path,
        "pipeline": args.pipeline,
        "image": args.image,
        "mask": args.mask,
        "output": str(output),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "device": args.device,
        "dtype": args.dtype,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
        "image_size": [image.size[0], image.size[1]],
    }
    if args.metadata_output:
        Path(args.metadata_output).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
