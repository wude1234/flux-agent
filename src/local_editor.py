"""Local image editing adapters for M5.3 inpainting experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol, Sequence


SOURCE_COLOR_MODES = {
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "any",
    "low_saturation",
}
LOW_SATURATION_ALIASES = {
    "low-saturation",
    "low_saturation",
    "transparent",
    "translucent",
    "clear",
    "silver",
    "gray",
    "grey",
    "white",
}


@dataclass(frozen=True)
class InpaintRegion:
    """A rectangular image region to edit."""

    name: str
    bbox: list[int]
    prompt: str
    negative_prompt: str = ""
    reason: str = ""
    canvas_size: list[int] = field(default_factory=lambda: [1024, 1024])
    mask_path: str | None = None

    def __post_init__(self) -> None:
        _clean_text(self.name, "name")
        _clean_text(self.prompt, "prompt")
        if self.mask_path is not None:
            _clean_text(self.mask_path, "mask_path")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be [x, y, width, height]")
        if any(int(value) < 0 for value in self.bbox):
            raise ValueError("bbox values must be non-negative")
        if int(self.bbox[2]) <= 0 or int(self.bbox[3]) <= 0:
            raise ValueError("bbox width and height must be positive")
        _canvas_size(self.canvas_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "reason": self.reason,
            "canvas_size": list(self.canvas_size),
            "mask_path": self.mask_path,
        }


class InpaintEditor(Protocol):
    """Image editing backend interface."""

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        ...


@dataclass
class MockInpaintEditor:
    """Write deterministic placeholder edit artifacts without loading models."""

    prefix: str = "mock_inpaint"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = str(Path(_clean_text(image_path, "image_path")).resolve())
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"
        output_path = output_dir / f"{self.prefix}_image_{self._counter:04d}.txt"
        image_size = _canvas_size(region.canvas_size)
        write_or_copy_region_mask(
            mask_path,
            source_mask_path=region.mask_path,
            image_size=image_size,
            fallback_bbox=region.bbox,
        )
        output_path.write_text(
            "\n".join(
                [
                    "MOCK_INPAINT_PLACEHOLDER",
                    f"source={image_path}",
                    f"region={region.to_dict()}",
                ]
            ),
            encoding="utf-8",
        )
        record = {
            "image_path": image_path,
            "edited_image": str(output_path),
            "mask_path": str(mask_path),
            "region": region.to_dict(),
            "mode": "mock",
        }
        self.calls.append(record)
        self._counter += 1
        return dict(record)


@dataclass
class DiffusersInpaintEditor:
    """Local SD1.5 inpainting adapter backed by ``diffusers``.

    Imports and model loading are lazy so normal tests do not need diffusers,
    torch, GPU, or model weights. This adapter is intended for local experiments
    with existing weights only.
    """

    model_path: str | Path
    device: str = "cuda"
    dtype: str = "float16"
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    strength: float = 0.85
    seed: int | None = None
    prefix: str = "sd15_inpaint"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0
    _pipe: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"inpaint model path does not exist: {self.model_path}")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if self.strength <= 0 or self.strength > 1:
            raise ValueError("strength must be in (0, 1]")

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = _clean_text(image_path, "image_path")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        source_image = Image.open(image_path).convert("RGB")
        scaled_bbox = scale_bbox(
            region.bbox,
            from_size=_canvas_size(region.canvas_size),
            to_size=source_image.size,
        )
        bbox_mask_path = output_dir / f"{self.prefix}_bbox_mask_{self._counter:04d}.png"
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"
        output_path = output_dir / f"{self.prefix}_image_{self._counter:04d}.png"
        write_bbox_mask(bbox_mask_path, image_size=source_image.size, bbox=scaled_bbox)
        write_or_copy_region_mask(
            mask_path,
            source_mask_path=region.mask_path,
            image_size=source_image.size,
            fallback_bbox=scaled_bbox,
        )
        mask_image = Image.open(mask_path).convert("L")

        pipe = self._load_pipe()
        kwargs: dict[str, Any] = {
            "prompt": region.prompt,
            "image": source_image,
            "mask_image": mask_image,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "generator": self._torch_generator(),
        }
        if region.negative_prompt.strip():
            kwargs["negative_prompt"] = region.negative_prompt.strip()
        result = pipe(**kwargs)
        result.images[0].save(output_path)

        record = {
            "image_path": image_path,
            "edited_image": str(output_path),
            "mask_path": str(mask_path),
            "bbox_mask_path": str(bbox_mask_path),
            "region": region.to_dict(),
            "scaled_bbox": scaled_bbox,
            "source_image_size": [source_image.size[0], source_image.size[1]],
            "mode": "sd15-inpaint",
            "model_path": str(self.model_path),
            "precomputed_mask_path": region.mask_path,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "seed": self.seed,
        }
        self.calls.append(record)
        self._counter += 1
        return dict(record)

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dtype = getattr(torch, self.dtype)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Use --device cpu for a slow smoke test or "
                "run the local inpaint experiment where CUDA is visible."
            )
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self._pipe = pipe
        return self._pipe

    def _torch_generator(self) -> Any:
        if self.seed is None:
            return None
        import torch

        return torch.Generator(device=self.device).manual_seed(int(self.seed))


@dataclass
class SubprocessInpaintEditor:
    """Run diffusers inpaint in a separate Python environment.

    This keeps the main agent environment light and prevents FLUX and inpaint
    models from being loaded into the same process.
    """

    model_path: str | Path
    python: str | Path = "/mnt/ssd1/conda/envs/sdxl/bin/python"
    script_path: str | Path | None = None
    device: str = "cuda"
    dtype: str = "float16"
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    strength: float = 0.85
    seed: int | None = None
    prefix: str = "subprocess_inpaint"
    pipeline: str = "sd15"
    timeout_seconds: int | None = 900
    cuda_visible_devices: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.python = Path(self.python)
        self.script_path = (
            Path(self.script_path)
            if self.script_path is not None
            else Path(__file__).resolve().parents[1] / "scripts" / "run_inpaint_subprocess.py"
        )
        if not self.python.exists():
            raise FileNotFoundError(f"inpaint Python executable does not exist: {self.python}")
        if not self.script_path.exists():
            raise FileNotFoundError(f"inpaint subprocess script does not exist: {self.script_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"inpaint model path does not exist: {self.model_path}")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if self.strength <= 0 or self.strength > 1:
            raise ValueError("strength must be in (0, 1]")
        if self.pipeline not in {"sd15", "sdxl"}:
            raise ValueError("pipeline must be sd15 or sdxl")

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = _clean_text(image_path, "image_path")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        source_image = Image.open(image_path).convert("RGB")
        scaled_bbox = scale_bbox(
            region.bbox,
            from_size=_canvas_size(region.canvas_size),
            to_size=source_image.size,
        )
        bbox_mask_path = output_dir / f"{self.prefix}_bbox_mask_{self._counter:04d}.png"
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"
        output_path = output_dir / f"{self.prefix}_image_{self._counter:04d}.png"
        metadata_path = output_dir / f"{self.prefix}_metadata_{self._counter:04d}.json"
        write_bbox_mask(bbox_mask_path, image_size=source_image.size, bbox=scaled_bbox)
        write_or_copy_region_mask(
            mask_path,
            source_mask_path=region.mask_path,
            image_size=source_image.size,
            fallback_bbox=scaled_bbox,
        )

        command = [
            str(self.python),
            str(self.script_path),
            "--model-path",
            str(self.model_path),
            "--pipeline",
            self.pipeline,
            "--image",
            image_path,
            "--mask",
            str(mask_path.resolve()),
            "--output",
            str(output_path.resolve()),
            "--metadata-output",
            str(metadata_path.resolve()),
            "--prompt",
            region.prompt,
            "--negative-prompt",
            region.negative_prompt,
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--steps",
            str(self.num_inference_steps),
            "--guidance-scale",
            str(self.guidance_scale),
            "--strength",
            str(self.strength),
        ]
        if self.seed is not None:
            command.extend(["--seed", str(int(self.seed) + self._counter)])
        env = None
        if self.cuda_visible_devices:
            import os

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "inpaint subprocess failed with exit code "
                f"{result.returncode}. stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}"
            )
        if not output_path.exists():
            raise RuntimeError(
                "inpaint subprocess finished but output image was not created. "
                f"stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}"
            )
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {"metadata_parse_error": str(metadata_path)}

        record = {
            "image_path": image_path,
            "edited_image": str(output_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "bbox_mask_path": str(bbox_mask_path.resolve()),
            "region": region.to_dict(),
            "scaled_bbox": scaled_bbox,
            "source_image_size": [source_image.size[0], source_image.size[1]],
            "mode": f"{self.pipeline}-subprocess-inpaint",
            "model_path": str(self.model_path),
            "python": str(self.python),
            "script_path": str(self.script_path),
            "command": command,
            "metadata_path": str(metadata_path.resolve()),
            "metadata": metadata,
            "precomputed_mask_path": region.mask_path,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-2000:],
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "seed": self.seed,
        }
        self.calls.append(record)
        self._counter += 1
        return dict(record)


@dataclass
class PowerPaintSubprocessEditor:
    """Run PowerPaint in a dedicated subprocess environment.

    PowerPaint has a tight dependency stack, so the main agent must never import
    it directly. This adapter only prepares a bbox mask and calls the PowerPaint
    CLI in a separate process.
    """

    checkpoint_dir: str | Path
    python: str | Path
    powerpaint_dir: str | Path = (
        "/home/zrr/t2i_agent_papers_2024_2025/"
        "mult-t2i-agent/code/T2I-Copilot-master/models/PowerPaint"
    )
    script_name: str = "test.py"
    device: str = "cuda"
    dtype: str = "float16"
    guidance_scale: float = 7.5
    num_inference_steps: int = 45
    strength: float = 1.0
    seed: int | None = None
    prefix: str = "powerpaint"
    task: str = "text-guided"
    timeout_seconds: int | None = 1800
    cuda_visible_devices: str | None = None
    local_files_only: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.python = Path(self.python)
        self.powerpaint_dir = Path(self.powerpaint_dir)
        if not self.python.exists():
            raise FileNotFoundError(f"PowerPaint Python executable does not exist: {self.python}")
        if not self.powerpaint_dir.exists():
            raise FileNotFoundError(f"PowerPaint directory does not exist: {self.powerpaint_dir}")
        if not (self.powerpaint_dir / self.script_name).exists():
            raise FileNotFoundError(
                f"PowerPaint script does not exist: {self.powerpaint_dir / self.script_name}"
            )
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"PowerPaint checkpoint dir does not exist: {self.checkpoint_dir}")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if self.strength <= 0:
            raise ValueError("strength must be positive")
        if self.task not in {"text-guided", "object-removal"}:
            raise ValueError("task must be text-guided or object-removal")

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = str(Path(_clean_text(image_path, "image_path")).resolve())
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        source_image = Image.open(image_path).convert("RGB")
        scaled_bbox = scale_bbox(
            region.bbox,
            from_size=_canvas_size(region.canvas_size),
            to_size=source_image.size,
        )
        bbox_mask_path = output_dir / f"{self.prefix}_bbox_mask_{self._counter:04d}.png"
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"
        output_path = output_dir / f"{self.prefix}_image_{self._counter:04d}.png"
        write_bbox_mask(bbox_mask_path, image_size=source_image.size, bbox=scaled_bbox)
        write_or_copy_region_mask(
            mask_path,
            source_mask_path=region.mask_path,
            image_size=source_image.size,
            fallback_bbox=scaled_bbox,
        )
        bbox_coordinates = [
            scaled_bbox[0],
            scaled_bbox[1],
            scaled_bbox[0] + scaled_bbox[2],
            scaled_bbox[1] + scaled_bbox[3],
        ]
        command = [
            str(self.python),
            str(self.powerpaint_dir / self.script_name),
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--version",
            "ppt-v2-1",
            "--weight_dtype",
            self.dtype,
            "--input_image",
            image_path,
            "--mask_image",
            str(mask_path.resolve()),
            "--output_path",
            str(output_path.resolve()),
            "--task",
            self.task,
            "--prompt",
            region.prompt,
            "--negative_prompt",
            region.negative_prompt,
            "--fitting_degree",
            str(self.strength),
            "--steps",
            str(self.num_inference_steps),
            "--guidance_scale",
            str(self.guidance_scale),
            "--bbox_coordinates",
            json.dumps(bbox_coordinates),
        ]
        if self.local_files_only:
            command.append("--local_files_only")
        if self.seed is not None:
            command.extend(["--seed", str(int(self.seed) + self._counter)])
        env = None
        if self.cuda_visible_devices:
            import os

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        result = subprocess.run(
            command,
            cwd=str(self.powerpaint_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "PowerPaint subprocess failed with exit code "
                f"{result.returncode}. stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}"
            )
        if not output_path.exists():
            raise RuntimeError(
                "PowerPaint subprocess finished but output image was not created. "
                f"stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}"
            )
        record = {
            "image_path": image_path,
            "edited_image": str(output_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "bbox_mask_path": str(bbox_mask_path.resolve()),
            "region": region.to_dict(),
            "scaled_bbox": scaled_bbox,
            "bbox_coordinates": bbox_coordinates,
            "source_image_size": [source_image.size[0], source_image.size[1]],
            "mode": "powerpaint-subprocess",
            "checkpoint_dir": str(self.checkpoint_dir),
            "python": str(self.python),
            "powerpaint_dir": str(self.powerpaint_dir),
            "command": command,
            "precomputed_mask_path": region.mask_path,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-2000:],
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "fitting_degree": self.strength,
            "task": self.task,
            "seed": self.seed,
        }
        self.calls.append(record)
        self._counter += 1
        return dict(record)


@dataclass
class ColorRecolorEditor:
    """Deterministic local color repair for existing objects inside a bbox.

    This is useful when the object shape is acceptable but cross-object color
    binding failed, for example a requested blue umbrella became red.
    """

    target_color: str = "#1d63d9"
    source_color: str = "red"
    saturation_threshold: int = 70
    value_threshold: int = 35
    keep_largest_component: bool = True
    feather_radius: float = 2.0
    exclude_bboxes: Sequence[Sequence[int]] = ()
    precomputed_mask_path: str | None = None
    dark_value_floor: int | None = None
    prefix: str = "recolor"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self) -> None:
        parse_rgb_color(self.target_color)
        source_color = normalize_source_color_mode(self.source_color)
        object.__setattr__(self, "source_color", source_color)
        if source_color not in SOURCE_COLOR_MODES:
            raise ValueError(
                "source_color must be one of: "
                + ", ".join(sorted(SOURCE_COLOR_MODES))
            )
        if self.saturation_threshold < 0 or self.saturation_threshold > 255:
            raise ValueError("saturation_threshold must be in [0, 255]")
        if self.value_threshold < 0 or self.value_threshold > 255:
            raise ValueError("value_threshold must be in [0, 255]")
        if self.dark_value_floor is not None and (
            self.dark_value_floor < 0 or self.dark_value_floor > 255
        ):
            raise ValueError("dark_value_floor must be in [0, 255]")

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = _clean_text(image_path, "image_path")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        source_image = Image.open(image_path).convert("RGB")
        scaled_bbox = scale_bbox(
            region.bbox,
            from_size=_canvas_size(region.canvas_size),
            to_size=source_image.size,
        )
        bbox_mask_path = output_dir / f"{self.prefix}_bbox_mask_{self._counter:04d}.png"
        selected_mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"
        output_path = output_dir / f"{self.prefix}_image_{self._counter:04d}.png"

        write_bbox_mask(bbox_mask_path, image_size=source_image.size, bbox=scaled_bbox)
        if self.precomputed_mask_path:
            selected_mask = load_binary_mask(
                self.precomputed_mask_path,
                image_size=source_image.size,
            )
        else:
            selected_mask, selected_count = build_color_selection_mask(
                source_image,
                bbox=scaled_bbox,
                source_color=self.source_color,
                saturation_threshold=self.saturation_threshold,
                value_threshold=self.value_threshold,
            )
            if self.keep_largest_component:
                selected_mask, selected_count = keep_largest_mask_component(selected_mask)
        if self.exclude_bboxes:
            selected_mask = subtract_bboxes_from_mask(selected_mask, self.exclude_bboxes)
        selected_count = count_mask_pixels(selected_mask)
        selected_mask.save(selected_mask_path)
        edited_image = recolor_image_with_mask(
            source_image,
            selected_mask,
            target_color=parse_rgb_color(self.target_color),
            feather_radius=self.feather_radius,
            value_floor=(
                self.dark_value_floor
                if self.dark_value_floor is not None
                else (96 if self.source_color == "black" else None)
            ),
        )
        edited_image.save(output_path)

        record = {
            "image_path": image_path,
            "edited_image": str(output_path),
            "mask_path": str(selected_mask_path),
            "bbox_mask_path": str(bbox_mask_path),
            "region": region.to_dict(),
            "scaled_bbox": scaled_bbox,
            "source_image_size": [source_image.size[0], source_image.size[1]],
            "mode": "recolor",
            "target_color": self.target_color,
            "source_color": self.source_color,
            "precomputed_mask_path": self.precomputed_mask_path,
            "selected_pixel_count": selected_count,
            "saturation_threshold": self.saturation_threshold,
            "value_threshold": self.value_threshold,
            "keep_largest_component": self.keep_largest_component,
            "feather_radius": self.feather_radius,
            "dark_value_floor": (
                self.dark_value_floor
                if self.dark_value_floor is not None
                else (96 if self.source_color == "black" else None)
            ),
            "exclude_bboxes": [list(bbox) for bbox in self.exclude_bboxes],
        }
        self.calls.append(record)
        self._counter += 1
        return dict(record)


def plan_inpaint_region_from_layout(
    layout_context: Mapping[str, Any],
    target_name: str,
    *,
    prompt: str,
    negative_prompt: str = "",
    expand: float = 0.08,
) -> InpaintRegion:
    """Find a target object in layout JSON and return an expanded inpaint region."""

    target = target_name.strip().lower()
    if not target:
        raise ValueError("target_name must not be empty")
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        raise TypeError("layout_context must contain a layout mapping")
    canvas_size = _canvas_size(layout.get("canvas_size", [1024, 1024]))
    objects = layout.get("objects", [])
    if not isinstance(objects, Sequence):
        raise ValueError("layout objects must be a sequence")
    matched: Mapping[str, Any] | None = None
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        name = str(obj.get("name", "")).lower()
        if target in name or name in target:
            matched = obj
            break
    if matched is None:
        raise ValueError(f"target object not found in layout: {target_name}")
    bbox = expand_bbox(
        [int(value) for value in matched.get("bbox", [])],
        image_size=canvas_size,
        expand=expand,
    )
    return InpaintRegion(
        name=str(matched.get("name", target_name)),
        bbox=bbox,
        prompt=_clean_text(prompt, "prompt"),
        negative_prompt=str(negative_prompt or ""),
        reason=f"local edit target from layout object: {matched.get('name', target_name)}",
        canvas_size=[canvas_size[0], canvas_size[1]],
    )


def detect_color_region_from_layout(
    image_path: str | Path,
    layout_context: Mapping[str, Any],
    target_name: str,
    *,
    prompt: str,
    negative_prompt: str = "",
    source_color: str = "red",
    saturation_threshold: int = 55,
    value_threshold: int = 40,
    search_expand: float = 0.65,
    component_padding: int = 8,
    min_component_area: int = 128,
    selection_strategy: str = "largest",
    reject_border_components: bool = True,
    target_region: str = "full",
    subtract_target_names: Sequence[str] = (),
    subtract_other_objects: bool = False,
    prefer_object_mask: bool = True,
    image_grounded_bbox: bool = False,
    mask_output_dir: str | Path | None = None,
) -> tuple[InpaintRegion, dict[str, Any]]:
    """Detect an image-grounded target object/part region for recoloring.

    The function keeps its historical name for compatibility, but the default
    policy is object-first: locate the target object or part from the
    layout/VLM bbox, then use color components only as diagnostics or an
    explicit fallback.
    """

    from PIL import Image

    source_color = normalize_source_color_mode(source_color)
    base_region = plan_inpaint_region_from_layout(
        layout_context,
        target_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        expand=0.0,
    )
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    layout_bbox = scale_bbox(
        base_region.bbox,
        from_size=_canvas_size(base_region.canvas_size),
        to_size=image_size,
    )
    constrained_bbox = (
        expand_bbox(layout_bbox, image_size=image_size, expand=0.0)
        if image_grounded_bbox
        else _target_region_bbox(
            layout_bbox,
            image_size=image_size,
            target_region=target_region,
        )
    )
    subtract_bboxes = _subtract_bboxes_from_layout(
        layout_context,
        subtract_target_names,
        from_size=_canvas_size(base_region.canvas_size),
        to_size=image_size,
    )
    if subtract_other_objects:
        subtract_bboxes.extend(
            _other_object_bboxes_from_layout(
                layout_context,
                target_name=base_region.name,
                from_size=_canvas_size(base_region.canvas_size),
                to_size=image_size,
            )
        )
        subtract_bboxes = _dedupe_bboxes(subtract_bboxes)
    effective_search_expand = (
        min(float(search_expand), 0.05)
        if source_color == "low_saturation"
        else float(search_expand)
    )
    effective_saturation_threshold = (
        max(int(saturation_threshold), 80)
        if source_color == "low_saturation"
        else int(saturation_threshold)
    )
    search_bbox = expand_bbox(
        constrained_bbox,
        image_size=image_size,
        expand=effective_search_expand,
    )
    color_mask, selected_pixel_count = build_color_selection_mask(
        image,
        bbox=search_bbox,
        source_color=source_color,
        saturation_threshold=effective_saturation_threshold,
        value_threshold=value_threshold,
    )
    if subtract_bboxes:
        color_mask = subtract_bboxes_from_mask(color_mask, subtract_bboxes)
    object_mask = build_object_region_mask(
        image,
        bbox=constrained_bbox,
        target_region=target_region,
        target_name=base_region.name,
        subtract_bboxes=subtract_bboxes,
        source_color=source_color,
        saturation_threshold=effective_saturation_threshold,
        value_threshold=value_threshold,
    )
    object_mask_count = count_mask_pixels(object_mask)
    object_components = mask_components(
        object_mask,
        min_area=max(1, min_component_area // 2),
    )
    object_selected = (
        select_component_for_bbox(
            object_components,
            preferred_bbox=constrained_bbox,
            strategy="layout_overlap",
        )
        if object_components
        else {
            "bbox": constrained_bbox,
            "area": object_mask_count,
            "center": list(_bbox_center(constrained_bbox)),
            "touches_image_border": False,
            "score": 0.0,
            "selection_strategy": "object_region",
        }
    )
    raw_components = mask_components(color_mask, min_area=min_component_area)
    components = [
        component
        for component in raw_components
        if not reject_border_components or not component.get("touches_image_border")
    ]
    if not components:
        components = raw_components
    if not components and not prefer_object_mask:
        raise ValueError(
            f"no {source_color} color component found near layout target: {target_name}"
        )
    color_selected = (
        select_component_for_bbox(
            components,
            preferred_bbox=constrained_bbox,
            strategy=selection_strategy,
        )
        if components
        else dict(object_selected)
    )
    color_selected_overlap_area = _bbox_overlap_area(color_selected["bbox"], constrained_bbox)
    color_selected_overlap_ratio = color_selected_overlap_area / max(
        1, _bbox_area(color_selected["bbox"])
    )
    color_geometry = validate_component_geometry(
        color_selected["bbox"],
        constrained_bbox=constrained_bbox,
        target_region=target_region,
        target_name=base_region.name,
    )
    use_object_mask = prefer_object_mask and _should_use_object_region_mask(
        source_color=source_color,
        color_selected=color_selected,
        color_geometry=color_geometry,
        constrained_bbox=constrained_bbox,
        object_mask_count=object_mask_count,
    )
    selected = dict(object_selected) if use_object_mask else dict(color_selected)
    geometry = validate_component_geometry(
        selected["bbox"],
        constrained_bbox=constrained_bbox,
        target_region=target_region,
        target_name=base_region.name,
    )
    selected_overlap_area = _bbox_overlap_area(selected["bbox"], constrained_bbox)
    selected_overlap_ratio = selected_overlap_area / max(1, _bbox_area(selected["bbox"]))
    detected_bbox = pad_bbox(
        selected["bbox"],
        image_size=image_size,
        padding=int(component_padding),
    )
    region = InpaintRegion(
        name=base_region.name,
        bbox=detected_bbox,
        prompt=base_region.prompt,
        negative_prompt=base_region.negative_prompt,
        reason=(
            f"object-region mask for {base_region.name}"
            if use_object_mask
            else f"auto color-component bbox for {source_color} target near layout object"
        ),
        canvas_size=[image_size[0], image_size[1]],
    )
    selected_mask_path = None
    if mask_output_dir is not None:
        mask_dir = Path(mask_output_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        selected_mask_path = mask_dir / "recolor_object_region_mask.png"
        (object_mask if use_object_mask else color_mask).save(selected_mask_path)
    detection = {
        "method": "object_region_near_layout" if use_object_mask else "color_component_near_layout",
        "mask_mode": "object_region" if use_object_mask else "color_component",
        "precomputed_mask_path": str(selected_mask_path) if selected_mask_path else None,
        "source_color": source_color,
        "layout_bbox_scaled": layout_bbox,
        "constrained_bbox": constrained_bbox,
        "search_bbox": search_bbox,
        "search_expand": search_expand,
        "effective_search_expand": effective_search_expand,
        "target_region": target_region,
        "subtract_bboxes": subtract_bboxes,
        "subtract_target_names": list(subtract_target_names),
        "subtract_other_objects": subtract_other_objects,
        "image_grounded_bbox": bool(image_grounded_bbox),
        "selected_pixel_count": selected_pixel_count,
        "raw_component_count": len(raw_components),
        "component_count": len(components),
        "reject_border_components": reject_border_components,
        "components": components[:8],
        "object_mask_pixel_count": object_mask_count,
        "object_components": object_components[:8],
        "object_selected_component": dict(object_selected),
        "color_selected_component": dict(color_selected),
        "color_selected_overlap_area": color_selected_overlap_area,
        "color_selected_overlap_ratio": round(color_selected_overlap_ratio, 6),
        "color_geometry": color_geometry,
        "selected_component": dict(selected),
        "selected_overlap_area": selected_overlap_area,
        "selected_overlap_ratio": round(selected_overlap_ratio, 6),
        "geometry": geometry,
        "geometry_failures": list(geometry["failures"]),
        "detected_bbox": detected_bbox,
        "component_padding": component_padding,
        "min_component_area": min_component_area,
        "selection_strategy": selection_strategy,
        "saturation_threshold": saturation_threshold,
        "effective_saturation_threshold": effective_saturation_threshold,
        "value_threshold": value_threshold,
    }
    return region, detection


def validate_component_geometry(
    component_bbox: Sequence[int],
    *,
    constrained_bbox: Sequence[int],
    target_region: str = "full",
    target_name: str = "",
) -> dict[str, Any]:
    """Return generic shape/layout diagnostics for a candidate repair component."""

    component_bbox = [int(value) for value in component_bbox]
    constrained_bbox = [int(value) for value in constrained_bbox]
    comp_area = max(1, _bbox_area(component_bbox))
    target_area = max(1, _bbox_area(constrained_bbox))
    overlap_area = _bbox_overlap_area(component_bbox, constrained_bbox)
    comp_x, comp_y, comp_w, comp_h = component_bbox
    target_x, target_y, target_w, target_h = constrained_bbox
    comp_aspect = comp_w / max(1, comp_h)
    target_aspect = target_w / max(1, target_h)
    overlap_component_ratio = overlap_area / comp_area
    overlap_target_ratio = overlap_area / target_area
    center_x, center_y = _bbox_center(component_bbox)
    target_center_x, target_center_y = _bbox_center(constrained_bbox)
    center_offset_x = abs(center_x - target_center_x) / max(1.0, target_w)
    center_offset_y = abs(center_y - target_center_y) / max(1.0, target_h)
    horizontal_target = _expects_horizontal_component(target_region, target_name)
    failures: list[dict[str, Any]] = []
    if horizontal_target and comp_aspect < 0.75:
        failures.append(
            {
                "type": "tall_component_for_horizontal_target",
                "value": round(comp_aspect, 6),
                "threshold": 0.75,
                "message": "candidate component is too tall/narrow for a horizontal target region",
            }
        )
    if horizontal_target and comp_h > max(target_h * 1.65, target_w * 0.9):
        failures.append(
            {
                "type": "component_height_exceeds_horizontal_target",
                "value": int(comp_h),
                "threshold": round(max(target_h * 1.65, target_w * 0.9), 3),
                "message": "candidate component extends far beyond the target region height",
            }
        )
    if overlap_target_ratio < 0.08:
        failures.append(
            {
                "type": "low_target_region_coverage",
                "value": round(overlap_target_ratio, 6),
                "threshold": 0.08,
                "message": "candidate covers too little of the target layout region",
            }
        )
    if center_offset_x > 0.95 or center_offset_y > 1.25:
        failures.append(
            {
                "type": "component_center_far_from_target",
                "value": [round(center_offset_x, 6), round(center_offset_y, 6)],
                "threshold": [0.95, 1.25],
                "message": "candidate component center is too far from the target layout center",
            }
        )
    return {
        "component_bbox": component_bbox,
        "constrained_bbox": constrained_bbox,
        "target_region": str(target_region or "full"),
        "target_name": str(target_name or ""),
        "horizontal_target": horizontal_target,
        "component_aspect": round(comp_aspect, 6),
        "target_aspect": round(target_aspect, 6),
        "overlap_area": overlap_area,
        "overlap_component_ratio": round(overlap_component_ratio, 6),
        "overlap_target_ratio": round(overlap_target_ratio, 6),
        "center_offset": [round(center_offset_x, 6), round(center_offset_y, 6)],
        "component_area_ratio": round(comp_area / target_area, 6),
        "failures": failures,
    }


def _should_use_object_region_mask(
    *,
    source_color: str,
    color_selected: Mapping[str, Any],
    color_geometry: Mapping[str, Any],
    constrained_bbox: Sequence[int],
    object_mask_count: int,
) -> bool:
    """Return whether recolor should use the object/part mask over color blobs."""

    if object_mask_count <= 0:
        return False
    # The default local-repair contract is target-object first. Color blobs are
    # still logged for coverage checks, but they should not decide where a mask
    # goes when the target object/part region is available.
    return True


def expand_bbox(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    expand: float = 0.08,
) -> list[int]:
    """Expand a bbox by a fraction while keeping it inside image bounds."""

    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    width, height = image_size
    x, y, box_w, box_h = [int(value) for value in bbox]
    if box_w <= 0 or box_h <= 0:
        raise ValueError("bbox width and height must be positive")
    pad_x = int(round(box_w * float(expand)))
    pad_y = int(round(box_h * float(expand)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + box_w + pad_x)
    y1 = min(height, y + box_h + pad_y)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def pad_bbox(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    padding: int,
) -> list[int]:
    """Pad a bbox by a fixed number of pixels inside image bounds."""

    if padding < 0:
        raise ValueError("padding must be non-negative")
    x, y, box_w, box_h = [int(value) for value in bbox]
    if padding == 0:
        return expand_bbox([x, y, box_w, box_h], image_size=image_size, expand=0.0)
    return _pad_bbox_pixels([x, y, box_w, box_h], image_size=image_size, padding=padding)


def scale_bbox(
    bbox: Sequence[int],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[int]:
    """Scale a bbox from layout canvas coordinates to image coordinates."""

    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    from_width, from_height = _canvas_size(from_size)
    to_width, to_height = _canvas_size(to_size)
    x, y, box_w, box_h = [int(value) for value in bbox]
    if box_w <= 0 or box_h <= 0:
        raise ValueError("bbox width and height must be positive")
    x0 = int(round(x * to_width / from_width))
    y0 = int(round(y * to_height / from_height))
    x1 = int(round((x + box_w) * to_width / from_width))
    y1 = int(round((y + box_h) * to_height / from_height))
    x0 = min(max(0, x0), to_width - 1)
    y0 = min(max(0, y0), to_height - 1)
    x1 = min(max(x0 + 1, x1), to_width)
    y1 = min(max(y0 + 1, y1), to_height)
    return [x0, y0, x1 - x0, y1 - y0]


def write_bbox_mask(
    path: str | Path,
    *,
    image_size: tuple[int, int],
    bbox: Sequence[int],
) -> Path:
    """Write a white rectangular mask on black background."""

    from PIL import Image, ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    x, y, box_w, box_h = [int(value) for value in bbox]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + box_w - 1, y + box_h - 1], fill=255)
    mask.save(path)
    return path


def write_or_copy_region_mask(
    path: str | Path,
    *,
    source_mask_path: str | Path | None,
    image_size: tuple[int, int],
    fallback_bbox: Sequence[int],
) -> Path:
    """Write the edit mask, preferring a precomputed SAM/VLM mask."""

    if source_mask_path:
        mask = load_binary_mask(source_mask_path, image_size=image_size)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(path)
        return path
    return write_bbox_mask(path, image_size=image_size, bbox=fallback_bbox)


def load_binary_mask(path: str | Path, *, image_size: tuple[int, int]) -> Any:
    """Load a grayscale mask and resize it to the source image if needed."""

    from PIL import Image

    mask = Image.open(path).convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if int(value) > 0 else 0)


def subtract_bboxes_from_mask(mask: Any, bboxes: Sequence[Sequence[int]]) -> Any:
    """Return a copy of ``mask`` with rectangular bbox regions removed."""

    from PIL import ImageDraw

    output = mask.convert("L").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x, y, box_w, box_h = [int(value) for value in bbox]
        if box_w <= 0 or box_h <= 0:
            continue
        x0 = min(max(0, x), width - 1)
        y0 = min(max(0, y), height - 1)
        x1 = min(max(x0 + 1, x + box_w), width)
        y1 = min(max(y0 + 1, y + box_h), height)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)
    return output


def build_object_region_mask(
    image: Any,
    *,
    bbox: Sequence[int],
    target_region: str = "full",
    target_name: str = "",
    subtract_bboxes: Sequence[Sequence[int]] = (),
    source_color: str = "any",
    saturation_threshold: int = 80,
    value_threshold: int = 40,
) -> Any:
    """Build a target-object mask from a layout/VLM bbox and simple shape priors."""

    from PIL import Image, ImageDraw

    source_color = normalize_source_color_mode(source_color)
    rgb = image.convert("RGB")
    width, height = rgb.size
    x, y, box_w, box_h = expand_bbox(
        bbox,
        image_size=(width, height),
        expand=0.0,
    )
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    target_text = f"{target_region} {target_name}".lower()
    if _expects_horizontal_component(target_region, target_name) or any(
        term in target_text for term in ("umbrella", "canopy", "parasol")
    ):
        mask = _horizontal_canopy_mask((width, height), [x, y, box_w, box_h])
    else:
        draw.rectangle([x, y, x + box_w - 1, y + box_h - 1], fill=255)

    if subtract_bboxes:
        mask = subtract_bboxes_from_mask(mask, subtract_bboxes)
    return mask


def _horizontal_canopy_mask(image_size: tuple[int, int], bbox: Sequence[int]) -> Any:
    """Approximate a canopy/parasol object inside a target bbox."""

    from PIL import Image

    width, height = image_size
    x, y, box_w, box_h = [int(value) for value in bbox]
    mask = Image.new("L", (width, height), 0)
    pixels = mask.load()
    cx = x + box_w / 2.0
    cy = y + box_h * 0.72
    rx = max(1.0, box_w / 2.0)
    ry = max(1.0, box_h * 0.72)
    for py in range(y, min(height, y + box_h)):
        for px in range(x, min(width, x + box_w)):
            nx = (px + 0.5 - cx) / rx
            ny = (py + 0.5 - cy) / ry
            if nx * nx + ny * ny <= 1.0 and py <= cy:
                pixels[px, py] = 255
            elif py >= y + box_h * 0.62 and abs(nx) <= 0.95:
                # Keep the lower canopy rim so flat umbrellas are covered.
                pixels[px, py] = 255
    return mask


def _mask_intersection(mask_a: Any, mask_b: Any) -> Any:
    from PIL import ImageChops

    return ImageChops.multiply(mask_a.convert("L"), mask_b.convert("L")).point(
        lambda value: 255 if int(value) > 0 else 0
    )


def count_mask_pixels(mask: Any) -> int:
    """Count nonzero pixels in a grayscale/binary mask."""

    source = mask.convert("L")
    return sum(1 for value in source.getdata() if int(value) > 0)


def measure_color_coverage(
    image_path: str | Path,
    *,
    bbox: Sequence[int],
    target_color: str,
    source_color: str,
    mask_path: str | Path | None = None,
    exclude_bboxes: Sequence[Sequence[int]] = (),
    saturation_threshold: int = 55,
    value_threshold: int = 40,
) -> dict[str, Any]:
    """Measure target/source color coverage inside a bbox or binary mask."""

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    target_mask, target_count = build_color_selection_mask(
        image,
        bbox=bbox,
        source_color=_color_name_or_any(target_color),
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
    )
    source_mask, source_count = build_color_selection_mask(
        image,
        bbox=bbox,
        source_color=source_color,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
    )
    bbox_mask = (
        load_binary_mask(mask_path, image_size=image_size)
        if mask_path
        else _mask_for_bbox(image_size, bbox)
    )
    if exclude_bboxes:
        target_mask = subtract_bboxes_from_mask(target_mask, exclude_bboxes)
        source_mask = subtract_bboxes_from_mask(source_mask, exclude_bboxes)
        bbox_mask = subtract_bboxes_from_mask(bbox_mask, exclude_bboxes)
    target_mask = _mask_intersection(target_mask, bbox_mask)
    source_mask = _mask_intersection(source_mask, bbox_mask)
    target_count = count_mask_pixels(target_mask)
    source_count = count_mask_pixels(source_mask)
    eligible_count = count_mask_pixels(bbox_mask)
    return {
        "bbox": [int(value) for value in bbox],
        "mask_path": str(mask_path) if mask_path else None,
        "exclude_bboxes": [list(bbox) for bbox in exclude_bboxes],
        "eligible_pixel_count": eligible_count,
        "target_color": target_color,
        "source_color": source_color,
        "target_pixel_count": target_count,
        "source_pixel_count": source_count,
        "target_coverage": round(target_count / max(1, eligible_count), 6),
        "source_coverage": round(source_count / max(1, eligible_count), 6),
        "saturation_threshold": saturation_threshold,
        "value_threshold": value_threshold,
    }


def build_color_selection_mask(
    image: Any,
    *,
    bbox: Sequence[int],
    source_color: str = "red",
    saturation_threshold: int = 70,
    value_threshold: int = 35,
) -> tuple[Any, int]:
    """Select colored pixels inside ``bbox`` for deterministic recoloring."""

    from PIL import Image

    source_color = normalize_source_color_mode(source_color)
    if source_color not in SOURCE_COLOR_MODES:
        raise ValueError(
            "source_color must be one of: " + ", ".join(sorted(SOURCE_COLOR_MODES))
        )
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    width, height = rgb.size
    x, y, box_w, box_h = [int(value) for value in bbox]
    x0 = min(max(0, x), width - 1)
    y0 = min(max(0, y), height - 1)
    x1 = min(max(x0 + 1, x + box_w), width)
    y1 = min(max(y0 + 1, y + box_h), height)
    mask = Image.new("L", (width, height), 0)
    hsv_pixels = hsv.load()
    mask_pixels = mask.load()
    selected_count = 0
    for py in range(y0, y1):
        for px in range(x0, x1):
            hue, saturation, value = hsv_pixels[px, py]
            if source_color == "black":
                if value > value_threshold:
                    continue
                mask_pixels[px, py] = 255
                selected_count += 1
                continue
            if value < value_threshold:
                continue
            if source_color == "low_saturation":
                if saturation > saturation_threshold:
                    continue
                mask_pixels[px, py] = 255
                selected_count += 1
                continue
            if saturation < saturation_threshold:
                continue
            if not _hue_matches(hue, source_color):
                continue
            mask_pixels[px, py] = 255
            selected_count += 1
    return mask, selected_count


def mask_components(mask: Any, *, min_area: int = 1) -> list[dict[str, Any]]:
    """Return connected components for nonzero pixels in a mask."""

    from collections import deque

    source = mask.convert("L")
    width, height = source.size
    pixels = source.load()
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0 or (x, y) in visited:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            area = 0
            min_x = max_x = x
            min_y = max_y = y
            sum_x = 0
            sum_y = 0
            while queue:
                current_x, current_y = queue.popleft()
                area += 1
                sum_x += current_x
                sum_y += current_y
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        point = (next_x, next_y)
                        if point in visited or pixels[next_x, next_y] == 0:
                            continue
                        visited.add(point)
                        queue.append(point)
            if area >= min_area:
                components.append(
                    {
                        "bbox": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1],
                        "area": area,
                        "center": [round(sum_x / area, 2), round(sum_y / area, 2)],
                        "touches_image_border": (
                            min_x == 0
                            or min_y == 0
                            or max_x == width - 1
                            or max_y == height - 1
                        ),
                    }
                )
    return sorted(components, key=lambda item: int(item["area"]), reverse=True)


def select_component_for_bbox(
    components: Sequence[Mapping[str, Any]],
    *,
    preferred_bbox: Sequence[int],
    strategy: str = "layout_overlap",
) -> dict[str, Any]:
    """Select the color component that best matches a preferred layout bbox."""

    if not components:
        raise ValueError("components must not be empty")
    if strategy not in {"layout_overlap", "largest"}:
        raise ValueError("strategy must be 'layout_overlap' or 'largest'")
    preferred_area = _bbox_area(preferred_bbox)
    preferred_center = _bbox_center(preferred_bbox)
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for component in components:
        bbox = component.get("bbox", [])
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        area = int(component.get("area", 0))
        overlap = _bbox_overlap_area(bbox, preferred_bbox)
        overlap_score = overlap / max(1, min(_bbox_area(bbox), preferred_area))
        area_score = min(1.0, area / max(1, preferred_area))
        center_x, center_y = _bbox_center(bbox)
        distance = (
            (center_x - preferred_center[0]) ** 2
            + (center_y - preferred_center[1]) ** 2
        ) ** 0.5
        distance_score = 1.0 / (1.0 + distance / 128.0)
        aspect_score = _aspect_reasonableness(bbox)
        if strategy == "largest":
            score = area_score * 3.0 + aspect_score + distance_score * 0.35 + overlap_score * 0.25
        else:
            score = overlap_score * 3.0 + area_score + distance_score
        scored.append((score, component))
    if not scored:
        raise ValueError("no valid components to score")
    score, selected = max(scored, key=lambda item: item[0])
    result = dict(selected)
    result["score"] = round(score, 4)
    result["overlap_with_layout"] = _bbox_overlap_area(result["bbox"], preferred_bbox)
    result["selection_strategy"] = strategy
    return result


def recolor_image_with_mask(
    image: Any,
    mask: Any,
    *,
    target_color: tuple[int, int, int],
    feather_radius: float = 2.0,
    value_floor: int | None = None,
) -> Any:
    """Replace hue/saturation under a mask while preserving local brightness."""

    from PIL import Image, ImageFilter

    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    hue_channel, saturation_channel, value_channel = hsv.split()
    mask_l = mask.convert("L")
    target_hue, target_saturation, _ = Image.new("RGB", (1, 1), target_color).convert(
        "HSV"
    ).getpixel((0, 0))

    hue_pixels = hue_channel.load()
    saturation_pixels = saturation_channel.load()
    value_pixels = value_channel.load()
    mask_pixels = mask_l.load()
    width, height = rgb.size
    floor = None if value_floor is None else max(0, min(255, int(value_floor)))
    for py in range(height):
        for px in range(width):
            if mask_pixels[px, py] == 0:
                continue
            hue_pixels[px, py] = target_hue
            saturation_pixels[px, py] = max(saturation_pixels[px, py], target_saturation)
            if floor is not None:
                value_pixels[px, py] = max(value_pixels[px, py], floor)

    recolored = Image.merge("HSV", (hue_channel, saturation_channel, value_channel)).convert(
        "RGB"
    )
    if feather_radius > 0:
        alpha = mask_l.filter(ImageFilter.GaussianBlur(float(feather_radius)))
    else:
        alpha = mask_l
    return Image.composite(recolored, rgb, alpha)


def keep_largest_mask_component(mask: Any) -> tuple[Any, int]:
    """Keep the largest 8-connected white component in a binary mask."""

    from collections import deque
    from PIL import Image

    source = mask.convert("L")
    width, height = source.size
    pixels = source.load()
    visited: set[tuple[int, int]] = set()
    largest: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0 or (x, y) in visited:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            while queue:
                current_x, current_y = queue.popleft()
                component.append((current_x, current_y))
                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        point = (next_x, next_y)
                        if point in visited or pixels[next_x, next_y] == 0:
                            continue
                        visited.add(point)
                        queue.append(point)
            if len(component) > len(largest):
                largest = component

    output = Image.new("L", (width, height), 0)
    output_pixels = output.load()
    for x, y in largest:
        output_pixels[x, y] = 255
    return output, len(largest)


def parse_rgb_color(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` or ``r,g,b`` color text."""

    value = _clean_text(value, "target_color")
    if value.startswith("#"):
        if len(value) != 7:
            raise ValueError("hex color must be #RRGGBB")
        return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("color must be #RRGGBB or r,g,b")
    channels = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("RGB color channels must be in [0, 255]")
    return channels


def _canvas_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or len(value) != 2:
        return 1024, 1024
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        return 1024, 1024
    return width, height


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _pad_bbox_pixels(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    padding: int,
) -> list[int]:
    width, height = image_size
    x, y, box_w, box_h = [int(value) for value in bbox]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_w + padding)
    y1 = min(height, y + box_h + padding)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _bbox_area(bbox: Sequence[int]) -> int:
    if len(bbox) != 4:
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_center(bbox: Sequence[int]) -> tuple[float, float]:
    x, y, width, height = [int(value) for value in bbox]
    return x + width / 2.0, y + height / 2.0


def _bbox_overlap_area(a: Sequence[int], b: Sequence[int]) -> int:
    ax, ay, aw, ah = [int(value) for value in a]
    bx, by, bw, bh = [int(value) for value in b]
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _target_region_bbox(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    target_region: str,
) -> list[int]:
    target_region = str(target_region or "full").strip().lower()
    if target_region in {"full", "object"}:
        return expand_bbox(bbox, image_size=image_size, expand=0.0)
    x, y, width, height = [int(value) for value in bbox]
    if target_region in {"upper", "canopy", "upper_half"}:
        return expand_bbox(
            [x, y, width, max(1, int(round(height * 0.58)))],
            image_size=image_size,
            expand=0.0,
        )
    if target_region in {"lower", "lower_half"}:
        top = y + int(round(height * 0.42))
        return expand_bbox(
            [x, top, width, max(1, y + height - top)],
            image_size=image_size,
            expand=0.0,
        )
    raise ValueError("target_region must be one of: full, upper, canopy, lower")


def _expects_horizontal_component(target_region: str, target_name: str) -> bool:
    text = f"{target_region} {target_name}".strip().lower()
    horizontal_terms = (
        "canopy",
        "umbrella",
        "parasol",
        "roof",
        "awning",
        "upper",
        "upper_half",
        "wide",
        "horizontal",
    )
    return any(term in text for term in horizontal_terms)


def _subtract_bboxes_from_layout(
    layout_context: Mapping[str, Any],
    target_names: Sequence[str],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[list[int]]:
    names = [str(name).strip().lower() for name in target_names if str(name).strip()]
    if not names:
        return []
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return []
    objects = layout.get("objects", [])
    if not isinstance(objects, Sequence):
        return []
    bboxes: list[list[int]] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        text = " ".join(
            str(obj.get(key, "")) for key in ("name", "description", "relations")
        ).lower()
        if not any(name in text or text in name for name in names):
            continue
        bbox = obj.get("bbox", [])
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        bboxes.append(
            scale_bbox(
                [int(value) for value in bbox],
                from_size=from_size,
                to_size=to_size,
            )
        )
    return bboxes


def _other_object_bboxes_from_layout(
    layout_context: Mapping[str, Any],
    *,
    target_name: str,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[list[int]]:
    layout = layout_context.get("layout", layout_context)
    if not isinstance(layout, Mapping):
        return []
    objects = layout.get("objects", [])
    if not isinstance(objects, Sequence):
        return []
    target = target_name.strip().lower()
    bboxes: list[list[int]] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        name = str(obj.get("name", "")).strip().lower()
        if not name:
            continue
        if target in name or name in target:
            continue
        bbox = obj.get("bbox", [])
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        bboxes.append(
            scale_bbox(
                [int(value) for value in bbox],
                from_size=from_size,
                to_size=to_size,
            )
        )
    return bboxes


def _dedupe_bboxes(bboxes: Sequence[Sequence[int]]) -> list[list[int]]:
    deduped: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        normalized = tuple(int(value) for value in bbox)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(list(normalized))
    return deduped


def _mask_for_bbox(image_size: tuple[int, int], bbox: Sequence[int]) -> Any:
    from PIL import Image, ImageDraw

    width, height = image_size
    x, y, box_w, box_h = [int(value) for value in bbox]
    x0 = min(max(0, x), width - 1)
    y0 = min(max(0, y), height - 1)
    x1 = min(max(x0 + 1, x + box_w), width)
    y1 = min(max(y0 + 1, y + box_h), height)
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
    return mask


def _color_name_or_any(value: str) -> str:
    lowered = normalize_source_color_mode(value)
    if lowered.startswith("#"):
        rgb = parse_rgb_color(lowered)
        if rgb[0] > rgb[1] and rgb[0] > rgb[2]:
            return "red"
        if rgb[1] > rgb[0] and rgb[1] > rgb[2]:
            return "green"
        if rgb[2] > rgb[0] and rgb[2] > rgb[1]:
            return "blue"
        return "any"
    return lowered if lowered in SOURCE_COLOR_MODES else "any"


def _aspect_reasonableness(bbox: Sequence[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    ratio = width / height
    if 0.45 <= ratio <= 3.5:
        return 1.0
    return 0.25


def _hue_matches(hue: int, source_color: str) -> bool:
    source_color = normalize_source_color_mode(source_color)
    if source_color == "any":
        return True
    if source_color == "red":
        return hue <= 20 or hue >= 235
    if source_color in {"orange", "brown"}:
        return 12 <= hue <= 32
    if source_color == "yellow":
        return 25 <= hue <= 50
    if source_color == "green":
        return 55 <= hue <= 120
    if source_color == "cyan":
        return 110 <= hue <= 150
    if source_color == "blue":
        return 135 <= hue <= 185
    if source_color == "purple":
        return 178 <= hue <= 215
    if source_color == "pink":
        return 210 <= hue <= 240
    return False


def normalize_source_color_mode(value: str) -> str:
    """Normalize source-color labels used for local recolor masks."""

    lowered = str(value or "").strip().lower().replace(" ", "_")
    lowered = lowered.replace("-", "_")
    if lowered in {item.replace("-", "_") for item in LOW_SATURATION_ALIASES}:
        return "low_saturation"
    return lowered
