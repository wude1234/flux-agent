"""Automatic local image-editing agent.

This module keeps heavy editing tools out of the main agent process. The
normal path is:

1. ask Grounded-SAM2 for a text-grounded mask in a subprocess,
2. dilate the mask to cover object boundaries,
3. pass source image + mask + edit prompt to a PowerPaint subprocess.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .local_editor import (
    InpaintEditor,
    InpaintRegion,
    count_mask_pixels,
    load_binary_mask,
    scale_bbox,
    write_bbox_mask,
)
from .logging_utils import write_json
from .ocr_verifier import verify_text_in_bbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUNDED_SAM2_DIR = (
    "/home/zrr/t2i_agent_papers_2024_2025/"
    "mult-t2i-agent/code/T2I-Copilot-master/models/Grounded_SAM2"
)
DEFAULT_GROUNDED_SAM2_PYTHON = "/mnt/ssd1/conda/envs/tweediemix/bin/python"
DEFAULT_GROUNDED_SAM2_HF_HOME = "/mnt/ssd1/powerpaint_envs/hf-cache"


class TextMaskGenerator(Protocol):
    """Generate a binary mask from a source image and text expression."""

    def generate(
        self,
        *,
        image_path: str | Path,
        text: str,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        ...


@dataclass
class GroundedSAM2SubprocessMasker:
    """Run T2I-Copilot's Grounded-SAM2 mask code in a separate process."""

    python: str | Path = DEFAULT_GROUNDED_SAM2_PYTHON
    grounded_sam2_dir: str | Path = DEFAULT_GROUNDED_SAM2_DIR
    script_path: str | Path = PROJECT_ROOT / "scripts" / "run_grounded_sam2_mask.py"
    timeout_seconds: int | None = 900
    cuda_visible_devices: str | None = None
    local_files_only: bool = True
    hf_home: str | Path | None = DEFAULT_GROUNDED_SAM2_HF_HOME
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.python = Path(self.python)
        self.grounded_sam2_dir = Path(self.grounded_sam2_dir)
        self.script_path = Path(self.script_path)
        if self.hf_home is not None:
            self.hf_home = Path(self.hf_home)
        if not self.python.exists():
            raise FileNotFoundError(f"Grounded-SAM2 Python executable does not exist: {self.python}")
        if not self.grounded_sam2_dir.exists():
            raise FileNotFoundError(f"Grounded-SAM2 directory does not exist: {self.grounded_sam2_dir}")
        if not self.script_path.exists():
            raise FileNotFoundError(f"Grounded-SAM2 wrapper script does not exist: {self.script_path}")

    def generate(
        self,
        *,
        image_path: str | Path,
        text: str,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            str(self.script_path),
            "--image",
            str(image_path),
            "--text",
            str(text),
            "--output-dir",
            str(output_dir),
            "--grounded-sam2-dir",
            str(self.grounded_sam2_dir),
        ]
        if self.local_files_only:
            command.append("--local-files-only")
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        if self.hf_home is not None:
            hf_home = str(self.hf_home)
            env["HF_HOME"] = hf_home
            env["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
            env["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.grounded_sam2_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "method": "grounded_sam2_subprocess",
                "error": f"Grounded-SAM2 subprocess timed out after {self.timeout_seconds}s",
                "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                "command": command,
            }
            self.calls.append(deepcopy(result))
            return result

        payload = _parse_last_json(completed.stdout)
        if completed.returncode != 0:
            result = {
                "ok": False,
                "method": "grounded_sam2_subprocess",
                "error": (
                    payload.get("error")
                    if isinstance(payload, Mapping) and payload.get("error")
                    else f"Grounded-SAM2 subprocess failed with exit code {completed.returncode}"
                ),
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-2000:],
                "command": command,
                "payload": dict(payload) if isinstance(payload, Mapping) else {},
            }
            self.calls.append(deepcopy(result))
            return result
        if not isinstance(payload, Mapping) or not payload.get("ok") or not payload.get("mask_path"):
            result = {
                "ok": False,
                "method": "grounded_sam2_subprocess",
                "error": "Grounded-SAM2 subprocess did not return a valid mask payload",
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-2000:],
                "command": command,
                "payload": dict(payload) if isinstance(payload, Mapping) else {},
            }
            self.calls.append(deepcopy(result))
            return result
        mask_path = Path(str(payload["mask_path"])).resolve()
        result = {
            **dict(payload),
            "ok": True,
            "method": "grounded_sam2_subprocess",
            "mask_path": str(mask_path),
            "command": command,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        self.calls.append(deepcopy(result))
        return result


@dataclass
class MaskGeneratingInpaintEditor:
    """Inpaint editor wrapper that creates a text-grounded mask before editing."""

    base_editor: InpaintEditor
    mask_generator: TextMaskGenerator | None = None
    mask_mode: str = "auto"
    mask_text: str | None = None
    allow_bbox_fallback: bool = True
    dilation_kernel_size: int = 51
    min_mask_area_ratio: float = 0.0005
    prefix: str = "editing_agent"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self) -> None:
        if self.mask_mode not in {"auto", "grounded-sam2", "bbox"}:
            raise ValueError("mask_mode must be auto, grounded-sam2, or bbox")
        if self.dilation_kernel_size < 1:
            raise ValueError("dilation_kernel_size must be positive")
        if self.min_mask_area_ratio < 0:
            raise ValueError("min_mask_area_ratio must be non-negative")

    def edit(
        self,
        image_path: str,
        region: InpaintRegion,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = output_dir / f"{self.prefix}_mask_{self._counter:04d}"
        mask_dir.mkdir(parents=True, exist_ok=True)

        effective_mask_mode = self.mask_mode
        auto_bbox_reason = None
        if effective_mask_mode == "auto":
            auto_bbox_reason = _auto_bbox_reason_for_region(region)
            if auto_bbox_reason:
                effective_mask_mode = "bbox"
        masked_region, mask_record = prepare_masked_inpaint_region(
            image_path=image_path,
            region=region,
            output_dir=mask_dir,
            mask_generator=self.mask_generator,
            mask_mode=effective_mask_mode,
            mask_text=self._mask_text_for_region(region),
            allow_bbox_fallback=self.allow_bbox_fallback,
            dilation_kernel_size=self.dilation_kernel_size,
            min_mask_area_ratio=self.min_mask_area_ratio,
        )
        if auto_bbox_reason:
            mask_record["auto_bbox_reason"] = auto_bbox_reason
            mask_record["requested_mask_mode"] = self.mask_mode
            write_json(mask_dir / "mask_agent_plan.json", mask_record)
        edit_result = self.base_editor.edit(image_path, masked_region, output_dir / "powerpaint")
        result = {
            **dict(edit_result),
            "mask_agent": mask_record,
            "region_before_mask_agent": region.to_dict(),
            "region_after_mask_agent": masked_region.to_dict(),
        }
        write_json(output_dir / f"{self.prefix}_edit_{self._counter:04d}.json", result)
        self.calls.append(deepcopy(result))
        self._counter += 1
        return result

    def _mask_text_for_region(self, region: InpaintRegion) -> str:
        if self.mask_text:
            return str(self.mask_text)
        if region.name and not region.name.startswith(("relation_action_contact", "forced edit")):
            return region.name
        return ""


@dataclass
class GroundedSAM2PowerPaintEditingAgent:
    """High-level editing agent for one automatic local PowerPaint repair."""

    editor: InpaintEditor
    mask_generator: TextMaskGenerator | None = None
    mask_mode: str = "auto"
    allow_bbox_fallback: bool = True
    dilation_kernel_size: int = 51
    min_mask_area_ratio: float = 0.0005
    prefix: str = "editing_agent"

    def edit(
        self,
        *,
        image_path: str | Path,
        output_dir: str | Path,
        target_object: str,
        edit_prompt: str,
        bbox: Sequence[int],
        canvas_size: Sequence[int] | None = None,
        mask_text: str | None = None,
        negative_prompt: str = "",
        reason: str = "automatic Grounded-SAM2 mask + PowerPaint edit",
    ) -> dict[str, Any]:
        image_path = str(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if canvas_size is None:
            with Image.open(image_path) as image:
                canvas_size = [image.width, image.height]
        region = InpaintRegion(
            name=str(target_object or "edit target"),
            bbox=[int(value) for value in bbox],
            prompt=str(edit_prompt),
            negative_prompt=str(negative_prompt or ""),
            reason=str(reason or ""),
            canvas_size=[int(value) for value in canvas_size],
        )
        wrapper = MaskGeneratingInpaintEditor(
            base_editor=self.editor,
            mask_generator=self.mask_generator,
            mask_mode=self.mask_mode,
            mask_text=mask_text,
            allow_bbox_fallback=self.allow_bbox_fallback,
            dilation_kernel_size=self.dilation_kernel_size,
            min_mask_area_ratio=self.min_mask_area_ratio,
            prefix=self.prefix,
        )
        result = wrapper.edit(image_path, region, output_dir)
        contact_sheet = write_edit_contact_sheet(
            source_image_path=image_path,
            raw_mask_path=result["mask_agent"].get("raw_mask_path"),
            dilated_mask_path=result["mask_agent"].get("dilated_mask_path"),
            edited_image_path=result.get("edited_image"),
            output_path=output_dir / "before_rawmask_dilatedmask_after.jpg",
            bbox=result["mask_agent"].get("source_bbox"),
        )
        payload = {
            "ok": True,
            "type": "grounded_sam2_powerpaint_editing_agent",
            "source_image": image_path,
            "target_object": target_object,
            "mask_text": mask_text,
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "result": result,
        }
        write_json(output_dir / "editing_agent_result.json", payload)
        return payload


@dataclass(frozen=True)
class EfficientRepairRequest:
    """A typed, low-overhead edit request.

    The router treats this as an execution contract from the repair planner:
    cheap deterministic repairs run locally, localized inpaint routes can call
    PowerPaint/SD, and low-editability layout failures are reported back for
    regeneration instead of wasting GPU time.
    """

    repair_kind: str
    image_path: str | Path
    output_dir: str | Path
    bbox: Sequence[int]
    target_object: str = ""
    prompt: str = ""
    text: str = ""
    symbol: str = ""
    fill_color: str = ""
    text_color: str = ""
    negative_prompt: str = ""
    mask_text: str | None = None
    canvas_size: Sequence[int] | None = None
    reason: str = ""


@dataclass
class EfficientRepairAgent:
    """Route repair requests through the cheapest reliable tool first."""

    inpaint_agent: GroundedSAM2PowerPaintEditingAgent | None = None
    default_font_path: str | Path | None = (
        "/home/zrr/t2i_agent_papers_2024_2025/"
        "mult-t2i-agent/code/guang/PosterMaker-main/assets/fonts/"
        "AlibabaPuHuiTi-3-55-Regular.ttf"
    )

    def repair(self, request: EfficientRepairRequest | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(request, Mapping):
            request = EfficientRepairRequest(**dict(request))
        kind = _normalize_repair_kind(request.repair_kind)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if kind in {"layout_regenerate", "count_rerank", "candidate_rerank"}:
            payload = {
                "ok": False,
                "accepted": False,
                "type": "efficient_repair_agent",
                "route": kind,
                "source_image": str(request.image_path),
                "reason": request.reason
                or "This failure is low-editability; route to regeneration or candidate reranking.",
                "gpu_used": False,
                "sam2_used": False,
                "powerpaint_used": False,
            }
            write_json(output_dir / "efficient_repair_result.json", payload)
            return payload
        if kind == "text_overlay":
            return self._text_overlay(request, output_dir)
        if kind == "symbol_overlay":
            return self._symbol_overlay(request, output_dir)
        if kind == "shape_overlay":
            return self._shape_overlay(request, output_dir)
        if kind == "bbox_shape_inpaint":
            return self._bbox_shape_inpaint(request, output_dir)
        if kind == "existing_object_inpaint":
            return self._existing_object_inpaint(request, output_dir)
        payload = {
            "ok": False,
            "accepted": False,
            "type": "efficient_repair_agent",
            "route": kind,
            "source_image": str(request.image_path),
            "error": f"unsupported repair_kind: {request.repair_kind}",
            "gpu_used": False,
            "sam2_used": False,
            "powerpaint_used": False,
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload

    def _shape_overlay(self, request: EfficientRepairRequest, output_dir: Path) -> dict[str, Any]:
        source = Image.open(request.image_path).convert("RGB")
        bbox = _scaled_request_bbox(request, source.size)
        fill = parse_color(request.fill_color or request.prompt, default=(210, 35, 42))
        edited = source.copy()
        draw = ImageDraw.Draw(edited)
        _draw_clean_box(draw, bbox, fill)
        edited_path = output_dir / "shape_overlay_repair.png"
        mask_path = output_dir / "shape_overlay_mask.png"
        edited.save(edited_path)
        write_bbox_mask(mask_path, image_size=source.size, bbox=bbox)
        contact_sheet = write_edit_contact_sheet(
            source_image_path=request.image_path,
            raw_mask_path=mask_path,
            dilated_mask_path=mask_path,
            edited_image_path=edited_path,
            output_path=output_dir / "before_mask_after.jpg",
            bbox=bbox,
        )
        payload = {
            "ok": True,
            "accepted": True,
            "type": "efficient_repair_agent",
            "route": "shape_overlay",
            "source_image": str(request.image_path),
            "edited_image": str(edited_path),
            "mask_path": str(mask_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "bbox": bbox,
            "target_object": request.target_object,
            "fill_color": list(fill),
            "gpu_used": False,
            "sam2_used": False,
            "powerpaint_used": False,
            "reason": request.reason or "deterministic simple-shape occluder overlay",
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload

    def _text_overlay(self, request: EfficientRepairRequest, output_dir: Path) -> dict[str, Any]:
        source = Image.open(request.image_path).convert("RGB")
        bbox = _scaled_request_bbox(request, source.size)
        fill = parse_color(request.fill_color, default=(12, 12, 12))
        text_color = parse_color(request.text_color, default=(245, 210, 35))
        text = str(request.text or _quoted_text_from_prompt(request.prompt) or request.target_object or "").strip()
        edited = source.copy()
        draw = ImageDraw.Draw(edited)
        _draw_clean_box(draw, bbox, fill)
        if text:
            font = _fit_overlay_font(draw, text, bbox, self.default_font_path)
            _draw_centered_text(draw, text, bbox, font, text_color)
        edited_path = output_dir / "text_overlay_repair.png"
        mask_path = output_dir / "text_overlay_mask.png"
        edited.save(edited_path)
        write_bbox_mask(mask_path, image_size=source.size, bbox=bbox)
        contact_sheet = write_edit_contact_sheet(
            source_image_path=request.image_path,
            raw_mask_path=mask_path,
            dilated_mask_path=mask_path,
            edited_image_path=edited_path,
            output_path=output_dir / "before_mask_after.jpg",
            bbox=bbox,
        )
        ocr_verification = verify_text_in_bbox(
            edited_path,
            expected_text=text,
            bbox=bbox,
            crop_output_path=output_dir / "text_overlay_ocr_crop.png",
        ) if text else {
            "available": False,
            "passed": None,
            "error": "missing rendered_text",
            "items": [],
        }
        payload = {
            "ok": True,
            "accepted": True,
            "type": "efficient_repair_agent",
            "route": "text_overlay",
            "source_image": str(request.image_path),
            "edited_image": str(edited_path),
            "mask_path": str(mask_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "bbox": bbox,
            "rendered_text": text,
            "fill_color": list(fill),
            "text_color": list(text_color),
            "ocr_verification": ocr_verification,
            "gpu_used": False,
            "sam2_used": False,
            "powerpaint_used": False,
            "reason": request.reason or "deterministic text rendering with OCR-ready geometry",
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload

    def _symbol_overlay(self, request: EfficientRepairRequest, output_dir: Path) -> dict[str, Any]:
        source = Image.open(request.image_path).convert("RGB")
        bbox = _scaled_request_bbox(request, source.size)
        fill = parse_color(request.fill_color, default=(20, 20, 20))
        symbol_color = parse_color(request.text_color, default=(245, 245, 245))
        symbol = str(request.symbol or _symbol_from_prompt(request.prompt) or request.text or "circle").strip()
        edited = source.copy()
        draw = ImageDraw.Draw(edited)
        _draw_clean_box(draw, bbox, fill)
        _draw_symbol(draw, bbox, symbol, symbol_color)
        edited_path = output_dir / "symbol_overlay_repair.png"
        mask_path = output_dir / "symbol_overlay_mask.png"
        edited.save(edited_path)
        write_bbox_mask(mask_path, image_size=source.size, bbox=bbox)
        contact_sheet = write_edit_contact_sheet(
            source_image_path=request.image_path,
            raw_mask_path=mask_path,
            dilated_mask_path=mask_path,
            edited_image_path=edited_path,
            output_path=output_dir / "before_mask_after.jpg",
            bbox=bbox,
        )
        payload = {
            "ok": True,
            "accepted": True,
            "type": "efficient_repair_agent",
            "route": "symbol_overlay",
            "source_image": str(request.image_path),
            "edited_image": str(edited_path),
            "mask_path": str(mask_path),
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "bbox": bbox,
            "symbol": symbol,
            "fill_color": list(fill),
            "symbol_color": list(symbol_color),
            "gpu_used": False,
            "sam2_used": False,
            "powerpaint_used": False,
            "reason": request.reason or "deterministic simple-symbol repair",
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload

    def _bbox_shape_inpaint(self, request: EfficientRepairRequest, output_dir: Path) -> dict[str, Any]:
        if self.inpaint_agent is None:
            payload = {
                "ok": False,
                "accepted": False,
                "type": "efficient_repair_agent",
                "route": "bbox_shape_inpaint",
                "source_image": str(request.image_path),
                "error": "bbox_shape_inpaint requires an inpaint_agent",
                "gpu_used": False,
                "sam2_used": False,
                "powerpaint_used": False,
            }
            write_json(output_dir / "efficient_repair_result.json", payload)
            return payload
        bbox_agent = GroundedSAM2PowerPaintEditingAgent(
            editor=self.inpaint_agent.editor,
            mask_generator=self.inpaint_agent.mask_generator,
            mask_mode="bbox",
            allow_bbox_fallback=True,
            dilation_kernel_size=1,
            min_mask_area_ratio=self.inpaint_agent.min_mask_area_ratio,
            prefix=self.inpaint_agent.prefix,
        )
        result = bbox_agent.edit(
            image_path=request.image_path,
            output_dir=output_dir,
            target_object=request.target_object or "new local object",
            edit_prompt=request.prompt,
            bbox=list(request.bbox),
            canvas_size=request.canvas_size,
            mask_text=None,
            negative_prompt=request.negative_prompt,
            reason=request.reason or "bbox/shape mask inpaint without SAM2",
        )
        nested_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        payload = {
            **dict(result),
            "type": "efficient_repair_agent",
            "route": "bbox_shape_inpaint",
            "edited_image": nested_result.get("edited_image") or result.get("edited_image"),
            "mask_path": nested_result.get("mask_path") or result.get("mask_path"),
            "accepted": bool(result.get("ok", False)),
            "gpu_used": _editor_result_uses_gpu(result),
            "sam2_used": False,
            "powerpaint_used": _editor_result_uses_powerpaint(result),
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload

    def _existing_object_inpaint(self, request: EfficientRepairRequest, output_dir: Path) -> dict[str, Any]:
        if self.inpaint_agent is None:
            payload = {
                "ok": False,
                "accepted": False,
                "type": "efficient_repair_agent",
                "route": "existing_object_inpaint",
                "source_image": str(request.image_path),
                "error": "existing_object_inpaint requires an inpaint_agent",
                "gpu_used": False,
                "sam2_used": False,
                "powerpaint_used": False,
            }
            write_json(output_dir / "efficient_repair_result.json", payload)
            return payload
        result = self.inpaint_agent.edit(
            image_path=request.image_path,
            output_dir=output_dir,
            target_object=request.target_object or request.mask_text or "existing object",
            edit_prompt=request.prompt,
            bbox=list(request.bbox),
            canvas_size=request.canvas_size,
            mask_text=request.mask_text or request.target_object,
            negative_prompt=request.negative_prompt,
            reason=request.reason or "existing-object localized inpaint",
        )
        nested_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        mask_agent = dict(nested_result.get("mask_agent", {}))
        payload = {
            **dict(result),
            "type": "efficient_repair_agent",
            "route": "existing_object_inpaint",
            "edited_image": nested_result.get("edited_image") or result.get("edited_image"),
            "mask_path": nested_result.get("mask_path") or result.get("mask_path"),
            "accepted": bool(result.get("ok", False)),
            "gpu_used": _editor_result_uses_gpu(result),
            "sam2_used": mask_agent.get("mask_source") == "grounded_sam2",
            "powerpaint_used": _editor_result_uses_powerpaint(result),
        }
        write_json(output_dir / "efficient_repair_result.json", payload)
        return payload


def route_repair_kind(repair_plan: Mapping[str, Any], prompt: str = "") -> str:
    """Map a planner/evaluator record to the cheapest repair route."""

    route = str(repair_plan.get("typed_route") or "").strip().lower()
    action = str(repair_plan.get("primary_action") or "").strip().lower()
    attr = str(repair_plan.get("target_attribute") or "").strip().lower()
    text = " ".join(
        str(value)
        for value in (
            prompt,
            repair_plan.get("reason", ""),
            repair_plan.get("target_object", ""),
            repair_plan.get("target_attribute", ""),
        )
    ).lower()
    if route in {
        "text_overlay",
        "symbol_overlay",
        "shape_overlay",
        "bbox_shape_inpaint",
        "existing_object_inpaint",
        "layout_regenerate",
        "count_rerank",
    }:
        return route
    if route in {"exact_text_overlay", "wrong_exact_text"}:
        return "text_overlay"
    if route in {"forbidden_symbol_removal", "forbidden_symbol_present"}:
        return "existing_object_inpaint"
    if route in {
        "forbidden_object_removal",
        "single_attribute_patch",
        "relation_contact_repair",
    }:
        return "existing_object_inpaint"
    if route in {
        "count_aware_regeneration",
        "comparative_count_rerank",
        "wrong_count",
    }:
        return "count_rerank"
    if route in {
        "layout_guided_regeneration",
        "relation_focused_regeneration",
        "multi_constraint_decompose",
        "broad_multi_failure",
        "comparative_attribute_binding",
        "role_action_binding_regeneration",
        "lexical_grounding_regeneration",
        "material_guided_regeneration",
        "object_type_guided_regeneration",
    }:
        return "layout_regenerate"
    if action in {"regenerate", "layout_regenerate"}:
        return "layout_regenerate"
    if action in {"count_rerank", "candidate_rerank"}:
        return "count_rerank"
    if action in {
        "text_overlay",
        "symbol_overlay",
        "shape_overlay",
        "bbox_shape_inpaint",
        "existing_object_inpaint",
    }:
        return action
    if any(token in route for token in ("layout", "spatial")) or attr in {
        "spatial_relation",
        "layout",
    }:
        return "layout_regenerate"
    if "count" in route or attr == "count":
        return "count_rerank"
    if any(token in text for token in ("exact text", " text ", "word", "letter")):
        return "text_overlay"
    if "symbol" in text or any(token in text for token in ("triangle", "moon", "circle", "star")):
        return "symbol_overlay"
    if route == "occlusion_object_insertion" and _simple_shape_overlay_target(
        repair_plan,
        text,
    ):
        return "shape_overlay"
    if route == "occlusion_object_insertion" or action == "object_insertion":
        return "bbox_shape_inpaint"
    if action in {"recolor", "relation_repair", "local_repair"}:
        return "existing_object_inpaint"
    return "layout_regenerate"


def _editor_result_uses_powerpaint(result: Mapping[str, Any]) -> bool:
    mode = str(result.get("mode") or "").strip().lower()
    if "powerpaint" in mode:
        return True
    nested = result.get("result")
    if isinstance(nested, Mapping):
        return _editor_result_uses_powerpaint(nested)
    return bool(result.get("powerpaint_used") is True)


def _editor_result_uses_gpu(result: Mapping[str, Any]) -> bool:
    if _editor_result_uses_powerpaint(result):
        return True
    mode = str(result.get("mode") or "").strip().lower()
    if mode == "mock":
        return False
    nested = result.get("result")
    if isinstance(nested, Mapping):
        return _editor_result_uses_gpu(nested)
    return bool(result.get("gpu_used") is True)


def prepare_masked_inpaint_region(
    *,
    image_path: str | Path,
    region: InpaintRegion,
    output_dir: str | Path,
    mask_generator: TextMaskGenerator | None = None,
    mask_mode: str = "auto",
    mask_text: str = "",
    allow_bbox_fallback: bool = True,
    dilation_kernel_size: int = 51,
    min_mask_area_ratio: float = 0.0005,
) -> tuple[InpaintRegion, dict[str, Any]]:
    """Return an ``InpaintRegion`` whose mask_path points to a dilated mask."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    source_bbox = scale_bbox(
        region.bbox,
        from_size=(int(region.canvas_size[0]), int(region.canvas_size[1])),
        to_size=image_size,
    )
    raw_mask_path = output_dir / "raw_mask.png"
    dilated_mask_path = output_dir / "dilated_mask.png"

    mask_generation: dict[str, Any] = {
        "attempted": False,
        "method": "none",
        "mask_text": str(mask_text or ""),
    }
    selected_source = "bbox"
    selected_raw_path: Path | None = None
    should_try_grounded = mask_mode in {"auto", "grounded-sam2"} and bool(mask_text.strip())
    if region.mask_path:
        selected_source = "precomputed_region_mask"
        selected_raw_path = Path(region.mask_path)
        mask_generation = {
            "attempted": False,
            "method": "precomputed_region_mask",
            "mask_path": str(selected_raw_path),
            "mask_text": str(mask_text or ""),
        }
    elif should_try_grounded and mask_generator is not None:
        mask_generation = mask_generator.generate(
            image_path=image_path,
            text=mask_text,
            output_dir=output_dir / "grounded_sam2",
        )
        if mask_generation.get("ok") and mask_generation.get("mask_path"):
            selected_source = "grounded_sam2"
            selected_raw_path = Path(str(mask_generation["mask_path"]))
        elif mask_mode == "grounded-sam2" and not allow_bbox_fallback:
            raise RuntimeError(
                "Grounded-SAM2 mask generation failed and bbox fallback is disabled: "
                f"{mask_generation.get('error', 'unknown error')}"
            )
    elif mask_mode == "grounded-sam2" and not allow_bbox_fallback:
        raise RuntimeError("Grounded-SAM2 mask requested but no mask generator/text was provided")

    fallback_reason = None
    if selected_raw_path is None:
        fallback_reason = _bbox_fallback_reason(mask_mode, mask_text, mask_generation)
        write_bbox_mask(raw_mask_path, image_size=image_size, bbox=source_bbox)
        selected_raw_path = raw_mask_path
        selected_source = "bbox_fallback" if fallback_reason else "bbox"
    else:
        mask = load_binary_mask(selected_raw_path, image_size=image_size)
        mask.save(raw_mask_path)

    raw_mask = load_binary_mask(raw_mask_path, image_size=image_size)
    raw_pixels = count_mask_pixels(raw_mask)
    raw_area_ratio = raw_pixels / max(1, image_size[0] * image_size[1])
    if raw_area_ratio < min_mask_area_ratio:
        if not allow_bbox_fallback:
            raise RuntimeError(
                f"generated mask is too small: area_ratio={raw_area_ratio:.6f}, "
                f"min={min_mask_area_ratio:.6f}"
            )
        fallback_reason = f"generated mask too small: area_ratio={raw_area_ratio:.6f}"
        selected_source = "bbox_fallback_small_mask"
        write_bbox_mask(raw_mask_path, image_size=image_size, bbox=source_bbox)

    dilation_record = dilate_mask_path(
        raw_mask_path,
        dilated_mask_path,
        image_size=image_size,
        kernel_size=dilation_kernel_size,
    )
    mask_bbox = bbox_from_mask(dilated_mask_path, image_size=image_size) or source_bbox
    masked_region = InpaintRegion(
        name=region.name,
        bbox=mask_bbox,
        prompt=region.prompt,
        negative_prompt=region.negative_prompt,
        reason=region.reason,
        canvas_size=[image_size[0], image_size[1]],
        mask_path=str(dilated_mask_path),
    )
    record: dict[str, Any] = {
        "method": "grounded_sam2_dilated_mask_powerpaint_prepare",
        "mask_mode": mask_mode,
        "mask_source": selected_source,
        "mask_text": str(mask_text or ""),
        "fallback_reason": fallback_reason,
        "source_image": str(image_path),
        "image_size": [image_size[0], image_size[1]],
        "source_bbox": source_bbox,
        "mask_bbox": mask_bbox,
        "raw_mask_path": str(raw_mask_path),
        "dilated_mask_path": str(dilated_mask_path),
        "raw_pixel_count": count_mask_pixels(load_binary_mask(raw_mask_path, image_size=image_size)),
        "dilated_pixel_count": dilation_record["pixel_count"],
        "dilated_area_ratio": dilation_record["area_ratio"],
        "dilation": dilation_record,
        "mask_generation": mask_generation,
        "region_before": region.to_dict(),
        "region_after": masked_region.to_dict(),
    }
    write_json(output_dir / "mask_agent_plan.json", record)
    return masked_region, record


def dilate_mask_path(
    mask_path: str | Path,
    output_path: str | Path,
    *,
    image_size: tuple[int, int],
    kernel_size: int = 51,
) -> dict[str, Any]:
    """Dilate a binary mask and write it to ``output_path``."""

    mask = load_binary_mask(mask_path, image_size=image_size)
    effective_kernel = int(kernel_size)
    if effective_kernel < 1:
        raise ValueError("kernel_size must be positive")
    if effective_kernel > 1:
        if effective_kernel % 2 == 0:
            effective_kernel += 1
        mask = mask.filter(ImageFilter.MaxFilter(effective_kernel))
    mask = mask.point(lambda value: 255 if int(value) > 0 else 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    pixels = count_mask_pixels(mask)
    return {
        "input_mask_path": str(mask_path),
        "output_mask_path": str(output_path),
        "requested_kernel_size": int(kernel_size),
        "effective_kernel_size": effective_kernel,
        "pixel_count": pixels,
        "area_ratio": round(pixels / max(1, image_size[0] * image_size[1]), 6),
    }


def bbox_from_mask(mask_path: str | Path, *, image_size: tuple[int, int]) -> list[int] | None:
    """Return [x, y, w, h] for nonzero mask pixels."""

    mask = load_binary_mask(mask_path, image_size=image_size)
    bbox = mask.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = [int(value) for value in bbox]
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def write_edit_contact_sheet(
    *,
    source_image_path: str | Path,
    raw_mask_path: str | Path | None,
    dilated_mask_path: str | Path | None,
    edited_image_path: str | Path | None,
    output_path: str | Path,
    bbox: Sequence[int] | None = None,
) -> Path | None:
    """Write a before/raw-mask/dilated-mask/after sheet when all images exist."""

    if not raw_mask_path or not dilated_mask_path or not edited_image_path:
        return None
    edited_path = Path(edited_image_path)
    if not edited_path.exists():
        return None
    try:
        source = Image.open(source_image_path).convert("RGB")
        raw = Image.open(raw_mask_path).convert("RGB").resize(source.size)
        dilated = Image.open(dilated_mask_path).convert("RGB").resize(source.size)
        edited = Image.open(edited_path).convert("RGB").resize(source.size)
    except Exception:
        return None
    source_panel = source.copy()
    if bbox:
        draw = ImageDraw.Draw(source_panel)
        x, y, width, height = [int(value) for value in bbox]
        line_width = max(2, source.width // 180)
        draw.rectangle([x, y, x + width, y + height], outline=(255, 0, 0), width=line_width)

    panels = [source_panel, raw, dilated, edited]
    labels = ["before + bbox", "raw mask", "dilated mask", "after"]
    label_h = max(28, source.height // 18)
    sheet = Image.new("RGB", (source.width * len(panels), source.height + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, panel in enumerate(panels):
        x0 = index * source.width
        sheet.paste(panel, (x0, label_h))
        draw.text((x0 + 8, 8), labels[index], fill=(20, 20, 20))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def _bbox_fallback_reason(
    mask_mode: str,
    mask_text: str,
    mask_generation: Mapping[str, Any],
) -> str | None:
    if mask_mode == "bbox":
        return None
    if not mask_text.strip():
        return "missing mask text"
    if not mask_generation:
        return "mask generator unavailable"
    return str(mask_generation.get("error") or "mask generator returned no usable mask")


def _auto_bbox_reason_for_region(region: InpaintRegion) -> str | None:
    text = " ".join([region.name, region.prompt, region.reason]).lower()
    if any(
        token in text
        for token in (
            "foreground occluder",
            "occluder",
            "occlusion",
            "add ",
            "insert ",
            "new object",
            "covering",
            "opaque",
            "flat",
            "panel",
            "screen",
            "sheet",
            "patch",
            "place the",
        )
    ):
        return "auto mode uses bbox for new-object/occlusion insertion instead of searching SAM2 for absent content"
    return None


def _parse_last_json(text: str) -> dict[str, Any]:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _normalize_repair_kind(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "text": "text_overlay",
        "ocr": "text_overlay",
        "text_symbol": "text_overlay",
        "symbol": "symbol_overlay",
        "simple_shape_overlay": "shape_overlay",
        "shape_overlay": "shape_overlay",
        "primitive_overlay": "shape_overlay",
        "flat_overlay": "shape_overlay",
        "occlusion_overlay": "shape_overlay",
        "bbox": "bbox_shape_inpaint",
        "shape": "shape_overlay",
        "object_insertion": "bbox_shape_inpaint",
        "occlusion_object_insertion": "bbox_shape_inpaint",
        "forbidden_object_removal": "existing_object_inpaint",
        "forbidden_symbol_removal": "existing_object_inpaint",
        "exact_text_overlay": "text_overlay",
        "single_attribute_patch": "existing_object_inpaint",
        "relation_contact_repair": "existing_object_inpaint",
        "count_aware_regeneration": "count_rerank",
        "comparative_count_rerank": "count_rerank",
        "layout_guided_regeneration": "layout_regenerate",
        "relation_focused_regeneration": "layout_regenerate",
        "multi_constraint_decompose": "layout_regenerate",
        "comparative_attribute_binding": "layout_regenerate",
        "role_action_binding_regeneration": "layout_regenerate",
        "lexical_grounding_regeneration": "layout_regenerate",
        "material_guided_regeneration": "layout_regenerate",
        "object_type_guided_regeneration": "layout_regenerate",
        "sam2_inpaint": "existing_object_inpaint",
        "grounded_sam2": "existing_object_inpaint",
        "recolor": "existing_object_inpaint",
        "relation_repair": "existing_object_inpaint",
        "regenerate": "layout_regenerate",
    }
    return aliases.get(text, text)


def _simple_shape_overlay_target(
    repair_plan: Mapping[str, Any],
    text: str,
) -> bool:
    """Return whether a local failure can be repaired by deterministic painting.

    This is a category-level decision, not a prompt-specific shortcut. It is
    meant for low-semantic planar primitives whose visual contract is mostly
    "put an opaque colored region at this bbox": panels, screens, cards,
    labels, patches, simple covers, and similar flat occluders. Objects that
    require realistic geometry, pose, identity, count, or spatial rearrangement
    stay on regeneration/inpaint routes.
    """

    attr = str(repair_plan.get("target_attribute") or "").strip().lower()
    target = str(
        repair_plan.get("target_object")
        or repair_plan.get("target_name")
        or ""
    ).strip().lower()
    region = str(repair_plan.get("target_region") or "").strip().lower()
    combined = " ".join([target, attr, region, text]).lower()
    if any(token in combined for token in ("spatial", "layout", "count", "exact text", "symbol")):
        return False
    if not any(token in combined for token in ("occlusion", "occluder", "cover", "hide", "hides", "hidden")):
        return False
    primitive_terms = {
        "panel",
        "screen",
        "card",
        "label",
        "tag",
        "patch",
        "sticker",
        "rectangle",
        "square",
        "bar",
        "strip",
        "sheet",
        "board",
        "sign",
        "cover",
        "curtain",
        "cloth",
        "blocker",
        "mask",
    }
    if any(term in target.split() for term in primitive_terms):
        return True
    if any(term in combined for term in ("flat", "opaque", "solid color", "plain colored")):
        return True
    return False


COLOR_TABLE: dict[str, tuple[int, int, int]] = {
    "black": (10, 10, 10),
    "white": (245, 245, 245),
    "yellow": (246, 210, 35),
    "red": (210, 35, 42),
    "green": (38, 145, 72),
    "blue": (42, 92, 196),
    "orange": (232, 123, 31),
    "purple": (126, 75, 188),
    "pink": (225, 108, 165),
    "cyan": (44, 188, 202),
    "turquoise": (30, 168, 160),
    "silver": (190, 192, 196),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def parse_color(value: str | Sequence[int] | None, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        channels = [int(channel) for channel in list(value)[:3]]
        if len(channels) == 3:
            return tuple(max(0, min(255, channel)) for channel in channels)  # type: ignore[return-value]
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in COLOR_TABLE:
        return COLOR_TABLE[text]
    if text.startswith("#") and len(text) in {4, 7}:
        try:
            if len(text) == 4:
                return tuple(int(ch * 2, 16) for ch in text[1:4])  # type: ignore[return-value]
            return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
        except ValueError:
            return default
    for name, color in COLOR_TABLE.items():
        if name in text:
            return color
    return default


def _scaled_request_bbox(request: EfficientRepairRequest, image_size: tuple[int, int]) -> list[int]:
    canvas = request.canvas_size
    if canvas is None:
        canvas = [image_size[0], image_size[1]]
    return scale_bbox(
        request.bbox,
        from_size=(int(canvas[0]), int(canvas[1])),
        to_size=image_size,
    )


def _draw_clean_box(draw: ImageDraw.ImageDraw, bbox: Sequence[int], fill: tuple[int, int, int]) -> None:
    x, y, width, height = [int(value) for value in bbox]
    radius = max(0, min(width, height) // 18)
    box = [x, y, x + width - 1, y + height - 1]
    if radius > 1:
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def _load_overlay_font(font_path: str | Path | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_overlay_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Sequence[int],
    font_path: str | Path | None,
) -> ImageFont.ImageFont:
    _, _, width, height = [int(value) for value in bbox]
    max_size = max(10, int(height * 0.70))
    min_size = 8
    for size in range(max_size, min_size - 1, -1):
        font = _load_overlay_font(font_path, size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        if text_w <= width * 0.78 and text_h <= height * 0.72:
            return font
    return _load_overlay_font(font_path, min_size)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Sequence[int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y, width, height = [int(value) for value in bbox]
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    px = x + max(0, (width - text_w) // 2) - text_bbox[0]
    py = y + max(0, (height - text_h) // 2) - text_bbox[1]
    draw.text((px, py), text, font=font, fill=fill)


def _draw_symbol(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[int],
    symbol: str,
    fill: tuple[int, int, int],
) -> None:
    x, y, width, height = [int(value) for value in bbox]
    margin = max(3, min(width, height) // 5)
    left = x + margin
    top = y + margin
    right = x + width - margin
    bottom = y + height - margin
    name = symbol.strip().lower()
    if "triangle" in name:
        cx = (left + right) / 2.0
        draw.polygon([(cx, top), (right, bottom), (left, bottom)], fill=fill)
    elif "moon" in name or "crescent" in name:
        draw.ellipse([left, top, right, bottom], fill=fill)
        cut = parse_color("black", default=(10, 10, 10))
        offset = max(2, (right - left) // 4)
        draw.ellipse([left + offset, top - 1, right + offset, bottom + 1], fill=cut)
    elif "star" in name:
        points = _star_points((left + right) / 2.0, (top + bottom) / 2.0, (right - left) / 2.0)
        draw.polygon(points, fill=fill)
    elif "square" in name:
        draw.rectangle([left, top, right, bottom], fill=fill)
    else:
        draw.ellipse([left, top, right, bottom], fill=fill)


def _star_points(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    import math

    points = []
    inner = radius * 0.45
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        r = radius if index % 2 == 0 else inner
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    return points


def _quoted_text_from_prompt(prompt: str) -> str:
    match = re.search(r"['\"]([^'\"]+)['\"]", str(prompt or ""))
    if match:
        return match.group(1)
    return ""


def _symbol_from_prompt(prompt: str) -> str:
    lowered = str(prompt or "").lower()
    for symbol in ("triangle", "moon", "crescent", "star", "circle", "square"):
        if symbol in lowered:
            return symbol
    return ""
