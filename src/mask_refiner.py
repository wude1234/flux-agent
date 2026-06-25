"""Mask refinement adapters for object-grounded local repair evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .local_editor import (
    count_mask_pixels,
    load_binary_mask,
    subtract_bboxes_from_mask,
    write_bbox_mask,
)


class MaskRefiner(Protocol):
    """Refine an existing bbox/point prompt into a binary mask.

    A refiner is not a detector. The caller must provide an image-grounded bbox
    or point prompt from VLM/layout/heuristics first.
    """

    def refine(
        self,
        image_path: str,
        target_name: str,
        bbox: Sequence[int],
        *,
        output_dir: str | Path,
        protected_bboxes: Sequence[Sequence[int]] = (),
        source: str = "bbox",
        points: Sequence[Sequence[int]] = (),
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class BBoxMaskRefiner:
    """Deterministic bbox-to-mask fallback used by tests and no-SAM runs."""

    prefix: str = "bbox_refine"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def refine(
        self,
        image_path: str,
        target_name: str,
        bbox: Sequence[int],
        *,
        output_dir: str | Path,
        protected_bboxes: Sequence[Sequence[int]] = (),
        source: str = "bbox",
        points: Sequence[Sequence[int]] = (),
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        image_size = _image_size(image_path, fallback=image_size)
        clean_bbox = _coerce_bbox(bbox, image_size=image_size)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_mask_path = output_dir / f"{self.prefix}_raw_mask_{self._counter:04d}.png"
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"

        write_bbox_mask(raw_mask_path, image_size=image_size, bbox=clean_bbox)
        raw_mask = load_binary_mask(raw_mask_path, image_size=image_size)
        protected = [_coerce_bbox(item, image_size=image_size) for item in protected_bboxes]
        protected_overlap = _protected_overlap(clean_bbox, protected)
        final_mask = subtract_bboxes_from_mask(raw_mask, protected) if protected else raw_mask
        final_mask.save(mask_path)
        selected_pixels = count_mask_pixels(final_mask)
        result = {
            "method": "bbox_fallback",
            "target_name": str(target_name or ""),
            "source_image": str(image_path),
            "source": str(source or "bbox"),
            "prompt_bbox": clean_bbox,
            "points": [list(point) for point in points],
            "protected_bboxes": [list(item) for item in protected],
            "protected_overlap": protected_overlap,
            "image_size": [image_size[0], image_size[1]],
            "mask_path": str(mask_path),
            "raw_mask_path": str(raw_mask_path),
            "selected_pixel_count": selected_pixels,
            "area_ratio": round(selected_pixels / max(1, image_size[0] * image_size[1]), 6),
            "geometry": _mask_geometry(clean_bbox, image_size, selected_pixels),
            "vram_note": "bbox fallback uses CPU and no model weights",
        }
        self.calls.append(deepcopy(result))
        self._counter += 1
        return result


@dataclass
class MockMaskRefiner(BBoxMaskRefiner):
    """Named mock refiner with the same behavior as the bbox fallback."""

    prefix: str = "mock_refine"

    def refine(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = super().refine(*args, **kwargs)
        result["method"] = "mock_mask_refiner"
        self.calls[-1]["method"] = "mock_mask_refiner"
        return result


@dataclass
class SamV1MaskRefiner:
    """Optional SAM v1 bbox-prompt refiner.

    Imports are lazy so the normal project does not depend on
    ``segment-anything`` or torch. This adapter should only be used after a
    feasibility smoke test records checkpoint path and VRAM.
    """

    checkpoint_path: str | Path
    model_type: str = "vit_l"
    device: str = "cuda"
    prefix: str = "sam_v1"
    allow_fallback: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    _predictor: Any = field(default=None, init=False, repr=False)
    _counter: int = 0

    def refine(
        self,
        image_path: str,
        target_name: str,
        bbox: Sequence[int],
        *,
        output_dir: str | Path,
        protected_bboxes: Sequence[Sequence[int]] = (),
        source: str = "bbox",
        points: Sequence[Sequence[int]] = (),
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        clean_image_size = _image_size(image_path, fallback=image_size)
        clean_bbox = _coerce_bbox(bbox, image_size=clean_image_size)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"{self.prefix}_mask_{self._counter:04d}.png"

        predictor = self._load_predictor()
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        predictor.set_image(np.array(image))
        x, y, width, height = clean_bbox
        box = np.array([x, y, x + width, y + height])
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        best_index = int(np.argmax(scores))
        mask_array = masks[best_index].astype("uint8") * 255
        mask = Image.fromarray(mask_array, mode="L")
        protected = [_coerce_bbox(item, image_size=clean_image_size) for item in protected_bboxes]
        protected_overlap = _protected_overlap(clean_bbox, protected)
        if protected:
            mask = subtract_bboxes_from_mask(mask, protected)
        mask.save(mask_path)
        selected_pixels = count_mask_pixels(mask)
        result = {
            "method": "sam_v1_bbox_prompt",
            "target_name": str(target_name or ""),
            "source_image": str(image_path),
            "source": str(source or "bbox"),
            "checkpoint_path": str(self.checkpoint_path),
            "model_type": self.model_type,
            "device": self.device,
            "prompt_bbox": clean_bbox,
            "points": [list(point) for point in points],
            "protected_bboxes": [list(item) for item in protected],
            "protected_overlap": protected_overlap,
            "image_size": [clean_image_size[0], clean_image_size[1]],
            "mask_path": str(mask_path),
            "selected_pixel_count": selected_pixels,
            "sam_score": float(scores[best_index]),
            "area_ratio": round(selected_pixels / max(1, clean_image_size[0] * clean_image_size[1]), 6),
            "geometry": _mask_geometry(clean_bbox, clean_image_size, selected_pixels),
            "vram_note": "SAM v1 backend; record observed VRAM in feasibility logs",
        }
        self.calls.append(deepcopy(result))
        self._counter += 1
        return result

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError(
                "segment-anything is not installed; use BBoxMaskRefiner or install "
                "SAM in a separate feasibility environment"
            ) from exc
        sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        return self._predictor


def refine_bbox_mask(
    refiner: MaskRefiner | None,
    image_path: str,
    target_name: str,
    bbox: Sequence[int],
    *,
    output_dir: str | Path,
    protected_bboxes: Sequence[Sequence[int]] = (),
    source: str = "bbox",
    points: Sequence[Sequence[int]] = (),
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run a mask refiner, falling back to deterministic bbox masks."""

    backend = refiner or BBoxMaskRefiner()
    allow_fallback = bool(getattr(backend, "allow_fallback", True))
    try:
        result = backend.refine(
            image_path,
            target_name,
            bbox,
            output_dir=output_dir,
            protected_bboxes=protected_bboxes,
            source=source,
            points=points,
            image_size=image_size,
        )
        result["fallback_used"] = isinstance(backend, BBoxMaskRefiner) and not isinstance(
            backend, SamV1MaskRefiner
        )
        return result
    except Exception as exc:
        if not allow_fallback:
            raise
        fallback = BBoxMaskRefiner(prefix="bbox_refine_fallback")
        result = fallback.refine(
            image_path,
            target_name,
            bbox,
            output_dir=output_dir,
            protected_bboxes=protected_bboxes,
            source=f"{source}:fallback_after_error",
            points=points,
            image_size=image_size,
        )
        result["fallback_used"] = True
        result["fallback_error"] = str(exc)
        return result


def constrain_refined_mask_to_prior(
    result: Mapping[str, Any],
    *,
    prior_mask_path: str | Path | None,
    output_dir: str | Path,
    protected_bboxes: Sequence[Sequence[int]] = (),
    prefix: str = "target_prior",
) -> dict[str, Any]:
    """Constrain a refined mask to an existing target-object prior mask.

    SAM and bbox refiners are mask refiners, not object detectors. The caller's
    target prior is the allowed edit region; the refiner may sharpen it, but it
    must not expand the edit to protected or unrelated content.
    """

    updated = deepcopy(dict(result))
    raw_mask_path = updated.get("mask_path")
    if not raw_mask_path or not prior_mask_path:
        updated["prior_constraint"] = {
            "applied": False,
            "reason": "missing refined mask or prior mask",
        }
        return updated

    prior_path = Path(prior_mask_path)
    if not prior_path.exists():
        updated["prior_constraint"] = {
            "applied": False,
            "reason": "prior mask path does not exist",
            "prior_mask_path": str(prior_path),
        }
        return updated

    image_size = _result_image_size(updated)
    raw_mask = load_binary_mask(raw_mask_path, image_size=image_size)
    prior_mask = load_binary_mask(prior_path, image_size=image_size)
    constrained_mask = _intersect_masks(raw_mask, prior_mask)
    protected = [_coerce_bbox(item, image_size=image_size) for item in protected_bboxes]
    if protected:
        constrained_mask = subtract_bboxes_from_mask(constrained_mask, protected)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_{Path(str(raw_mask_path)).stem}_mask.png"
    constrained_mask.save(output_path)

    raw_count = count_mask_pixels(raw_mask)
    prior_count = count_mask_pixels(prior_mask)
    constrained_count = count_mask_pixels(constrained_mask)
    image_area = max(1, image_size[0] * image_size[1])
    prompt_bbox = (
        updated.get("prompt_bbox")
        if _is_bbox_like(updated.get("prompt_bbox"))
        else [0, 0, image_size[0], image_size[1]]
    )
    updated["raw_mask_path"] = str(raw_mask_path)
    updated["mask_path"] = str(output_path)
    updated["selected_pixel_count"] = constrained_count
    updated["area_ratio"] = round(constrained_count / image_area, 6)
    updated["geometry"] = _mask_geometry(prompt_bbox, image_size, constrained_count)
    updated["prior_constraint"] = {
        "applied": True,
        "prior_mask_path": str(prior_path),
        "raw_mask_path": str(raw_mask_path),
        "constrained_mask_path": str(output_path),
        "protected_bboxes": [list(item) for item in protected],
        "raw_pixel_count": raw_count,
        "prior_pixel_count": prior_count,
        "constrained_pixel_count": constrained_count,
        "raw_outside_prior_pixel_count": max(0, raw_count - count_mask_pixels(_intersect_masks(raw_mask, prior_mask))),
        "prior_keep_ratio": round(constrained_count / max(1, prior_count), 6),
        "refined_keep_ratio": round(constrained_count / max(1, raw_count), 6),
        "note": (
            "final editable mask is refined_mask intersect target prior mask, "
            "then protected bboxes are subtracted"
        ),
    }
    return updated


def _image_size(
    image_path: str | Path,
    *,
    fallback: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if fallback is not None:
        width, height = int(fallback[0]), int(fallback[1])
        if width > 0 and height > 0:
            return width, height
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        return 1024, 1024


def _result_image_size(result: Mapping[str, Any]) -> tuple[int, int]:
    value = result.get("image_size")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        width, height = int(value[0]), int(value[1])
        if width > 0 and height > 0:
            return width, height
    source_image = result.get("source_image")
    if source_image:
        return _image_size(str(source_image))
    return 1024, 1024


def _coerce_bbox(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
) -> list[int]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    width, height = image_size
    x, y, box_w, box_h = [int(round(float(value))) for value in bbox]
    if box_w <= 0 or box_h <= 0:
        raise ValueError("bbox width and height must be positive")
    x0 = min(max(0, x), width - 1)
    y0 = min(max(0, y), height - 1)
    x1 = min(max(x0 + 1, x + box_w), width)
    y1 = min(max(y0 + 1, y + box_h), height)
    return [x0, y0, x1 - x0, y1 - y0]


def _is_bbox_like(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 4:
        return False
    try:
        _, _, width, height = [int(float(part)) for part in value]
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0


def _intersect_masks(mask_a: Any, mask_b: Any) -> Any:
    from PIL import ImageChops

    return ImageChops.multiply(mask_a.convert("L"), mask_b.convert("L")).point(
        lambda value: 255 if int(value) > 0 else 0
    )


def _protected_overlap(
    bbox: Sequence[int],
    protected_bboxes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    overlaps = []
    total = 0
    for protected in protected_bboxes:
        area = _bbox_overlap_area(bbox, protected)
        if area > 0:
            overlaps.append({"bbox": list(protected), "overlap_area": area})
            total += area
    bbox_area = max(1, _bbox_area(bbox))
    return {
        "overlap_area": total,
        "overlap_ratio": round(total / bbox_area, 6),
        "items": overlaps,
    }


def _mask_geometry(
    bbox: Sequence[int],
    image_size: tuple[int, int],
    selected_pixels: int,
) -> dict[str, Any]:
    bbox_area = max(1, _bbox_area(bbox))
    image_area = max(1, image_size[0] * image_size[1])
    return {
        "bbox": list(bbox),
        "bbox_area": bbox_area,
        "selected_pixel_count": int(selected_pixels),
        "mask_to_bbox_ratio": round(int(selected_pixels) / bbox_area, 6),
        "mask_to_image_ratio": round(int(selected_pixels) / image_area, 6),
    }


def _bbox_area(bbox: Sequence[int]) -> int:
    if len(bbox) != 4:
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_overlap_area(a: Sequence[int], b: Sequence[int]) -> int:
    ax, ay, aw, ah = [int(value) for value in a]
    bx, by, bw, bh = [int(value) for value in b]
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return max(0, x1 - x0) * max(0, y1 - y0)
