"""Optional OCR verifier for deterministic text repairs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image, ImageOps


class OCRBackend(Protocol):
    def __call__(self, image: Image.Image) -> Any:
        ...


def verify_text_in_bbox(
    image_path: str | Path,
    *,
    expected_text: str,
    bbox: Sequence[int],
    backend: OCRBackend | None = None,
    crop_output_path: str | Path | None = None,
    padding: int = 8,
    pass_threshold: float = 0.72,
) -> dict[str, Any]:
    """Run OCR on a bbox crop and compare it to the expected text."""

    expected = str(expected_text or "").strip()
    if not expected:
        return {
            "available": False,
            "passed": None,
            "error": "missing expected_text",
            "expected": expected,
            "items": [],
        }
    image = Image.open(image_path).convert("RGB")
    crop = crop_text_region(image, bbox, padding=padding)
    crop_path = None
    if crop_output_path is not None:
        crop_path = Path(crop_output_path)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
    try:
        ocr = backend or _rapid_ocr()
    except Exception as exc:
        return {
            "available": False,
            "passed": None,
            "error": str(exc),
            "expected": expected,
            "bbox": [int(value) for value in bbox],
            "crop_path": str(crop_path) if crop_path else None,
            "items": [],
        }
    raw_result = ocr(crop)
    recognized, confidences = parse_ocr_result(raw_result)
    similarity = text_similarity(normalize_text(expected), normalize_text(recognized))
    passed = similarity >= float(pass_threshold)
    return {
        "available": True,
        "passed": passed,
        "expected": expected,
        "recognized": recognized,
        "similarity": similarity,
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "bbox": [int(value) for value in bbox],
        "pass_threshold": float(pass_threshold),
        "crop_path": str(crop_path) if crop_path else None,
        "items": [
            {
                "content": expected,
                "recognized": recognized,
                "similarity": similarity,
                "pass": passed,
            }
        ],
    }


def crop_text_region(
    image: Image.Image,
    bbox: Sequence[int],
    *,
    padding: int = 8,
) -> Image.Image:
    x, y, width, height = [int(value) for value in bbox]
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.width, x + width + padding)
    bottom = min(image.height, y + height + padding)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = 2 if max(crop.size) < 900 else 1
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(crop)


def parse_ocr_result(raw_result: Any) -> tuple[str, list[float]]:
    if isinstance(raw_result, tuple) and raw_result:
        raw_result = raw_result[0]
    if isinstance(raw_result, dict):
        raw_result = raw_result.get("result") or raw_result.get("items") or []
    recognized_parts: list[str] = []
    confidences: list[float] = []
    if isinstance(raw_result, list):
        for line in raw_result:
            text = None
            confidence = None
            if isinstance(line, dict):
                text = line.get("text") or line.get("recognized")
                confidence = line.get("confidence") or line.get("score")
            elif isinstance(line, (list, tuple)):
                if len(line) >= 2:
                    text = line[1]
                if len(line) >= 3:
                    confidence = line[2]
            elif isinstance(line, str):
                text = line
            if text is not None:
                recognized_parts.append(str(text))
            if confidence is not None:
                try:
                    confidences.append(float(confidence))
                except (TypeError, ValueError):
                    pass
    elif isinstance(raw_result, str):
        recognized_parts.append(raw_result)
    return "".join(recognized_parts), confidences


def normalize_text(value: str) -> str:
    return "".join(str(value).split()).lower()


def text_similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return round(ratio(a, b) / 100.0, 4)
    except Exception:
        if not a and not b:
            return 1.0
        if a == b:
            return 1.0
        return _sequence_similarity(a, b)


def _sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    distance = previous[-1]
    return round(1.0 - distance / max(len(a), len(b)), 4)


@lru_cache(maxsize=1)
def _rapid_ocr() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()
