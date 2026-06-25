from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ocr_verifier
from src.ocr_verifier import (
    crop_text_region,
    normalize_text,
    parse_ocr_result,
    verify_text_in_bbox,
)


def test_verify_text_in_bbox_passes_with_fake_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (20, 20, 20)).save(image_path)
    crop_path = tmp_path / "crop.png"

    def fake_backend(image):
        assert image.width > 0
        return ([["bbox", "NO", 0.91]], None)

    result = verify_text_in_bbox(
        image_path,
        expected_text="NO",
        bbox=[10, 8, 40, 22],
        backend=fake_backend,
        crop_output_path=crop_path,
    )

    assert result["available"] is True
    assert result["passed"] is True
    assert result["recognized"] == "NO"
    assert result["mean_confidence"] == 0.91
    assert result["crop_path"] == str(crop_path)
    assert crop_path.exists()


def test_verify_text_in_bbox_flags_mismatch_with_fake_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (20, 20, 20)).save(image_path)

    def fake_backend(_image):
        return ([["bbox", "GO", 0.88]], None)

    result = verify_text_in_bbox(
        image_path,
        expected_text="NO",
        bbox=[10, 8, 40, 22],
        backend=fake_backend,
    )

    assert result["available"] is True
    assert result["passed"] is False
    assert result["expected"] == "NO"
    assert result["recognized"] == "GO"
    assert result["similarity"] < result["pass_threshold"]


def test_verify_text_in_bbox_missing_ocr_dependency_is_nonfatal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (20, 20, 20)).save(image_path)
    crop_path = tmp_path / "crop.png"

    def missing_backend():
        raise ImportError("rapidocr_onnxruntime is not installed")

    monkeypatch.setattr(ocr_verifier, "_rapid_ocr", missing_backend)

    result = verify_text_in_bbox(
        image_path,
        expected_text="NO",
        bbox=[10, 8, 40, 22],
        crop_output_path=crop_path,
    )

    assert result["available"] is False
    assert result["passed"] is None
    assert "rapidocr_onnxruntime" in result["error"]
    assert result["crop_path"] == str(crop_path)
    assert crop_path.exists()


def test_parse_ocr_result_handles_common_shapes() -> None:
    assert parse_ocr_result(([["box", "NO", 0.9]], None)) == ("NO", [0.9])
    assert parse_ocr_result({"items": [{"text": "N", "score": 0.8}, {"text": "O"}]}) == (
        "NO",
        [0.8],
    )
    assert parse_ocr_result(["N", "O"]) == ("NO", [])
    assert parse_ocr_result("NO") == ("NO", [])


def test_crop_text_region_clamps_and_upscales_small_crop() -> None:
    image = Image.new("RGB", (32, 24), (100, 100, 100))

    crop = crop_text_region(image, [-4, -3, 12, 8], padding=4)

    assert crop.width > 0
    assert crop.height > 0
    assert max(crop.size) < 900


def test_normalize_text_removes_spacing_and_case() -> None:
    assert normalize_text(" N O ") == "no"
