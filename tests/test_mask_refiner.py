from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_editor import count_mask_pixels, load_binary_mask
from src.mask_refiner import (
    BBoxMaskRefiner,
    MaskRefiner,
    MockMaskRefiner,
    SamV1MaskRefiner,
    constrain_refined_mask_to_prior,
    refine_bbox_mask,
)


def test_bbox_mask_refiner_writes_mask_and_subtracts_protected_bbox(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (64, 32), (0, 0, 0)).save(image_path)
    refiner = BBoxMaskRefiner()

    result = refiner.refine(
        str(image_path),
        "umbrella",
        [10, 5, 30, 20],
        output_dir=tmp_path,
        protected_bboxes=[[20, 10, 10, 10]],
        source="vlm_bbox",
    )

    mask = load_binary_mask(result["mask_path"], image_size=(64, 32))
    assert Path(result["mask_path"]).exists()
    assert result["method"] == "bbox_fallback"
    assert result["prompt_bbox"] == [10, 5, 30, 20]
    assert result["protected_overlap"]["overlap_area"] == 100
    assert count_mask_pixels(mask) == 500
    assert result["geometry"]["mask_to_bbox_ratio"] == 0.833333
    assert result["vram_note"] == "bbox fallback uses CPU and no model weights"


def test_mock_mask_refiner_uses_same_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (20, 20), (0, 0, 0)).save(image_path)
    refiner = MockMaskRefiner()

    result = refiner.refine(
        str(image_path),
        "cat",
        [2, 3, 8, 7],
        output_dir=tmp_path,
    )

    assert result["method"] == "mock_mask_refiner"
    assert refiner.calls[0]["target_name"] == "cat"
    assert Path(result["raw_mask_path"]).exists()


def test_constrain_refined_mask_to_prior_subtracts_protected_region(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (40, 40), (0, 0, 0)).save(image_path)
    raw_result = BBoxMaskRefiner().refine(
        str(image_path),
        "target object",
        [5, 5, 30, 25],
        output_dir=tmp_path / "raw",
    )
    prior_path = tmp_path / "target_prior.png"
    prior = Image.new("L", (40, 40), 0)
    for y in range(5, 18):
        for x in range(5, 35):
            prior.putpixel((x, y), 255)
    prior.save(prior_path)

    constrained = constrain_refined_mask_to_prior(
        raw_result,
        prior_mask_path=prior_path,
        output_dir=tmp_path / "constrained",
        protected_bboxes=[[15, 8, 10, 8]],
    )
    mask = load_binary_mask(constrained["mask_path"], image_size=(40, 40))

    assert Path(constrained["raw_mask_path"]).exists()
    assert Path(constrained["mask_path"]).exists()
    assert constrained["prior_constraint"]["applied"] is True
    assert constrained["prior_constraint"]["prior_pixel_count"] == 390
    assert constrained["selected_pixel_count"] == 310
    assert mask.getpixel((10, 10)) == 255
    assert mask.getpixel((20, 10)) == 0
    assert mask.getpixel((10, 25)) == 0


def test_refine_bbox_mask_falls_back_after_backend_error(tmp_path: Path) -> None:
    class FailingRefiner:
        def refine(self, *args, **kwargs):
            raise RuntimeError("backend unavailable")

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)

    result = refine_bbox_mask(
        FailingRefiner(),
        str(image_path),
        "dog",
        [1, 1, 6, 6],
        output_dir=tmp_path,
    )

    assert result["fallback_used"] is True
    assert "backend unavailable" in result["fallback_error"]
    assert result["method"] == "bbox_fallback"


def test_refine_bbox_mask_does_not_fallback_for_explicit_sam_request(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
    refiner = SamV1MaskRefiner(
        checkpoint_path=tmp_path / "missing_sam.pth",
        model_type="vit_l",
        device="cpu",
    )

    try:
        refine_bbox_mask(
            refiner,
            str(image_path),
            "dog",
            [1, 1, 6, 6],
            output_dir=tmp_path,
        )
    except FileNotFoundError as exc:
        assert "SAM checkpoint does not exist" in str(exc)
    else:
        raise AssertionError("explicit SAM requests must not silently fallback")
