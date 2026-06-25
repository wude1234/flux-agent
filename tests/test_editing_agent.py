from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.editing_agent import (
    EfficientRepairAgent,
    EfficientRepairRequest,
    GroundedSAM2PowerPaintEditingAgent,
    GroundedSAM2SubprocessMasker,
    MaskGeneratingInpaintEditor,
    bbox_from_mask,
    dilate_mask_path,
    prepare_masked_inpaint_region,
    route_repair_kind,
)
from src.local_editor import InpaintRegion, MockInpaintEditor, count_mask_pixels, load_binary_mask


class FailingMasker:
    def generate(self, *, image_path, text, output_dir):
        del image_path, output_dir
        return {
            "ok": False,
            "method": "grounded_sam2_subprocess",
            "error": f"cannot segment {text}",
        }


class FixedMasker:
    def __init__(self, mask_path: Path) -> None:
        self.mask_path = mask_path

    def generate(self, *, image_path, text, output_dir):
        del image_path, text, output_dir
        return {
            "ok": True,
            "method": "grounded_sam2_subprocess",
            "mask_path": str(self.mask_path),
        }


class ExplodingMasker:
    def generate(self, *, image_path, text, output_dir):
        del image_path, text, output_dir
        raise AssertionError("SAM2 should not be called for new-object bbox insertion")


def test_dilate_mask_path_expands_binary_mask(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.png"
    raw = Image.new("L", (16, 16), 0)
    raw.putpixel((8, 8), 255)
    raw.save(raw_path)

    result = dilate_mask_path(
        raw_path,
        tmp_path / "dilated.png",
        image_size=(16, 16),
        kernel_size=5,
    )
    dilated = load_binary_mask(result["output_mask_path"], image_size=(16, 16))

    assert result["effective_kernel_size"] == 5
    assert count_mask_pixels(dilated) == 25
    assert bbox_from_mask(result["output_mask_path"], image_size=(16, 16)) == [6, 6, 5, 5]


def test_prepare_masked_region_falls_back_to_bbox_after_sam_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (40, 20), (0, 0, 0)).save(image_path)
    region = InpaintRegion(
        name="green suitcase",
        bbox=[10, 5, 12, 8],
        prompt="add a red screen over the suitcase",
        canvas_size=[40, 20],
    )

    masked_region, record = prepare_masked_inpaint_region(
        image_path=image_path,
        region=region,
        output_dir=tmp_path / "mask",
        mask_generator=FailingMasker(),
        mask_mode="auto",
        mask_text="green suitcase",
        dilation_kernel_size=1,
    )

    assert record["mask_source"] == "bbox_fallback"
    assert "cannot segment green suitcase" in record["fallback_reason"]
    assert Path(masked_region.mask_path).exists()
    assert masked_region.bbox == [10, 5, 12, 8]
    mask = load_binary_mask(masked_region.mask_path, image_size=(40, 20))
    assert count_mask_pixels(mask) == 96


def test_prepare_masked_region_uses_grounded_mask_and_dilates(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (30, 30), (0, 0, 0)).save(image_path)
    sam_mask_path = tmp_path / "sam.png"
    sam_mask = Image.new("L", (30, 30), 0)
    for y in range(10, 15):
        for x in range(10, 15):
            sam_mask.putpixel((x, y), 255)
    sam_mask.save(sam_mask_path)
    region = InpaintRegion(
        name="suitcase",
        bbox=[0, 0, 30, 30],
        prompt="add a red screen",
        canvas_size=[30, 30],
    )

    masked_region, record = prepare_masked_inpaint_region(
        image_path=image_path,
        region=region,
        output_dir=tmp_path / "mask",
        mask_generator=FixedMasker(sam_mask_path),
        mask_mode="auto",
        mask_text="suitcase",
        dilation_kernel_size=3,
    )

    assert record["mask_source"] == "grounded_sam2"
    assert record["raw_pixel_count"] == 25
    assert record["dilated_pixel_count"] == 49
    assert masked_region.bbox == [9, 9, 7, 7]


def test_mask_generating_editor_passes_dilated_mask_to_base_editor(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (20, 20), (0, 0, 0)).save(image_path)
    base_editor = MockInpaintEditor(prefix="base")
    wrapper = MaskGeneratingInpaintEditor(
        base_editor=base_editor,
        mask_generator=FailingMasker(),
        mask_mode="auto",
        mask_text="green suitcase",
        dilation_kernel_size=1,
    )
    region = InpaintRegion(
        name="suitcase",
        bbox=[2, 3, 5, 6],
        prompt="repaint the suitcase green",
        canvas_size=[20, 20],
    )

    result = wrapper.edit(str(image_path), region, tmp_path / "out")

    assert result["mask_agent"]["mask_source"] == "bbox_fallback"
    assert base_editor.calls[0]["region"]["mask_path"] == result["mask_agent"]["dilated_mask_path"]
    assert Path(result["mask_path"]).exists()


def test_mask_generating_editor_auto_uses_bbox_for_new_occluder(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
    base_editor = MockInpaintEditor(prefix="base")
    wrapper = MaskGeneratingInpaintEditor(
        base_editor=base_editor,
        mask_generator=ExplodingMasker(),
        mask_mode="auto",
        mask_text="red screen",
        dilation_kernel_size=1,
    )
    region = InpaintRegion(
        name="red screen",
        bbox=[4, 18, 20, 8],
        prompt="add a flat red screen covering the lower half of the suitcase",
        reason="typed occlusion repair",
        canvas_size=[32, 32],
    )

    result = wrapper.edit(str(image_path), region, tmp_path / "out")

    assert result["mask_agent"]["mask_source"] == "bbox"
    assert result["mask_agent"]["requested_mask_mode"] == "auto"
    assert "new-object/occlusion" in result["mask_agent"]["auto_bbox_reason"]
    assert base_editor.calls[0]["region"]["bbox"] == [4, 18, 20, 8]


def test_grounded_sam2_subprocess_masker_parses_last_json(monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    grounded_dir = tmp_path / "Grounded_SAM2"
    grounded_dir.mkdir()
    script_path = tmp_path / "wrapper.py"
    script_path.write_text("# wrapper", encoding="utf-8")
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)
    mask_path = tmp_path / "mask.png"
    Image.new("L", (4, 4), 255).save(mask_path)
    calls = []

    def fake_run(command, cwd, capture_output, text, timeout, env):
        del capture_output, text, timeout
        calls.append((command, cwd, env))
        stdout = "loading model\n{\"ok\": true, \"mask_path\": \"" + str(mask_path) + "\"}\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("src.editing_agent.subprocess.run", fake_run)
    masker = GroundedSAM2SubprocessMasker(
        python=python,
        grounded_sam2_dir=grounded_dir,
        script_path=script_path,
        cuda_visible_devices="1",
    )

    result = masker.generate(image_path=image_path, text="green suitcase", output_dir=tmp_path / "out")

    assert result["ok"] is True
    assert result["mask_path"] == str(mask_path)
    assert calls[0][1] == str(grounded_dir)
    assert calls[0][2]["CUDA_VISIBLE_DEVICES"] == "1"
    command = calls[0][0]
    assert Path(command[command.index("--image") + 1]).is_absolute()
    assert Path(command[command.index("--output-dir") + 1]).is_absolute()


def test_grounded_sam2_subprocess_masker_returns_error_on_timeout(monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    grounded_dir = tmp_path / "Grounded_SAM2"
    grounded_dir.mkdir()
    script_path = tmp_path / "wrapper.py"
    script_path.write_text("# wrapper", encoding="utf-8")
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sam2", timeout=5, output="slow", stderr="busy")

    monkeypatch.setattr("src.editing_agent.subprocess.run", fake_run)
    masker = GroundedSAM2SubprocessMasker(
        python=python,
        grounded_sam2_dir=grounded_dir,
        script_path=script_path,
        timeout_seconds=5,
    )

    result = masker.generate(image_path=image_path, text="green suitcase", output_dir=tmp_path / "out")

    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_efficient_text_overlay_does_not_call_inpaint(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (80, 120, 180)).save(image_path)
    base_editor = MockInpaintEditor(prefix="should_not_run")
    inpaint_agent = GroundedSAM2PowerPaintEditingAgent(editor=base_editor)
    agent = EfficientRepairAgent(inpaint_agent=inpaint_agent)
    calls = []

    def fake_verify_text_in_bbox(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "available": True,
            "passed": True,
            "expected": kwargs["expected_text"],
            "recognized": kwargs["expected_text"],
            "similarity": 1.0,
            "crop_path": str(kwargs["crop_output_path"]),
            "items": [],
        }

    monkeypatch.setattr("src.editing_agent.verify_text_in_bbox", fake_verify_text_in_bbox)

    result = agent.repair(
        EfficientRepairRequest(
            repair_kind="text_overlay",
            image_path=image_path,
            output_dir=tmp_path / "text",
            bbox=[10, 8, 42, 24],
            text="NO",
            fill_color="black",
            text_color="yellow",
            target_object="top sign",
        )
    )

    assert result["ok"] is True
    assert result["route"] == "text_overlay"
    assert result["gpu_used"] is False
    assert result["sam2_used"] is False
    assert result["powerpaint_used"] is False
    assert base_editor.calls == []
    assert Path(result["edited_image"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["ocr_verification"]["passed"] is True
    assert result["ocr_verification"]["crop_path"].endswith("text_overlay_ocr_crop.png")
    assert calls[0][1]["expected_text"] == "NO"


def test_efficient_symbol_overlay_does_not_call_inpaint(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (64, 64), (220, 180, 80)).save(image_path)
    base_editor = MockInpaintEditor(prefix="should_not_run")
    agent = EfficientRepairAgent(
        inpaint_agent=GroundedSAM2PowerPaintEditingAgent(editor=base_editor)
    )

    result = agent.repair(
        EfficientRepairRequest(
            repair_kind="symbol_overlay",
            image_path=image_path,
            output_dir=tmp_path / "symbol",
            bbox=[12, 12, 32, 32],
            symbol="triangle",
            fill_color="purple",
            text_color="white",
            target_object="folder front",
        )
    )

    assert result["ok"] is True
    assert result["route"] == "symbol_overlay"
    assert result["symbol"] == "triangle"
    assert base_editor.calls == []
    assert Path(result["edited_image"]).exists()


def test_efficient_shape_overlay_for_planar_occluder_is_cpu_only(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (40, 140, 90)).save(image_path)
    base_editor = MockInpaintEditor(prefix="should_not_run")
    agent = EfficientRepairAgent(
        inpaint_agent=GroundedSAM2PowerPaintEditingAgent(editor=base_editor)
    )

    result = agent.repair(
        EfficientRepairRequest(
            repair_kind="shape_overlay",
            image_path=image_path,
            output_dir=tmp_path / "shape",
            bbox=[10, 34, 50, 18],
            target_object="flat opaque panel",
            prompt="Add a flat opaque blue panel covering the bottom portion.",
            fill_color="blue",
        )
    )

    assert result["ok"] is True
    assert result["route"] == "shape_overlay"
    assert result["gpu_used"] is False
    assert result["sam2_used"] is False
    assert result["powerpaint_used"] is False
    assert base_editor.calls == []
    assert Path(result["edited_image"]).exists()
    assert result["fill_color"] == [42, 92, 196]


def test_efficient_bbox_shape_inpaint_forces_bbox_without_sam(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (50, 50), (0, 0, 0)).save(image_path)
    base_editor = MockInpaintEditor(prefix="bbox_insert")
    agent = EfficientRepairAgent(
        inpaint_agent=GroundedSAM2PowerPaintEditingAgent(
            editor=base_editor,
            mask_generator=FailingMasker(),
            mask_mode="grounded-sam2",
            allow_bbox_fallback=False,
        )
    )

    result = agent.repair(
        EfficientRepairRequest(
            repair_kind="bbox_shape_inpaint",
            image_path=image_path,
            output_dir=tmp_path / "bbox",
            bbox=[5, 20, 30, 15],
            target_object="red screen",
            prompt="add a flat red screen",
        )
    )

    assert result["ok"] is True
    assert result["route"] == "bbox_shape_inpaint"
    assert result["sam2_used"] is False
    assert result["gpu_used"] is False
    assert result["powerpaint_used"] is False
    assert result["edited_image"].endswith("bbox_insert_image_0000.txt")
    assert result["mask_path"].endswith("bbox_insert_mask_0000.png")
    assert base_editor.calls[0]["region"]["bbox"] == [5, 20, 30, 15]
    assert base_editor.calls[0]["region"]["mask_path"].endswith("dilated_mask.png")
    assert result["result"]["mask_agent"]["mask_source"] == "bbox"


def test_efficient_low_editability_routes_to_regeneration_without_gpu(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
    base_editor = MockInpaintEditor(prefix="should_not_run")
    agent = EfficientRepairAgent(
        inpaint_agent=GroundedSAM2PowerPaintEditingAgent(editor=base_editor)
    )

    result = agent.repair(
        EfficientRepairRequest(
            repair_kind="layout_regenerate",
            image_path=image_path,
            output_dir=tmp_path / "regen",
            bbox=[0, 0, 10, 10],
            reason="left/right relation reversed",
        )
    )

    assert result["ok"] is False
    assert result["route"] == "layout_regenerate"
    assert result["gpu_used"] is False
    assert base_editor.calls == []


def test_route_repair_kind_prefers_cheapest_tool() -> None:
    assert route_repair_kind({"typed_route": "text_overlay"}) == "text_overlay"
    assert route_repair_kind({"typed_route": "exact_text_overlay"}) == "text_overlay"
    assert route_repair_kind({"typed_route": "forbidden_symbol_removal"}) == "existing_object_inpaint"
    assert route_repair_kind({"typed_route": "forbidden_object_removal"}) == "existing_object_inpaint"
    assert route_repair_kind({"typed_route": "single_attribute_patch"}) == "existing_object_inpaint"
    assert route_repair_kind({"typed_route": "relation_contact_repair"}) == "existing_object_inpaint"
    assert route_repair_kind({"typed_route": "count_aware_regeneration"}) == "count_rerank"
    assert route_repair_kind({"typed_route": "layout_guided_regeneration"}) == "layout_regenerate"
    assert route_repair_kind({"typed_route": "multi_constraint_decompose"}) == "layout_regenerate"
    assert (
        route_repair_kind(
            {"primary_action": "regenerate", "reason": "too many objects"},
            "A sign displays the exact text 'NO'.",
        )
        == "layout_regenerate"
    )
    assert (
        route_repair_kind(
            {
                "typed_route": "occlusion_object_insertion",
                "primary_action": "object_insertion",
                "target_object": "opaque panel",
                "target_attribute": "occlusion",
            }
        )
        == "shape_overlay"
    )
    assert (
        route_repair_kind(
            {
                "typed_route": "occlusion_object_insertion",
                "primary_action": "object_insertion",
                "target_object": "toy car",
                "target_attribute": "occlusion",
            }
        )
        == "bbox_shape_inpaint"
    )
    assert (
        route_repair_kind(
            {
                "primary_action": "regenerate",
                "target_attribute": "spatial_relation",
            }
        )
        == "layout_regenerate"
    )
    assert (
        route_repair_kind(
            {
                "primary_action": "recolor",
                "target_attribute": "color",
            }
        )
        == "existing_object_inpaint"
    )
    assert route_repair_kind({"primary_action": "none"}, "exact text 'NO'") == "text_overlay"
