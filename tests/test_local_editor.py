from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_editor import (
    ColorRecolorEditor,
    DiffusersInpaintEditor,
    InpaintRegion,
    MockInpaintEditor,
    PowerPaintSubprocessEditor,
    SubprocessInpaintEditor,
    build_color_selection_mask,
    count_mask_pixels,
    detect_color_region_from_layout,
    expand_bbox,
    keep_largest_mask_component,
    mask_components,
    measure_color_coverage,
    normalize_source_color_mode,
    pad_bbox,
    plan_inpaint_region_from_layout,
    parse_rgb_color,
    recolor_image_with_mask,
    scale_bbox,
    select_component_for_bbox,
    write_or_copy_region_mask,
    write_bbox_mask,
)


def test_expand_bbox_clamps_to_image_bounds() -> None:
    assert expand_bbox([10, 20, 100, 50], image_size=(200, 120), expand=0.1) == [
        0,
        15,
        120,
        60,
    ]


def test_write_bbox_mask_creates_white_region(tmp_path: Path) -> None:
    mask_path = write_bbox_mask(
        tmp_path / "mask.png",
        image_size=(32, 32),
        bbox=[8, 10, 12, 6],
    )

    mask = Image.open(mask_path)

    assert mask.mode == "L"
    assert mask.getpixel((1, 1)) == 0
    assert mask.getpixel((10, 12)) == 255


def test_write_or_copy_region_mask_prefers_precomputed_mask(tmp_path: Path) -> None:
    precomputed = tmp_path / "sam_mask.png"
    mask = Image.new("L", (8, 8), 0)
    for y in range(2, 5):
        for x in range(3, 7):
            mask.putpixel((x, y), 255)
    mask.save(precomputed)

    output = write_or_copy_region_mask(
        tmp_path / "out_mask.png",
        source_mask_path=precomputed,
        image_size=(16, 16),
        fallback_bbox=[0, 0, 16, 16],
    )

    copied = Image.open(output).convert("L")
    assert copied.size == (16, 16)
    assert copied.getpixel((8, 6)) == 255
    assert copied.getpixel((1, 1)) == 0


def test_scale_bbox_maps_layout_canvas_to_image_size() -> None:
    assert scale_bbox(
        [512, 256, 256, 512],
        from_size=(1024, 1024),
        to_size=(768, 512),
    ) == [384, 128, 192, 256]


def test_pad_bbox_clamps_to_image_bounds() -> None:
    assert pad_bbox([2, 3, 5, 6], image_size=(12, 12), padding=4) == [0, 0, 11, 12]


def test_plan_inpaint_region_from_layout_finds_target_object() -> None:
    layout_context = {
        "layout": {
            "canvas_size": [1024, 1024],
            "objects": [
                {
                    "name": "blue umbrella",
                    "bbox": [420, 180, 320, 360],
                },
                {
                    "name": "small red robot",
                    "bbox": [460, 340, 240, 380],
                },
            ],
        }
    }

    region = plan_inpaint_region_from_layout(
        layout_context,
        "umbrella",
        prompt="vivid cobalt blue umbrella canopy with dark handle",
        negative_prompt="red umbrella",
    )

    assert region.name == "blue umbrella"
    assert region.bbox == [394, 151, 372, 418]
    assert region.canvas_size == [1024, 1024]
    assert "cobalt blue" in region.prompt
    assert region.negative_prompt == "red umbrella"


def test_mock_inpaint_editor_writes_artifacts(tmp_path: Path) -> None:
    region = plan_inpaint_region_from_layout(
        {
            "canvas_size": [1024, 1024],
            "objects": [{"name": "blue umbrella", "bbox": [420, 180, 320, 360]}],
        },
        "umbrella",
        prompt="vivid cobalt blue umbrella canopy",
    )
    editor = MockInpaintEditor()

    result = editor.edit("source.png", region, tmp_path)

    assert Path(result["edited_image"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["mode"] == "mock"
    assert editor.calls[0]["region"]["name"] == "blue umbrella"


def test_color_selection_mask_selects_only_requested_color() -> None:
    image = Image.new("RGB", (8, 4), (0, 0, 0))
    image.putpixel((2, 1), (220, 20, 20))
    image.putpixel((4, 1), (20, 20, 220))

    mask, selected_count = build_color_selection_mask(
        image,
        bbox=[0, 0, 8, 4],
        source_color="red",
    )

    assert selected_count == 1
    assert mask.getpixel((2, 1)) == 255
    assert mask.getpixel((4, 1)) == 0


def test_low_saturation_mask_selects_transparent_target_region() -> None:
    image = Image.new("RGB", (12, 6), (20, 80, 20))
    for y in range(1, 4):
        for x in range(2, 10):
            image.putpixel((x, y), (180, 185, 185))
    image.putpixel((3, 4), (220, 20, 20))

    mask, selected_count = build_color_selection_mask(
        image,
        bbox=[0, 0, 12, 6],
        source_color="transparent",
        saturation_threshold=80,
        value_threshold=40,
    )

    assert normalize_source_color_mode("silver") == "low_saturation"
    assert selected_count == 24
    assert mask.getpixel((4, 2)) == 255
    assert mask.getpixel((3, 4)) == 0


def test_color_selection_mask_supports_brown_source_color() -> None:
    image = Image.new("RGB", (8, 4), (0, 0, 0))
    image.putpixel((2, 1), (150, 82, 28))
    image.putpixel((4, 1), (20, 20, 220))

    mask, selected_count = build_color_selection_mask(
        image,
        bbox=[0, 0, 8, 4],
        source_color="brown",
    )

    assert selected_count == 1
    assert mask.getpixel((2, 1)) == 255
    assert mask.getpixel((4, 1)) == 0


def test_recolor_image_with_mask_preserves_unmasked_pixels() -> None:
    image = Image.new("RGB", (3, 1), (10, 10, 10))
    image.putpixel((1, 0), (220, 30, 30))
    mask = Image.new("L", (3, 1), 0)
    mask.putpixel((1, 0), 255)

    result = recolor_image_with_mask(
        image,
        mask,
        target_color=parse_rgb_color("#1d63d9"),
        feather_radius=0,
    )

    assert result.getpixel((0, 0)) == (10, 10, 10)
    red, green, blue = result.getpixel((1, 0))
    assert blue > red
    assert blue > green


def test_recolor_image_with_mask_can_lift_dark_source_color() -> None:
    image = Image.new("RGB", (2, 1), (5, 5, 5))
    mask = Image.new("L", (2, 1), 0)
    mask.putpixel((0, 0), 255)

    no_boost = recolor_image_with_mask(
        image,
        mask,
        target_color=parse_rgb_color("#1d63d9"),
        feather_radius=0,
    )
    boosted = recolor_image_with_mask(
        image,
        mask,
        target_color=parse_rgb_color("#1d63d9"),
        feather_radius=0,
        value_floor=120,
    )

    assert max(no_boost.getpixel((0, 0))) < 20
    assert boosted.getpixel((0, 0))[2] > boosted.getpixel((0, 0))[0]
    assert max(boosted.getpixel((0, 0))) >= 110
    assert boosted.getpixel((1, 0)) == (5, 5, 5)


def test_keep_largest_mask_component_drops_disconnected_specks() -> None:
    mask = Image.new("L", (8, 4), 0)
    for y in range(1, 3):
        for x in range(1, 4):
            mask.putpixel((x, y), 255)
    mask.putpixel((7, 3), 255)

    result, count = keep_largest_mask_component(mask)

    assert count == 6
    assert result.getpixel((2, 1)) == 255
    assert result.getpixel((7, 3)) == 0


def test_mask_components_and_select_component_for_bbox_layout_overlap() -> None:
    mask = Image.new("L", (30, 20), 0)
    for y in range(5, 10):
        for x in range(8, 18):
            mask.putpixel((x, y), 255)
    for y in range(2, 5):
        for x in range(24, 28):
            mask.putpixel((x, y), 255)

    components = mask_components(mask, min_area=1)
    selected = select_component_for_bbox(
        components,
        preferred_bbox=[7, 4, 12, 8],
        strategy="layout_overlap",
    )

    assert len(components) == 2
    assert components[0]["area"] == 50
    assert components[0]["touches_image_border"] is False
    assert selected["bbox"] == [8, 5, 10, 5]


def test_select_component_for_bbox_largest_strategy() -> None:
    components = [
        {"bbox": [0, 0, 4, 4], "area": 16},
        {"bbox": [20, 20, 12, 8], "area": 96},
    ]

    selected = select_component_for_bbox(
        components,
        preferred_bbox=[0, 0, 5, 5],
        strategy="largest",
    )

    assert selected["bbox"] == [20, 20, 12, 8]
    assert selected["selection_strategy"] == "largest"


def test_detect_color_region_from_layout_uses_actual_image_component(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    for y in range(42, 66):
        for x in range(30, 70):
            image.putpixel((x, y), (220, 20, 20))
    for y in range(5, 20):
        for x in range(75, 95):
            image.putpixel((x, y), (220, 20, 20))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [100, 100],
            "objects": [{"name": "red umbrella", "bbox": [28, 38, 44, 30]}],
        }
    }

    region, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "umbrella",
        prompt="make umbrella blue",
        source_color="red",
        component_padding=3,
        min_component_area=10,
        selection_strategy="largest",
        prefer_object_mask=False,
    )

    assert region.canvas_size == [100, 100]
    assert region.bbox == [27, 39, 46, 30]
    assert detection["component_count"] == 2
    assert detection["selected_component"]["bbox"] == [30, 42, 40, 24]


def test_detect_color_region_can_constrain_target_and_subtract_other_objects(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    for y in range(20, 50):
        for x in range(20, 80):
            image.putpixel((x, y), (220, 20, 20))
    for y in range(45, 85):
        for x in range(38, 62):
            image.putpixel((x, y), (220, 20, 20))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [100, 100],
            "objects": [
                {"name": "target canopy", "bbox": [20, 20, 60, 60]},
                {"name": "other red object", "bbox": [38, 45, 24, 40]},
            ],
        }
    }

    region, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "target canopy",
        prompt="make target blue",
        source_color="red",
        target_region="upper",
        subtract_other_objects=True,
        search_expand=0.0,
        component_padding=0,
        min_component_area=10,
        prefer_object_mask=False,
    )
    editor = ColorRecolorEditor(
        target_color="#1d63d9",
        source_color="red",
        keep_largest_component=False,
        feather_radius=0,
        exclude_bboxes=detection["subtract_bboxes"],
    )
    result = editor.edit(str(image_path), region, tmp_path / "out")
    mask = Image.open(result["mask_path"]).convert("L")

    assert detection["constrained_bbox"] == [20, 20, 60, 35]
    assert detection["subtract_other_objects"] is True
    assert detection["subtract_bboxes"] == [[38, 45, 24, 40]]
    assert count_mask_pixels(mask) > 0
    assert mask.getpixel((45, 47)) == 0


def test_color_recolor_editor_writes_png_and_mask(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    image = Image.new("RGB", (20, 20), (0, 0, 0))
    for y in range(5, 15):
        for x in range(4, 16):
            image.putpixel((x, y), (220, 30, 30))
    image.save(source_path)
    editor = ColorRecolorEditor(target_color="#1d63d9", feather_radius=0)
    region = InpaintRegion(
        name="umbrella",
        bbox=[4, 5, 12, 10],
        prompt="make the umbrella blue",
        canvas_size=[20, 20],
    )

    result = editor.edit(str(source_path), region, tmp_path / "out")
    edited = Image.open(result["edited_image"]).convert("RGB")

    assert Path(result["mask_path"]).exists()
    assert result["mode"] == "recolor"
    assert result["selected_pixel_count"] == 120
    assert edited.getpixel((10, 10))[2] > edited.getpixel((10, 10))[0]


def test_color_recolor_editor_subtracts_protected_bbox_from_precomputed_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    image = Image.new("RGB", (24, 16), (0, 0, 0))
    for y in range(4, 12):
        for x in range(4, 20):
            image.putpixel((x, y), (220, 30, 30))
    image.save(source_path)
    precomputed_mask = tmp_path / "wide_mask.png"
    Image.new("L", (24, 16), 0).save(precomputed_mask)
    mask = Image.open(precomputed_mask).convert("L")
    for y in range(4, 12):
        for x in range(4, 20):
            mask.putpixel((x, y), 255)
    mask.save(precomputed_mask)
    editor = ColorRecolorEditor(
        target_color="#1d63d9",
        source_color="red",
        feather_radius=0,
        exclude_bboxes=[[12, 4, 8, 8]],
        precomputed_mask_path=str(precomputed_mask),
    )
    region = InpaintRegion(
        name="target object",
        bbox=[4, 4, 16, 8],
        prompt="make the target blue",
        canvas_size=[24, 16],
    )

    result = editor.edit(str(source_path), region, tmp_path / "out")
    edited = Image.open(result["edited_image"]).convert("RGB")
    output_mask = Image.open(result["mask_path"]).convert("L")

    assert result["selected_pixel_count"] == 64
    assert output_mask.getpixel((8, 8)) == 255
    assert output_mask.getpixel((16, 8)) == 0
    assert edited.getpixel((8, 8))[2] > edited.getpixel((8, 8))[0]
    assert edited.getpixel((16, 8))[0] > edited.getpixel((16, 8))[2]


def test_measure_color_coverage_tracks_source_and_target_colors(tmp_path: Path) -> None:
    image_path = tmp_path / "coverage.png"
    image = Image.new("RGB", (10, 10), (0, 0, 0))
    for y in range(0, 10):
        for x in range(0, 5):
            image.putpixel((x, y), (220, 20, 20))
        for x in range(5, 10):
            image.putpixel((x, y), (20, 20, 220))
    image.save(image_path)

    metrics = measure_color_coverage(
        image_path,
        bbox=[0, 0, 10, 10],
        target_color="#1d63d9",
        source_color="red",
    )

    assert metrics["eligible_pixel_count"] == 100
    assert metrics["source_coverage"] == 0.5
    assert metrics["target_coverage"] == 0.5


def test_detect_color_region_flags_tall_component_for_horizontal_target(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    for y in range(5, 95):
        for x in range(58, 66):
            image.putpixel((x, y), (220, 20, 20))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [100, 100],
            "objects": [
                {"name": "blue canopy", "bbox": [20, 20, 60, 35]},
            ],
        }
    }

    _, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "blue canopy",
        prompt="make canopy blue",
        source_color="red",
        target_region="canopy",
        search_expand=1.0,
        component_padding=0,
        min_component_area=10,
        selection_strategy="layout_overlap",
        prefer_object_mask=False,
    )

    assert detection["geometry"]["horizontal_target"] is True
    failure_types = {item["type"] for item in detection["geometry_failures"]}
    assert "tall_component_for_horizontal_target" in failure_types


def test_detect_color_region_can_use_low_saturation_canopy_mask(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "transparent_umbrella.png"
    image = Image.new("RGB", (120, 100), (20, 90, 30))
    for y in range(16, 44):
        for x in range(22, 98):
            if abs((x - 60) / 38) + abs((y - 44) / 28) < 1.25:
                image.putpixel((x, y), (188, 190, 188))
    for y in range(48, 82):
        for x in range(50, 70):
            image.putpixel((x, y), (220, 20, 20))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [120, 100],
            "objects": [
                {"name": "blue umbrella", "bbox": [20, 12, 80, 50]},
                {"name": "small red robot", "bbox": [48, 48, 26, 36]},
            ],
        }
    }

    region, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "umbrella",
        prompt="make umbrella blue",
        source_color="low_saturation",
        target_region="canopy",
        subtract_other_objects=True,
        search_expand=0.85,
        component_padding=2,
        min_component_area=10,
        selection_strategy="layout_overlap",
        mask_output_dir=tmp_path,
    )
    editor = ColorRecolorEditor(
        target_color="#1d63d9",
        source_color="low_saturation",
        feather_radius=0,
        keep_largest_component=True,
        exclude_bboxes=detection["subtract_bboxes"],
        precomputed_mask_path=detection["precomputed_mask_path"],
    )
    result = editor.edit(str(image_path), region, tmp_path / "out")
    edited = Image.open(result["edited_image"]).convert("RGB")
    mask = Image.open(result["mask_path"]).convert("L")

    assert detection["source_color"] == "low_saturation"
    assert detection["mask_mode"] == "object_region"
    assert detection["precomputed_mask_path"]
    assert detection["effective_search_expand"] == 0.05
    assert detection["geometry"]["overlap_target_ratio"] >= 0.08
    assert not detection["geometry_failures"]
    assert region.bbox[2] > region.bbox[3]
    assert count_mask_pixels(mask) > 1000
    assert edited.getpixel((60, 28))[2] > edited.getpixel((60, 28))[0]
    assert edited.getpixel((60, 60))[0] > edited.getpixel((60, 60))[2]


def test_image_grounded_bbox_is_not_cropped_again_by_target_region(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image = Image.new("RGB", (120, 100), (20, 20, 20))
    for y in range(14, 54):
        for x in range(20, 100):
            image.putpixel((x, y), (220, 20, 20))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [120, 100],
            "objects": [
                {
                    "name": "umbrella",
                    "bbox": [20, 14, 80, 40],
                    "bbox_source": "vlm_target_region_locator",
                },
            ],
        }
    }

    _, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "umbrella",
        prompt="make umbrella blue",
        source_color="red",
        target_region="canopy",
        image_grounded_bbox=True,
        search_expand=0.0,
        component_padding=0,
        min_component_area=10,
        mask_output_dir=tmp_path,
    )

    assert detection["image_grounded_bbox"] is True
    assert detection["layout_bbox_scaled"] == [20, 14, 80, 40]
    assert detection["constrained_bbox"] == [20, 14, 80, 40]


def test_object_region_mask_targets_requested_object_not_same_color_background(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "robot_scene.png"
    image = Image.new("RGB", (120, 90), (10, 10, 10))
    for y in range(5, 80):
        for x in range(2, 25):
            image.putpixel((x, y), (175, 175, 175))
    for y in range(28, 70):
        for x in range(70, 102):
            image.putpixel((x, y), (180, 182, 181))
    image.save(image_path)
    layout_context = {
        "layout": {
            "canvas_size": [120, 90],
            "objects": [
                {"name": "small gray robot", "bbox": [68, 26, 36, 46]},
            ],
        }
    }

    region, detection = detect_color_region_from_layout(
        image_path,
        layout_context,
        "robot",
        prompt="make the robot blue",
        source_color="gray",
        target_region="full",
        search_expand=1.0,
        component_padding=0,
        min_component_area=10,
        selection_strategy="largest",
        mask_output_dir=tmp_path,
    )
    editor = ColorRecolorEditor(
        target_color="#1d63d9",
        source_color="gray",
        feather_radius=0,
        precomputed_mask_path=detection["precomputed_mask_path"],
    )
    result = editor.edit(str(image_path), region, tmp_path / "out")
    edited = Image.open(result["edited_image"]).convert("RGB")
    mask = Image.open(result["mask_path"]).convert("L")

    assert detection["mask_mode"] == "object_region"
    assert detection["selected_component"]["bbox"] == [68, 26, 36, 46]
    assert region.bbox == [68, 26, 36, 46]
    assert mask.getpixel((80, 40)) == 255
    assert mask.getpixel((10, 40)) == 0
    assert edited.getpixel((80, 40))[2] > edited.getpixel((80, 40))[0]
    assert edited.getpixel((10, 40)) == (175, 175, 175)


def test_diffusers_inpaint_editor_lazy_loads_and_saves_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "sd15-inpaint"
    model_dir.mkdir()
    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (120, 0, 0)).save(source_path)

    class FakeImage:
        def save(self, path: Path) -> None:
            path.write_text("edited", encoding="utf-8")

    class FakePipe:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pipe = cls()
            pipe.load_args = args
            pipe.load_kwargs = kwargs
            return pipe

        def to(self, device):
            self.device = device
            return self

        def enable_attention_slicing(self):
            self.attention_slicing = True

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(images=[FakeImage()])

    fake_torch = SimpleNamespace(
        float16="float16",
        cuda=SimpleNamespace(is_available=lambda: True),
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: ("gen", device, seed)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(StableDiffusionInpaintPipeline=FakePipe),
    )

    editor = DiffusersInpaintEditor(model_path=model_dir, seed=77)
    region = InpaintRegion(
        name="blue umbrella",
        bbox=[512, 256, 256, 512],
        prompt="paint only a vivid cobalt blue umbrella canopy",
        negative_prompt="red umbrella",
        canvas_size=[1024, 1024],
    )

    result = editor.edit(str(source_path), region, tmp_path / "out")

    assert Path(result["edited_image"]).read_text(encoding="utf-8") == "edited"
    assert Path(result["mask_path"]).exists()
    assert result["scaled_bbox"] == [32, 8, 16, 16]
    assert result["mode"] == "sd15-inpaint"
    assert editor._pipe.last_kwargs["prompt"] == "paint only a vivid cobalt blue umbrella canopy"
    assert editor._pipe.last_kwargs["negative_prompt"] == "red umbrella"
    assert editor._pipe.last_kwargs["generator"] == ("gen", "cuda", 77)


def test_subprocess_inpaint_editor_runs_isolated_python_and_saves_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "sd15-inpaint"
    model_dir.mkdir()
    script_path = tmp_path / "run_inpaint_subprocess.py"
    script_path.write_text("# inpaint script", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (120, 0, 0)).save(source_path)
    calls = []

    def fake_run(command, capture_output, text, timeout, env):
        del capture_output, text, timeout
        calls.append((command, env))
        output_path = Path(command[command.index("--output") + 1])
        metadata_path = Path(command[command.index("--metadata-output") + 1])
        Image.new("RGB", (64, 32), (0, 0, 255)).save(output_path)
        metadata_path.write_text('{"ok": true}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("src.local_editor.subprocess.run", fake_run)

    editor = SubprocessInpaintEditor(
        model_path=model_dir,
        python=python,
        script_path=script_path,
        seed=77,
        cuda_visible_devices="1",
    )
    region = InpaintRegion(
        name="blue umbrella",
        bbox=[512, 256, 256, 512],
        prompt="paint only a vivid cobalt blue umbrella canopy",
        negative_prompt="red umbrella",
        canvas_size=[1024, 1024],
    )

    result = editor.edit(str(source_path), region, tmp_path / "out")

    command, env = calls[0]
    assert command[:2] == [str(python), str(script_path)]
    assert command[command.index("--model-path") + 1] == str(model_dir)
    assert Path(command[command.index("--image") + 1]).is_absolute()
    assert Path(command[command.index("--mask") + 1]).is_absolute()
    assert Path(command[command.index("--output") + 1]).is_absolute()
    assert Path(command[command.index("--metadata-output") + 1]).is_absolute()
    assert command[command.index("--prompt") + 1] == region.prompt
    assert command[command.index("--negative-prompt") + 1] == region.negative_prompt
    assert command[command.index("--seed") + 1] == "77"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert Path(result["edited_image"]).exists()
    assert Path(result["mask_path"]).exists()
    assert Path(result["edited_image"]).is_absolute()
    assert Path(result["mask_path"]).is_absolute()
    assert result["scaled_bbox"] == [32, 8, 16, 16]
    assert result["mode"] == "sd15-subprocess-inpaint"
    assert result["metadata"] == {"ok": True}


def test_powerpaint_subprocess_editor_runs_dedicated_process_and_mask(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "ppt-v2-1"
    checkpoint_dir.mkdir()
    powerpaint_dir = tmp_path / "PowerPaint"
    powerpaint_dir.mkdir()
    script_path = powerpaint_dir / "test.py"
    script_path.write_text("# powerpaint cli", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (120, 0, 0)).save(source_path)
    calls = []

    def fake_run(command, cwd, capture_output, text, timeout, env):
        del capture_output, text, timeout
        calls.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        Image.new("RGB", (64, 32), (0, 255, 0)).save(output_path)
        return SimpleNamespace(returncode=0, stdout="saved", stderr="")

    monkeypatch.setattr("src.local_editor.subprocess.run", fake_run)

    editor = PowerPaintSubprocessEditor(
        checkpoint_dir=checkpoint_dir,
        python=python,
        powerpaint_dir=powerpaint_dir,
        seed=91,
        cuda_visible_devices="1",
        num_inference_steps=12,
        strength=0.8,
    )
    region = InpaintRegion(
        name="red screen",
        bbox=[512, 256, 256, 512],
        prompt="paint a red screen covering the lower half of the suitcase",
        negative_prompt="no screen",
        canvas_size=[1024, 1024],
    )

    result = editor.edit(str(source_path), region, tmp_path / "out")

    command, cwd, env = calls[0]
    assert command[:2] == [str(python), str(script_path)]
    assert cwd == str(powerpaint_dir)
    assert command[command.index("--checkpoint_dir") + 1] == str(checkpoint_dir)
    assert Path(command[command.index("--input_image") + 1]).is_absolute()
    assert Path(command[command.index("--mask_image") + 1]).is_absolute()
    assert Path(command[command.index("--output_path") + 1]).is_absolute()
    assert command[command.index("--prompt") + 1] == region.prompt
    assert command[command.index("--negative_prompt") + 1] == region.negative_prompt
    assert command[command.index("--steps") + 1] == "12"
    assert command[command.index("--fitting_degree") + 1] == "0.8"
    assert "--local_files_only" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert Path(result["edited_image"]).exists()
    assert Path(result["mask_path"]).exists()
    assert Path(result["edited_image"]).is_absolute()
    assert Path(result["mask_path"]).is_absolute()
    assert result["scaled_bbox"] == [32, 8, 16, 16]
    assert result["bbox_coordinates"] == [32, 8, 48, 24]
    assert result["mode"] == "powerpaint-subprocess"
