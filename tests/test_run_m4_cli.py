from __future__ import annotations

from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_m4 import build_parser, _effective_n_images, _is_hard_compositional_prompt


def test_run_m4_accepts_legacy_m6_flag_aliases() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red robot holding a blue umbrella",
            "--enable-evaluator",
            "--evaluator-vlm",
            "api",
            "--evaluator-vlm-model",
            "qwen-vl-plus",
            "--enable-reward-rerank",
            "--reward-backend",
            "api",
            "--reward-vlm-model",
            "qwen-vl-plus",
            "--enable-local-repair",
            "--local-editor",
            "recolor",
        ]
    )

    assert args.enable_m6_evaluator is True
    assert args.evaluator_vlm == "api"
    assert args.evaluator_vlm_model == "qwen-vl-plus"
    assert args.enable_api_reward_reranker is True
    assert args.reward_backend == "api"
    assert args.reward_vlm_model == "qwen-vl-plus"
    assert args.enable_local_repair is True
    assert args.local_editor == "recolor"


def test_run_m4_accepts_decision_only_flag() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A red cube is left of a blue sphere.",
            "--enable-evaluator",
            "--max-rounds",
            "2",
            "--decision-only",
        ]
    )

    assert args.decision_only is True
    assert args.enable_typed_action_backend is False


def test_run_m4_accepts_specialist_agent_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a cyan cat holding a red umbrella handle",
            "--disable-specialist-reports",
            "--enable-specialist-vlm-observation",
        ]
    )

    assert args.disable_specialist_reports is True
    assert args.enable_specialist_vlm_observation is True


def test_run_m4_auto_candidate_count_for_hard_prompts() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A blue notebook shows a yellow star symbol next to a plain green notebook.",
            "--n-images",
            "1",
            "--auto-n-images-for-hard-prompts",
            "--hard-prompt-n-images",
            "2",
        ]
    )

    assert _is_hard_compositional_prompt(args.prompt) is True
    assert _effective_n_images(args) == (2, "auto_hard_compositional_prompt")


def test_run_m4_auto_candidate_count_keeps_simple_prompts_small() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A portrait of a red robot.",
            "--n-images",
            "1",
            "--auto-n-images-for-hard-prompts",
            "--hard-prompt-n-images",
            "2",
        ]
    )

    assert _is_hard_compositional_prompt(args.prompt) is False
    assert _effective_n_images(args) == (1, "simple_or_unclassified_prompt")


def test_run_m4_auto_hard_prompts_enable_variants_in_mock_run(tmp_path: Path) -> None:
    from src import run_m4

    run_m4.main(
        [
            "--prompt",
            "A red cube is left of a blue sphere, and the blue sphere is under a green cone.",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "auto-hard-variants",
            "--generator",
            "mock",
            "--llm",
            "mock",
            "--vlm",
            "mock",
            "--max-rounds",
            "1",
            "--n-images",
            "1",
            "--auto-n-images-for-hard-prompts",
            "--hard-prompt-n-images",
            "2",
        ]
    )

    run_log = json.loads((tmp_path / "auto-hard-variants" / "run.json").read_text())
    image_event = next(
        event for event in run_log["events"] if event["type"] == "images_generated"
    )
    variant_event = next(
        event for event in run_log["events"] if event["type"] == "binding_prompt_variants"
    )
    assert len(image_event["prompts"]) == 2
    assert variant_event["strategies"] == ["base", "spatial_literal"]


def test_run_m4_decision_only_disables_action_backends_in_mock_run(tmp_path: Path) -> None:
    from src import run_m4

    run_m4.main(
        [
            "--prompt",
            "A red cube is left of a blue sphere.",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "decision-only",
            "--generator",
            "mock",
            "--llm",
            "mock",
            "--vlm",
            "mock",
            "--enable-evaluator",
            "--max-rounds",
            "2",
            "--decision-only",
            "--enable-typed-action-backend",
            "--enable-local-repair",
            "--enable-relation-repair",
            "--enable-object-insertion-repair",
            "--enable-efficient-repair-agent",
        ]
    )

    run_log = json.loads((tmp_path / "decision-only" / "run.json").read_text())
    config_log = json.loads((tmp_path / "decision-only" / "config.json").read_text())
    assert run_log["config"]["max_rounds"] == 1
    m6_config = config_log["m6"]
    assert m6_config["enable_typed_action_backend"] is False
    assert m6_config["enable_local_repair"] is False
    assert m6_config["enable_relation_repair"] is False
    assert m6_config["enable_object_insertion_repair"] is False


def test_run_m4_accepts_object_insertion_repair_flag() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "two yellow birds sitting on a black bicycle near a white dog",
            "--enable-object-insertion-repair",
            "--relation-editor",
            "sd15-inpaint",
        ]
    )

    assert args.enable_object_insertion_repair is True
    assert args.enable_relation_repair is False
    assert args.relation_editor == "sd15-inpaint"


def test_run_m4_accepts_subprocess_inpaint_repair_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a robot holding a blue umbrella handle",
            "--enable-relation-repair",
            "--relation-editor",
            "sd15-subprocess-inpaint",
            "--relation-inpaint-model-path",
            "/models/sd15-inpaint",
            "--relation-inpaint-python",
            "/envs/sdxl/bin/python",
            "--relation-inpaint-timeout-seconds",
            "123",
            "--relation-inpaint-cuda-visible-devices",
            "1",
        ]
    )

    assert args.enable_relation_repair is True
    assert args.relation_editor == "sd15-subprocess-inpaint"
    assert args.relation_inpaint_model_path == "/models/sd15-inpaint"
    assert args.relation_inpaint_python == "/envs/sdxl/bin/python"
    assert args.relation_inpaint_timeout_seconds == 123
    assert args.relation_inpaint_cuda_visible_devices == "1"


def test_run_m4_builds_subprocess_inpaint_repairer(tmp_path: Path) -> None:
    from src.run_m4 import _build_relation_repairer
    from src.local_editor import SubprocessInpaintEditor
    from src.clients import MockVLMClient

    model_dir = tmp_path / "sd15-inpaint"
    model_dir.mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--prompt",
            "a robot holding a blue umbrella handle",
            "--enable-relation-repair",
            "--relation-editor",
            "sd15-subprocess-inpaint",
            "--relation-inpaint-model-path",
            str(model_dir),
            "--relation-inpaint-python",
            str(python),
            "--relation-inpaint-cuda-visible-devices",
            "1",
        ]
    )

    repairer = _build_relation_repairer(args, MockVLMClient())

    assert repairer is not None
    assert isinstance(repairer.editor, SubprocessInpaintEditor)
    assert repairer.editor.python == python
    assert repairer.editor.cuda_visible_devices == "1"


def test_run_m4_builds_powerpaint_subprocess_repairer(tmp_path: Path) -> None:
    from src.run_m4 import _build_relation_repairer
    from src.local_editor import PowerPaintSubprocessEditor
    from src.clients import MockVLMClient

    checkpoint_dir = tmp_path / "ppt-v2-1"
    checkpoint_dir.mkdir()
    powerpaint_dir = tmp_path / "PowerPaint"
    powerpaint_dir.mkdir()
    (powerpaint_dir / "test.py").write_text("# powerpaint", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red screen hides a suitcase",
            "--enable-object-insertion-repair",
            "--relation-editor",
            "powerpaint-subprocess",
            "--powerpaint-python",
            str(python),
            "--powerpaint-dir",
            str(powerpaint_dir),
            "--powerpaint-checkpoint-dir",
            str(checkpoint_dir),
            "--powerpaint-timeout-seconds",
            "321",
            "--relation-inpaint-cuda-visible-devices",
            "1",
        ]
    )

    repairer = _build_relation_repairer(args, MockVLMClient())

    assert repairer is not None
    assert isinstance(repairer.editor, PowerPaintSubprocessEditor)
    assert repairer.editor.python == python
    assert repairer.editor.powerpaint_dir == powerpaint_dir
    assert repairer.editor.checkpoint_dir == checkpoint_dir
    assert repairer.editor.timeout_seconds == 321
    assert repairer.editor.cuda_visible_devices == "1"


def test_run_m4_wraps_powerpaint_with_editing_mask_agent(tmp_path: Path) -> None:
    from src.run_m4 import _build_relation_repairer
    from src.clients import MockVLMClient
    from src.editing_agent import GroundedSAM2SubprocessMasker, MaskGeneratingInpaintEditor
    from src.local_editor import PowerPaintSubprocessEditor

    checkpoint_dir = tmp_path / "ppt-v2-1"
    checkpoint_dir.mkdir()
    powerpaint_dir = tmp_path / "PowerPaint"
    powerpaint_dir.mkdir()
    (powerpaint_dir / "test.py").write_text("# powerpaint", encoding="utf-8")
    powerpaint_python = tmp_path / "powerpaint_python"
    powerpaint_python.write_text("#!/bin/sh\n", encoding="utf-8")
    grounded_dir = tmp_path / "Grounded_SAM2"
    grounded_dir.mkdir()
    grounded_python = tmp_path / "sam2_python"
    grounded_python.write_text("#!/bin/sh\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red screen hides a suitcase",
            "--enable-object-insertion-repair",
            "--relation-editor",
            "powerpaint-subprocess",
            "--powerpaint-python",
            str(powerpaint_python),
            "--powerpaint-dir",
            str(powerpaint_dir),
            "--powerpaint-checkpoint-dir",
            str(checkpoint_dir),
            "--enable-editing-mask-agent",
            "--editing-mask-mode",
            "auto",
            "--editing-mask-text",
            "green suitcase",
            "--editing-mask-dilation-kernel-size",
            "31",
            "--grounded-sam2-python",
            str(grounded_python),
            "--grounded-sam2-dir",
            str(grounded_dir),
            "--grounded-sam2-cuda-visible-devices",
            "1",
        ]
    )

    repairer = _build_relation_repairer(args, MockVLMClient())

    assert repairer is not None
    assert isinstance(repairer.editor, MaskGeneratingInpaintEditor)
    assert isinstance(repairer.editor.base_editor, PowerPaintSubprocessEditor)
    assert isinstance(repairer.editor.mask_generator, GroundedSAM2SubprocessMasker)
    assert repairer.editor.mask_text == "green suitcase"
    assert repairer.editor.dilation_kernel_size == 31
    assert repairer.editor.mask_generator.cuda_visible_devices == "1"
    assert str(repairer.editor.mask_generator.hf_home).endswith("powerpaint_envs/hf-cache")


def test_run_m4_accepts_efficient_repair_agent_flag() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A black sign displays the exact yellow text 'NO'.",
            "--enable-efficient-repair-agent",
        ]
    )

    assert args.enable_efficient_repair_agent is True


def test_run_m4_builds_efficient_repair_agent_without_heavy_editor() -> None:
    from src.run_m4 import _build_efficient_repair_agent
    from src.editing_agent import EfficientRepairAgent

    args = build_parser().parse_args(
        [
            "--prompt",
            "A black sign displays the exact yellow text 'NO'.",
            "--enable-efficient-repair-agent",
        ]
    )

    agent = _build_efficient_repair_agent(args, relation_repairer=None)

    assert isinstance(agent, EfficientRepairAgent)
    assert agent.inpaint_agent is None


def test_run_m4_editing_mask_agent_can_use_bbox_without_grounded_sam2() -> None:
    from src.run_m4 import _build_relation_repairer
    from src.clients import MockVLMClient
    from src.editing_agent import MaskGeneratingInpaintEditor

    args = build_parser().parse_args(
        [
            "--prompt",
            "a red screen hides a suitcase",
            "--enable-object-insertion-repair",
            "--enable-editing-mask-agent",
            "--editing-mask-mode",
            "bbox",
            "--editing-mask-dilation-kernel-size",
            "21",
        ]
    )

    repairer = _build_relation_repairer(args, MockVLMClient())

    assert repairer is not None
    assert isinstance(repairer.editor, MaskGeneratingInpaintEditor)
    assert repairer.editor.mask_generator is None
    assert repairer.editor.mask_mode == "bbox"
    assert repairer.editor.dilation_kernel_size == 21


def test_run_m4_requires_powerpaint_python_for_powerpaint_subprocess(tmp_path: Path) -> None:
    from src.run_m4 import _build_relation_repairer
    from src.clients import MockVLMClient

    checkpoint_dir = tmp_path / "ppt-v2-1"
    checkpoint_dir.mkdir()
    powerpaint_dir = tmp_path / "PowerPaint"
    powerpaint_dir.mkdir()
    (powerpaint_dir / "test.py").write_text("# powerpaint", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red screen hides a suitcase",
            "--enable-object-insertion-repair",
            "--relation-editor",
            "powerpaint-subprocess",
            "--powerpaint-dir",
            str(powerpaint_dir),
            "--powerpaint-checkpoint-dir",
            str(checkpoint_dir),
        ]
    )

    repairer = _build_relation_repairer(args, MockVLMClient())

    assert repairer is not None
    assert str(repairer.editor.python).endswith("/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python")


def test_run_m4_accepts_mask_refiner_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red robot holding a blue umbrella",
            "--enable-local-repair",
            "--mask-refiner",
            "sam-v1",
            "--sam-checkpoint-path",
            "/tmp/sam.pth",
            "--sam-model-type",
            "vit_l",
            "--sam-device",
            "cuda:0",
        ]
    )

    assert args.enable_local_repair is True
    assert args.mask_refiner == "sam-v1"
    assert args.sam_checkpoint_path == "/tmp/sam.pth"
    assert args.sam_model_type == "vit_l"
    assert args.sam_device == "cuda:0"


def test_run_m4_accepts_flux_generator_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red cube on a white table",
            "--generator",
            "flux",
            "--steps",
            "1",
            "--width",
            "256",
            "--height",
            "256",
            "--cuda-visible-devices",
            "1",
        ]
    )

    assert args.generator == "flux"
    assert args.steps == 1
    assert args.flux_repo == "/home/zrr/flux"
    assert args.flux_python == "/mnt/ssd1/conda/envs/flux-dev/bin/python"
    assert args.flux_name == "flux-dev"
    assert args.flux_model_path.endswith("flux1-dev.safetensors")
    assert args.flux_ae_path.endswith("ae.safetensors")
    assert args.flux_hf_home == "/mnt/ssd3/zrr/hf_cache"
    assert args.flux_attn_mode == "mgrag"
    assert args.mgrag_bias_scale == 1.0
    assert args.mgrag_delta_scale == 1.3
    assert args.mgrag_intervene_steps == 20
    assert args.flux_online is False
    assert args.flux_no_offload is False
    assert args.cuda_visible_devices == "1"


def test_run_m4_defaults_to_mgrag_flux_30_steps_and_20_intervention() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red cube on a white table",
            "--generator",
            "flux",
        ]
    )

    assert args.steps == 30
    assert args.flux_attn_mode == "mgrag"
    assert args.mgrag_bias_scale == 1.0
    assert args.mgrag_delta_scale == 1.3
    assert args.mgrag_intervene_steps == 20


def test_run_m4_can_switch_flux_back_to_baseline_attention() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a red cube on a white table",
            "--generator",
            "flux",
            "--flux-attn-mode",
            "baseline",
        ]
    )

    assert args.flux_attn_mode == "baseline"


def test_run_m4_accepts_flux_first_fusion_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "a cyan cat holding a red umbrella handle",
            "--generator",
            "fusion",
            "--fusion-policy",
            "parallel",
            "--sdxl-device",
            "cuda:1",
            "--flux-timeout-seconds",
            "600",
            "--steps",
            "30",
            "--width",
            "512",
            "--height",
            "512",
        ]
    )

    assert args.generator == "fusion"
    assert args.fusion_policy == "parallel"
    assert args.sdxl_model_path == "/mnt/ssd1/models/stable-diffusion-xl-base-1.0"
    assert args.sdxl_device == "cuda:1"
    assert args.flux_timeout_seconds == 600
    assert args.max_rounds == 2
    assert args.steps == 30
    assert args.width == 512
    assert args.height == 512


def test_run_m4_accepts_typed_action_backend_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A pencil holder with more pens than pencils.",
            "--enable-evaluator",
            "--enable-typed-action-backend",
            "--typed-action-candidates",
            "3",
            "--typed-action-max-candidates",
            "4",
        ]
    )

    assert args.enable_typed_action_backend is True
    assert args.disable_typed_action_backend is False
    assert args.typed_action_candidates == 3
    assert args.typed_action_max_candidates == 4
