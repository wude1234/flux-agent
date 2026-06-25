from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mini_benchmark import (
    build_parser,
    _build_batch_mgrag_command,
    _build_command,
    _select_cases,
    _aggregate_results,
    _classify_process_failure,
    _determine_failure_layer,
    _command_for_attempt,
    _run_batch_decision,
    _run_case_with_retries,
    _seed_for_case,
)
from scripts.build_compact_benchmark import build_benchmark_payload


def test_mini_hard_prompts_have_two_cases_per_category() -> None:
    path = PROJECT_ROOT / "benchmarks" / "hard_prompts_mini.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    cases = data["cases"]
    categories = Counter(case["category"] for case in cases)

    assert data["version"] == "flux_agent_mini_hard_prompts_v1"
    assert len(cases) == 16
    assert categories == {
        "attribute_binding": 2,
        "color_binding": 2,
        "count_quantity": 2,
        "interaction_relation": 2,
        "negation_absence": 2,
        "occlusion_visibility": 2,
        "spatial_layout": 2,
        "text_symbol": 2,
    }
    assert all(case["id"] and case["prompt"] and case["expected"] for case in cases)


def test_holdout_hard_prompts_mirror_mini_categories_without_prompt_reuse() -> None:
    original = json.loads(
        (PROJECT_ROOT / "benchmarks" / "hard_prompts_mini.json").read_text(
            encoding="utf-8"
        )
    )
    holdout = json.loads(
        (PROJECT_ROOT / "benchmarks" / "hard_prompts_mini_holdout.json").read_text(
            encoding="utf-8"
        )
    )

    original_cases = original["cases"]
    holdout_cases = holdout["cases"]
    original_categories = Counter(case["category"] for case in original_cases)
    holdout_categories = Counter(case["category"] for case in holdout_cases)
    original_prompts = {case["prompt"].lower() for case in original_cases}
    holdout_prompts = {case["prompt"].lower() for case in holdout_cases}

    assert holdout["version"] == "flux_agent_mini_hard_prompts_holdout_v2"
    assert len(original_cases) == 16
    assert len(holdout_cases) == 26
    expected_holdout_categories = Counter(original_categories)
    expected_holdout_categories.update(
        {
            "color_binding": 2,
            "count_quantity": 2,
            "negation_absence": 2,
            "spatial_layout": 2,
            "text_symbol": 2,
        }
    )
    assert holdout_categories == expected_holdout_categories
    assert not (original_prompts & holdout_prompts)
    assert all(case["id"].startswith("holdout_") for case in holdout_cases)
    assert all(case["prompt"] and case["expected"] for case in holdout_cases)


def test_compact_benchmark_balances_single_axis_and_multi_axis_cases() -> None:
    expected_categories = Counter(
        {
            "attribute_binding": 2,
            "color_binding": 2,
            "count_quantity": 2,
            "interaction_relation": 2,
            "negation_absence": 2,
            "occlusion_visibility": 2,
            "spatial_layout": 2,
            "text_symbol": 2,
            "multi_compositional": 4,
        }
    )

    for split in ("dev", "holdout"):
        data = build_benchmark_payload(split)
        cases = data["cases"]
        categories = Counter(case["category"] for case in cases)
        single_axis_cases = [
            case for case in cases if "single_axis" in set(case.get("focus", []))
        ]
        multi_axis_cases = [
            case for case in cases if case["category"] == "multi_compositional"
        ]

        assert data["version"] == f"flux_agent_compact_compositional_{split}_v1"
        assert len(cases) == 20
        assert categories == expected_categories
        assert len(single_axis_cases) == 16
        assert len(multi_axis_cases) == 4
        assert all(case["prompt"] and case["expected"] and case["focus"] for case in cases)


def test_compact_dev_and_holdout_do_not_reuse_prompt_text() -> None:
    dev = build_benchmark_payload("dev")
    holdout = build_benchmark_payload("holdout")
    dev_prompts = {case["prompt"].lower() for case in dev["cases"]}
    holdout_prompts = {case["prompt"].lower() for case in holdout["cases"]}

    assert not (dev_prompts & holdout_prompts)
    assert all(case["id"].startswith("compact_dev_") for case in dev["cases"])
    assert all(case["id"].startswith("compact_holdout_") for case in holdout["cases"])


def test_mini_benchmark_aggregates_focus_tags() -> None:
    aggregate = _aggregate_results(
        [
            {
                "status": "completed",
                "category": "color_binding",
                "focus": ["color_binding", "material_binding"],
                "completion_passed": True,
                "constraint_passed": True,
                "evaluation_passed": True,
            },
            {
                "status": "completed",
                "category": "multi_compositional",
                "focus": ["color_binding", "count"],
                "completion_passed": False,
                "constraint_passed": False,
                "evaluation_passed": False,
            },
        ]
    )

    assert aggregate["focus_counts"] == {
        "color_binding": 2,
        "count": 1,
        "material_binding": 1,
    }
    assert aggregate["focus_completion_passed"] == {
        "color_binding": 1,
        "material_binding": 1,
    }


def test_mini_benchmark_aggregates_typed_route_and_edit_metrics() -> None:
    aggregate = _aggregate_results(
        [
            {
                "status": "completed",
                "category": "text_symbol",
                "focus": ["text_symbol"],
                "completion_passed": False,
                "constraint_passed": True,
                "evaluation_passed": False,
                "typed_routes": ["exact_text_overlay", "none"],
                "route_none_count": 1,
                "efficient_edit_attempts": 1,
                "accepted_edit_count": 0,
            },
            {
                "status": "completed",
                "category": "occlusion_visibility",
                "focus": ["occlusion"],
                "completion_passed": True,
                "constraint_passed": True,
                "evaluation_passed": True,
                "typed_routes": ["occlusion_object_insertion"],
                "efficient_edit_attempts": 2,
                "accepted_edit_count": 1,
            },
        ]
    )

    assert aggregate["typed_route_counts"] == {
        "exact_text_overlay": 1,
        "none": 1,
        "occlusion_object_insertion": 1,
    }
    assert aggregate["route_none_count"] == 1
    assert aggregate["efficient_edit_attempts"] == 3
    assert aggregate["accepted_edit_count"] == 1
    assert aggregate["false_pass_blocked_count"] == 1


def test_determine_failure_layer_l1_generation() -> None:
    # Infrastructure / subprocess / no-image cases all attribute to L1.
    assert _determine_failure_layer({"status": "subprocess_timeout"}) == "L1_generation"
    assert _determine_failure_layer({"status": "gpu_oom"}) == "L1_generation"
    assert _determine_failure_layer({"status": "subprocess_failed"}) == "L1_generation"
    assert (
        _determine_failure_layer({"status": "missing_pregenerated_image"})
        == "L1_generation"
    )
    # Run "completed" but no image was actually selected.
    assert (
        _determine_failure_layer(
            {"status": "completed", "rounds": 0, "selected_image": None}
        )
        == "L1_generation"
    )


def test_determine_failure_layer_success_and_clarify() -> None:
    assert _determine_failure_layer({"completion_passed": True}) == "none"
    assert (
        _determine_failure_layer({"status": "needs_clarification"}) == "unclear"
    )


def test_determine_failure_layer_l2_missing_judgment() -> None:
    # L1 produced an image, run completed, but no judgment exists at all.
    assert (
        _determine_failure_layer(
            {
                "status": "completed",
                "rounds": 1,
                "selected_image": "img.png",
                "constraint_passed": None,
                "evaluation_passed": None,
                "completion_passed": False,
            }
        )
        == "L2_judgment"
    )


def test_determine_failure_layer_l2_false_pass() -> None:
    # L2 said the hard constraint passed but the case is still wrong.
    assert (
        _determine_failure_layer(
            {
                "status": "completed",
                "rounds": 1,
                "selected_image": "img.png",
                "constraint_passed": True,
                "completion_passed": False,
            }
        )
        == "L2_judgment"
    )


def test_determine_failure_layer_l3_not_triggered() -> None:
    # The misattribution trap: L2 correctly flagged a hard-constraint failure
    # (constraint_passed=False) but L3 never fired any repair. This must NOT
    # be blamed on L2.
    assert (
        _determine_failure_layer(
            {
                "status": "max_rounds_reached",
                "rounds": 2,
                "selected_image": "img.png",
                "constraint_passed": False,
                "evaluation_passed": False,
                "completion_passed": False,
                "typed_routes": None,
                "efficient_edit_attempts": 0,
                "typed_action_attempts": 0,
            }
        )
        == "L3_not_triggered"
    )


def test_determine_failure_layer_none_routes_are_not_repair() -> None:
    # A typed_route of "none" means the router declined to fire any real repair.
    # ['none', 'none'] must count as L3_not_triggered, NOT L3_repair — otherwise
    # a do-nothing router gets credit for "attempting" a repair.
    assert (
        _determine_failure_layer(
            {
                "status": "max_rounds_reached",
                "rounds": 2,
                "selected_image": "img.png",
                "constraint_passed": False,
                "evaluation_passed": False,
                "completion_passed": False,
                "typed_routes": ["none", "none"],
                "efficient_edit_attempts": 0,
                "typed_action_attempts": 0,
            }
        )
        == "L3_not_triggered"
    )


def test_determine_failure_layer_l3_repair_failed() -> None:
    # L2 flagged the failure, L3 fired a repair, but completion still failed.
    assert (
        _determine_failure_layer(
            {
                "status": "max_rounds_reached",
                "rounds": 2,
                "selected_image": "img.png",
                "constraint_passed": False,
                "completion_passed": False,
                "typed_routes": ["count_aware_regeneration"],
                "efficient_edit_attempts": 1,
            }
        )
        == "L3_repair"
    )


def test_aggregate_results_counts_failure_layers() -> None:
    aggregate = _aggregate_results(
        [
            {"status": "completed", "completion_passed": True,
             "constraint_passed": True, "selected_image": "a.png", "rounds": 1},
            {"status": "subprocess_timeout", "completion_passed": False},
            {
                "status": "max_rounds_reached",
                "rounds": 2,
                "selected_image": "b.png",
                "constraint_passed": False,
                "completion_passed": False,
                "typed_routes": None,
                "efficient_edit_attempts": 0,
            },
        ]
    )
    assert aggregate["failure_layer_counts"] == {
        "L1_generation": 1,
        "L3_not_triggered": 1,
        "none": 1,
    }


def test_mini_benchmark_uses_case_id_stable_seed_by_default() -> None:
    args = build_parser().parse_args(["--seed-base", "7100"])
    case = {"id": "holdout_spatial_001", "category": "spatial_layout", "prompt": "p"}

    assert _seed_for_case(args, case, 0) == _seed_for_case(args, case, 12)
    assert _seed_for_case(args, case, 0) != _seed_for_case(
        args,
        {"id": "holdout_spatial_002", "category": "spatial_layout", "prompt": "p"},
        0,
    )


def test_mini_benchmark_can_use_sequential_seed_policy() -> None:
    args = build_parser().parse_args(
        ["--seed-base", "7100", "--seed-policy", "sequential"]
    )
    case = {"id": "holdout_spatial_001", "category": "spatial_layout", "prompt": "p"}

    assert _seed_for_case(args, case, 0) == 7100
    assert _seed_for_case(args, case, 3) == 7103


def test_mini_benchmark_prefers_case_embedded_seed() -> None:
    args = build_parser().parse_args(
        ["--seed-base", "7100", "--seed-policy", "sequential"]
    )
    case = {
        "id": "drawbench_colors_001",
        "category": "drawbench_colors",
        "prompt": "A red colored car.",
        "seed": 304711,
    }

    assert _seed_for_case(args, case, 99) == 304711


def test_mini_benchmark_can_select_all_cases_for_repeated_categories() -> None:
    args = build_parser().parse_args(
        [
            "--category",
            "spatial_layout",
            "--category",
            "text_symbol",
        ]
    )
    cases = [
        {"id": "a", "category": "spatial_layout", "prompt": "p"},
        {"id": "b", "category": "text_symbol", "prompt": "p"},
        {"id": "c", "category": "color_binding", "prompt": "p"},
        {"id": "d", "category": "spatial_layout", "prompt": "p"},
    ]

    selected = _select_cases(cases, args)

    assert [case["id"] for case in selected] == ["a", "b", "d"]


def test_mini_benchmark_forwards_layout_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--generator",
            "flux",
            "--use-layout-planner",
            "--layout-llm",
            "mock",
            "--layout-canvas-width",
            "512",
            "--layout-canvas-height",
            "512",
        ]
    )
    command = _build_command(
        args,
        {"prompt": "A red cube is left of a blue sphere.", "id": "case", "category": "spatial"},
        tmp_path,
        "run-id",
        123,
    )

    assert "--use-layout-planner" in command
    assert command[command.index("--layout-llm") + 1] == "mock"
    assert command[command.index("--layout-canvas-width") + 1] == "512"
    assert command[command.index("--layout-canvas-height") + 1] == "512"


def test_mini_benchmark_forwards_auto_candidate_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--auto-n-images-for-hard-prompts",
            "--hard-prompt-n-images",
            "3",
        ]
    )
    command = _build_command(
        args,
        {
            "prompt": "A red cube is left of a blue sphere.",
            "id": "case",
            "category": "spatial",
        },
        tmp_path,
        "run-id",
        123,
    )

    assert "--auto-n-images-for-hard-prompts" in command
    assert command[command.index("--hard-prompt-n-images") + 1] == "3"


def test_mini_benchmark_auto_enables_efficient_repair_for_editable_category(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "--generator",
            "flux",
            "--cuda-visible-devices",
            "0",
            "--auto-efficient-repair-for-categories",
        ]
    )
    command = _build_command(
        args,
        {
            "prompt": "A red screen hides the lower half of a green suitcase.",
            "id": "case",
            "category": "occlusion_visibility",
        },
        tmp_path,
        "run-id",
        123,
    )

    assert "--enable-object-insertion-repair" in command
    assert "--enable-relation-repair" in command
    assert "--enable-vlm-target-locator" in command
    assert "--enable-efficient-repair-agent" in command
    assert "--enable-editing-mask-agent" in command
    assert command[command.index("--editing-mask-mode") + 1] == "auto"
    assert command[command.index("--relation-editor") + 1] == "powerpaint-subprocess"
    assert command[command.index("--relation-candidates") + 1] == "1"
    assert command[command.index("--relation-inpaint-cuda-visible-devices") + 1] == "0"
    assert command[command.index("--grounded-sam2-cuda-visible-devices") + 1] == "0"
    assert command[command.index("--powerpaint-timeout-seconds") + 1] == "1800"


def test_mini_benchmark_auto_efficient_repair_skips_count_category(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "--generator",
            "flux",
            "--auto-efficient-repair-for-categories",
        ]
    )
    command = _build_command(
        args,
        {
            "prompt": "Exactly five silver coins are arranged in a row.",
            "id": "case",
            "category": "count_quantity",
        },
        tmp_path,
        "run-id",
        123,
    )

    assert "--enable-efficient-repair-agent" not in command
    assert "--enable-object-insertion-repair" not in command
    assert "--relation-editor" not in command


def test_mini_benchmark_classifies_sigkill_separately() -> None:
    result = _classify_process_failure(
        -9,
        "",
        "RuntimeError: FLUX generation failed with exit code -9.",
    )

    assert result["status"] == "process_killed"
    assert result["failure_category"] == "infrastructure"


def test_mini_benchmark_classifies_wrapped_flux_sigkill() -> None:
    result = _classify_process_failure(
        1,
        "",
        "RuntimeError: FLUX generation failed with exit code -9.\nstderr:\nloading...",
    )

    assert result["status"] == "process_killed"
    assert result["failure_category"] == "infrastructure"


def test_mini_benchmark_retries_retryable_infra_failure(monkeypatch) -> None:
    calls = []

    def fake_run_case_once(case, command, seed, *, timeout, attempt_index=0):
        calls.append(attempt_index)
        if attempt_index == 0:
            return {
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "seed": seed,
                "command": list(command),
                "attempt": attempt_index,
                "status": "process_killed",
                "returncode": 1,
                "failure_category": "infrastructure",
                "failure_reason": "FLUX generation failed with exit code -9",
            }
        return {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "seed": seed,
            "command": list(command),
            "attempt": attempt_index,
            "status": "completed",
            "returncode": 0,
            "completion_passed": True,
        }

    monkeypatch.setattr(
        "scripts.run_mini_benchmark._run_case_once",
        fake_run_case_once,
    )

    result = _run_case_with_retries(
        {"id": "case", "category": "count", "prompt": "prompt"},
        ["python", "-m", "src.run_m4"],
        123,
        timeout=1,
        max_retries=1,
        retry_delay_seconds=0,
    )

    assert calls == [0, 1]
    assert result["status"] == "completed"
    assert result["attempt"] == 1
    assert [item["status"] for item in result["attempts"]] == [
        "process_killed",
        "completed",
    ]


def test_mini_benchmark_retry_attempt_uses_isolated_run_id() -> None:
    command = [
        "python",
        "-m",
        "src.run_m4",
        "--run-id",
        "bench-case",
        "--prompt",
        "prompt",
    ]

    assert _command_for_attempt(command, 0)[3:5] == ["--run-id", "bench-case"]
    assert _command_for_attempt(command, 2)[3:5] == ["--run-id", "bench-case-retry2"]


def test_mini_benchmark_forwards_image_edit_repair_flags() -> None:
    args = build_parser().parse_args(
        [
            "--benchmark",
            "benchmarks/hard_prompts_mini.json",
            "--generator",
            "flux",
            "--enable-local-repair",
            "--enable-vlm-target-locator",
            "--local-editor",
            "recolor",
            "--mask-refiner",
            "bbox",
            "--enable-object-insertion-repair",
            "--enable-relation-repair",
            "--relation-editor",
            "powerpaint-subprocess",
            "--relation-inpaint-model-path",
            "/models/sd15-inpaint",
            "--relation-inpaint-python",
            "/envs/sdxl/bin/python",
            "--relation-inpaint-timeout-seconds",
            "123",
            "--relation-inpaint-cuda-visible-devices",
            "1",
            "--powerpaint-python",
            "/envs/powerpaint/bin/python",
            "--powerpaint-dir",
            "/code/PowerPaint",
            "--powerpaint-checkpoint-dir",
            "/models/PowerPaint/ppt-v2-1",
            "--powerpaint-timeout-seconds",
            "456",
            "--relation-candidates",
            "2",
            "--relation-steps",
            "12",
            "--relation-guidance-scale",
            "6.5",
            "--relation-strength",
            "0.7",
            "--device",
            "cuda:0",
        ]
    )

    command = _build_command(
        args,
        {"id": "case", "category": "color_binding", "prompt": "A red cup."},
        Path("runs"),
        "bench-case",
        123,
    )

    assert "--enable-local-repair" in command
    assert "--enable-vlm-target-locator" in command
    assert command[command.index("--local-editor") + 1] == "recolor"
    assert command[command.index("--mask-refiner") + 1] == "bbox"
    assert "--enable-object-insertion-repair" in command
    assert "--enable-relation-repair" in command
    assert command[command.index("--relation-editor") + 1] == "powerpaint-subprocess"
    assert command[command.index("--relation-inpaint-model-path") + 1] == "/models/sd15-inpaint"
    assert command[command.index("--relation-inpaint-python") + 1] == "/envs/sdxl/bin/python"
    assert command[command.index("--relation-inpaint-timeout-seconds") + 1] == "123"
    assert command[command.index("--relation-inpaint-cuda-visible-devices") + 1] == "1"
    assert command[command.index("--powerpaint-python") + 1] == "/envs/powerpaint/bin/python"
    assert command[command.index("--powerpaint-dir") + 1] == "/code/PowerPaint"
    assert command[command.index("--powerpaint-checkpoint-dir") + 1] == "/models/PowerPaint/ppt-v2-1"
    assert command[command.index("--powerpaint-timeout-seconds") + 1] == "456"
    assert command[command.index("--relation-candidates") + 1] == "2"
    assert command[command.index("--relation-steps") + 1] == "12"
    assert command[command.index("--relation-guidance-scale") + 1] == "6.5"
    assert command[command.index("--relation-strength") + 1] == "0.7"
    assert command[command.index("--device") + 1] == "cuda:0"


def test_mini_benchmark_decision_only_disables_auto_efficient_repair() -> None:
    args = build_parser().parse_args(
        [
            "--benchmark",
            "benchmarks/hard_prompts_mini.json",
            "--generator",
            "flux",
            "--decision-only",
            "--auto-efficient-repair-for-categories",
        ]
    )

    command = _build_command(
        args,
        {
            "id": "case",
            "category": "occlusion_visibility",
            "prompt": "A red screen hides the lower half of a green suitcase.",
        },
        Path("runs"),
        "bench-case",
        123,
    )

    assert "--decision-only" in command
    assert "--enable-efficient-repair-agent" not in command
    assert "--enable-editing-mask-agent" not in command
    assert "--relation-editor" not in command
    assert "--powerpaint-python" not in command


def test_mini_benchmark_batch_decision_builds_one_mgrag_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--benchmark",
            "benchmarks/hard_prompts_mini.json",
            "--batch-decision",
            "--dry-run",
            "--generator",
            "flux",
            "--cuda-visible-devices",
            "0",
        ]
    )
    cases = [
        {"id": "case_a", "category": "spatial", "prompt": "A red cube left of a blue sphere."},
        {"id": "case_b", "category": "count", "prompt": "Exactly two yellow cups."},
    ]
    seeds = [_seed_for_case(args, case, index) for index, case in enumerate(cases)]

    command = _build_batch_mgrag_command(
        args,
        [str(case["prompt"]) for case in cases],
        seeds,
        tmp_path / "images",
    )

    assert command.count("--prompt") == 2
    assert command.count("--prompt_seed") == 2
    assert str(seeds[0]) in command
    assert str(seeds[1]) in command


def test_mini_benchmark_batch_decision_dry_run_returns_batch_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--benchmark",
            "benchmarks/hard_prompts_mini.json",
            "--batch-decision",
            "--dry-run",
            "--generator",
            "flux",
        ]
    )
    cases = [
        {"id": "case_a", "category": "spatial", "prompt": "A red cube left of a blue sphere."},
        {"id": "case_b", "category": "count", "prompt": "Exactly two yellow cups."},
    ]

    results = _run_batch_decision(
        args,
        {"version": "test", "cases": cases},
        cases,
        tmp_path,
        "batch-test",
    )

    assert len(results) == 2
    assert results[0]["dry_run_command"].count("--prompt") == 2
    assert results[0]["dry_run_command"] == results[1]["dry_run_command"]
    assert (tmp_path / "batch_decision_generation_command.json").exists()


def test_mini_benchmark_batch_decision_uses_micro_batches(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--benchmark",
            "benchmarks/hard_prompts_mini.json",
            "--batch-decision",
            "--batch-decision-chunk-size",
            "2",
            "--dry-run",
            "--generator",
            "flux",
        ]
    )
    cases = [
        {"id": f"case_{index}", "category": "spatial", "prompt": f"Prompt {index}."}
        for index in range(5)
    ]

    results = _run_batch_decision(
        args,
        {"version": "test", "cases": cases},
        cases,
        tmp_path,
        "batch-test",
    )

    commands = results[0]["dry_run_commands"]
    assert len(commands) == 3
    assert [command.count("--prompt") for command in commands] == [2, 2, 1]
    assert commands[1][commands[1].index("--start_index") + 1] == "2"
    assert commands[2][commands[2].index("--start_index") + 1] == "4"
