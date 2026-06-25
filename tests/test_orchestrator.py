import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockLLMClient, MockVLMClient
from src.evaluators import VLMJudgeEvaluator
from src.image_generator import FusionImageGenerator, MockImageGenerator
from src.layout_planner import LayoutPlanner, build_mock_layout_response
from src.logging_utils import atomic_write_text, build_final_report, create_run_dir, write_json
from src.orchestrator import (
    OrchestratorAgent,
    _best_final_selection,
    _configure_mock_placeholder_dir,
    _efficient_repair_gate,
    _efficient_repair_attempted_editing_backend,
    _efficient_repair_request_from_plan,
    _first_recolor_repair_plan,
    _guard_prompt_relation_drift,
    _completion_score,
    _hard_pass_guard,
    _local_repair_pre_edit_gate_failures,
    _local_recolor_hard_gate_failures,
    _mask_refinement_geometry_checks,
    _merge_constraint_check,
    _mark_accepted_local_edit,
    _merge_localized_repair_hint,
    _object_insertion_hard_gate_failures,
    _object_insertion_region,
    _observation_from_existing_feedback,
    _ocr_repair_failures,
    _question_level_constraints_passed,
    _route_all_count_failures_to_regenerate,
    _semantic_evaluation_error_type,
    _target_for_efficient_bbox_localization,
)
from src.prompt_constraints import approx_clip_token_count, extract_constraints
from src.editing_agent import EfficientRepairAgent, GroundedSAM2PowerPaintEditingAgent
from src.local_editor import InpaintRegion, MockInpaintEditor
from src.mask_refiner import MockMaskRefiner
from src.relation_repair import RelationActionRepairer
from src.reward_reranker import MockRewardBackend, RewardReranker
from src.state import AgentConfig, AgentState


class FixedRepairPlanner:
    def __init__(self, plan):
        self.plan_payload = dict(plan)
        self.calls = []

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.plan_payload)


def test_logging_writes_json_atomically(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    write_json(path, {"old": True})
    write_json(path, {"new": True})
    atomic_write_text(tmp_path / "report.md", "ok\n")

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "ok\n"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_count_failure_route_disables_local_repair_when_all_candidates_fail_count() -> None:
    plan = {
        "primary_action": "object_insertion",
        "tool_sequence": ["object_insertion"],
        "repairable": True,
        "target_object": "apples",
        "target_attribute": "count",
    }
    arbitration = {
        "ranking": [
            {
                "index": 0,
                "constraint_summary": {
                    "human_rule_tier": "reject_missing_or_wrong_count"
                },
            },
            {
                "index": 1,
                "constraint_summary": {
                    "human_rule_tier": "reject_missing_or_wrong_count"
                },
            },
        ]
    }

    routed = _route_all_count_failures_to_regenerate(
        plan,
        arbitration,
        can_regenerate=True,
    )
    blocked = _route_all_count_failures_to_regenerate(
        plan,
        arbitration,
        can_regenerate=False,
    )

    assert routed["primary_action"] == "regenerate"
    assert routed["tool_sequence"] == ["regenerate"]
    assert routed["repairable"] is False
    assert routed["source"] == "m6212_count_failure_route"
    assert blocked["primary_action"] == "none"
    assert blocked["tool_sequence"] == []


def test_ocr_repair_failures_only_block_when_ocr_is_available_and_failed() -> None:
    assert _ocr_repair_failures({}) == []
    assert _ocr_repair_failures({"ocr_verification": {"available": False}}) == []

    failures = _ocr_repair_failures(
        {
            "ocr_verification": {
                "available": True,
                "passed": False,
                "expected": "NO",
                "recognized": "ON",
                "similarity": 0.0,
            }
        }
    )

    assert failures == [
        {
            "type": "ocr_text_mismatch",
            "expected": "NO",
            "recognized": "ON",
            "similarity": 0.0,
            "message": "deterministic text repair failed OCR verification",
        }
    ]


def test_efficient_repair_gate_allows_deterministic_overlay_without_inpaint() -> None:
    gate = _efficient_repair_gate(
        "text_overlay",
        {"target_attribute": "exact text"},
        EfficientRepairAgent(inpaint_agent=None),
        canvas_size=(512, 512),
    )
    shape_gate = _efficient_repair_gate(
        "shape_overlay",
        {
            "target_object": "opaque panel",
            "target_attribute": "occlusion",
            "bbox": [100, 300, 220, 120],
        },
        EfficientRepairAgent(inpaint_agent=None),
        canvas_size=(512, 512),
    )

    assert gate["allowed"] is True
    assert shape_gate["allowed"] is True


def test_efficient_repair_gate_requires_backend_and_good_bbox_for_inpaint() -> None:
    no_backend = _efficient_repair_gate(
        "bbox_shape_inpaint",
        {
            "target_object": "red screen",
            "bbox": [100, 300, 220, 120],
            "bbox_confidence": 0.9,
        },
        EfficientRepairAgent(inpaint_agent=None),
        canvas_size=(512, 512),
    )
    low_confidence = _efficient_repair_gate(
        "bbox_shape_inpaint",
        {
            "target_object": "red screen",
            "bbox": [100, 300, 220, 120],
            "bbox_confidence": 0.32,
        },
        EfficientRepairAgent(
            inpaint_agent=object(),  # type: ignore[arg-type]
        ),
        canvas_size=(512, 512),
    )

    assert no_backend["allowed"] is False
    assert "inpaint backend" in no_backend["reason"]
    assert low_confidence["allowed"] is False
    assert "confidence" in low_confidence["reason"]


def test_efficient_repair_target_locator_augments_missing_forbidden_bbox(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (120, 120, 120)).save(image_path)
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "found": True,
                    "bbox": [10, 12, 14, 8],
                    "confidence": 0.88,
                    "reason": "visible zipper pull on backpack",
                }
            )
        ]
    )
    agent = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=vlm,
        image_generator=MockImageGenerator(),
        enable_vlm_target_locator=True,
        efficient_repair_agent=EfficientRepairAgent(
            inpaint_agent=GroundedSAM2PowerPaintEditingAgent(
                editor=MockInpaintEditor()
            )
        ),
        enable_efficient_repair_agent=True,
        runs_dir=tmp_path / "runs",
    )
    constraints = extract_constraints(
        "A yellow backpack has no visible zipper pull and no side pocket."
    )
    plan = {
        "typed_route": "forbidden_object_removal",
        "primary_action": "object_insertion",
        "target_object": "no visible zipper pull",
        "target_attribute": "forbidden_object",
        "reason": "zipper pull is visible",
    }
    events: list[dict] = []

    updated = agent._augment_efficient_repair_plan_with_target_bbox(
        route="existing_object_inpaint",
        repair_plan=plan,
        selected_image=str(image_path),
        selected_prompt=constraints.original_prompt,
        user_prompt=constraints.original_prompt,
        constraints=constraints,
        run_dir=tmp_path,
        events=events,
        round_index=0,
    )

    assert updated["target_object"] == "zipper pull"
    assert updated["target_bbox"] == [10, 12, 14, 8]
    assert updated["bbox_confidence"] == 0.88
    assert events[0]["type"] == "efficient_repair_target_locator"


def test_forbidden_localization_target_strips_negative_words() -> None:
    constraints = extract_constraints(
        "A yellow backpack has no visible zipper pull and no side pocket."
    )
    assert (
        _target_for_efficient_bbox_localization(
            {"typed_route": "forbidden_object_removal", "target_object": "no visible zipper pull"},
            constraints,
        )
        == "zipper pull"
    )


def test_efficient_repair_gate_rejects_spatial_and_count_inpaint() -> None:
    agent = EfficientRepairAgent(inpaint_agent=object())  # type: ignore[arg-type]

    spatial = _efficient_repair_gate(
        "existing_object_inpaint",
        {
            "target_object": "yellow pyramid",
            "bbox": [60, 60, 150, 150],
            "target_attribute": "spatial_relation",
        },
        agent,
        canvas_size=(512, 512),
    )
    count = _efficient_repair_gate(
        "bbox_shape_inpaint",
        {
            "target_object": "mug",
            "bbox": [60, 60, 150, 150],
            "target_attribute": "count",
        },
        agent,
        canvas_size=(512, 512),
    )

    assert spatial["allowed"] is False
    assert "spatial" in spatial["reason"]
    assert count["allowed"] is False
    assert "count" in count["reason"]


def test_efficient_repair_gate_allows_safe_occlusion_bbox_inpaint() -> None:
    gate = _efficient_repair_gate(
        "bbox_shape_inpaint",
        {
            "target_object": "red screen",
            "typed_route": "occlusion_object_insertion",
            "bbox": [100, 300, 220, 120],
            "bbox_confidence": 0.75,
        },
        EfficientRepairAgent(
            inpaint_agent=object(),  # type: ignore[arg-type]
        ),
        canvas_size=(512, 512),
    )

    assert gate["allowed"] is True
    assert gate["area_ratio"] > 0


def test_efficient_repair_gate_derives_bbox_for_typed_occlusion_without_layout() -> None:
    gate = _efficient_repair_gate(
        "shape_overlay",
        {
            "target_object": "screen",
            "typed_route": "occlusion_object_insertion",
            "target_region": "lower_half",
            "target_attribute": "occlusion",
            "occlusion_spec": {
                "occluder": "screen",
                "target": "suitcase",
                "hidden_part": "lower half",
                "visible_part": "suitcase handle",
            },
        },
        EfficientRepairAgent(
            inpaint_agent=object(),  # type: ignore[arg-type]
        ),
        canvas_size=(512, 512),
    )

    assert gate["allowed"] is True
    assert gate["bbox"] == [112, 296, 286, 112]
    assert gate["area_ratio"] > 0


def test_efficient_repair_request_uses_typed_occlusion_prompt_and_bbox(tmp_path: Path) -> None:
    constraints = extract_constraints(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )

    request = _efficient_repair_request_from_plan(
        "shape_overlay",
        {
            "target_object": "screen",
            "typed_route": "occlusion_object_insertion",
            "target_region": "lower_half",
            "target_attribute": "occlusion",
            "reason": "The red screen is missing.",
            "occlusion_spec": {
                "occluder": "screen",
                "target": "suitcase",
                "hidden_part": "lower half",
                "visible_part": "suitcase handle",
            },
        },
        "/tmp/source.png",
        constraints.original_prompt,
        constraints,
        output_dir=tmp_path / "edit",
        canvas_size=(512, 512),
    )

    assert request.repair_kind == "shape_overlay"
    assert request.bbox == [112, 296, 286, 112]
    assert request.target_object == "screen"
    assert "foreground occluder" in request.prompt
    assert "lower half" in request.prompt
    assert "preserve the visible suitcase handle" in request.prompt
    assert request.negative_prompt == ""


def test_configure_generator_output_dir_resolves_absolute_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project"
    flux_repo = tmp_path / "flux"
    project_dir.mkdir()
    flux_repo.mkdir()
    monkeypatch.chdir(project_dir)
    generator = MockImageGenerator()
    setattr(generator, "output_dir", Path("old-relative-output"))

    _configure_mock_placeholder_dir(generator, Path("runs") / "case" / "images")

    assert getattr(generator, "output_dir") == (
        project_dir / "runs" / "case" / "images"
    ).resolve()
    assert getattr(generator, "output_dir") != (
        flux_repo / "runs" / "case" / "images"
    ).resolve()


def test_generation_prompt_guard_removes_wrong_interaction_target() -> None:
    constraints = extract_constraints(
        "A cyan cat holds a red umbrella handle while sitting beside a purple "
        "teapot, no color leakage between the objects."
    )
    prompt = (
        "the subject visibly holds the teapot, cyan cat, red umbrella handle, "
        "purple teapot, front paws gripping a slender red umbrella handle"
    )

    guarded, event = _guard_prompt_relation_drift(
        prompt,
        constraints,
        0,
        strategy="single",
    )

    assert event is not None
    assert "holds the teapot" not in guarded.lower()
    assert "cat holds umbrella handle" in guarded.lower()
    assert event["removed"] == ["holds the teapot"]


def test_generation_prompt_guard_is_generic_for_carry_relation() -> None:
    constraints = extract_constraints(
        "A green wizard carries a silver lantern while standing beside an orange barrel."
    )
    prompt = (
        "the subject visibly carries the barrel, green wizard, silver lantern, "
        "orange barrel"
    )

    guarded, event = _guard_prompt_relation_drift(
        prompt,
        constraints,
        0,
        strategy="single",
    )

    assert event is not None
    assert "carries the barrel" not in guarded.lower()
    assert "wizard carries lantern" in guarded.lower()
    assert event["removed"] == ["carries the barrel"]


def test_specialist_positive_feedback_keeps_structured_spatial_relations() -> None:
    constraints = extract_constraints(
        "A blue notebook shows a yellow star symbol on its cover, next to a "
        "plain green notebook with no symbol."
    )

    observation = _observation_from_existing_feedback({}, constraints)

    assert {
        "subject": "blue notebook",
        "phrase": "next to",
        "object": "green notebook",
        "passed": True,
        "confidence": 0.51,
        "evidence": "No specialist failure evidence found in existing feedback.",
    } in observation["spatial_relations"]


def test_specialist_feedback_marks_spatial_failure_from_existing_errors() -> None:
    constraints = extract_constraints(
        "A gray dog stands behind a teal bench, while a pink ball rests under the bench."
    )

    observation = _observation_from_existing_feedback(
        {
            "errors": [
                {
                    "type": "wrong_relation",
                    "target": "bench",
                    "evidence": "The pink ball is resting on top of the teal bench, not under it.",
                }
            ]
        },
        constraints,
    )

    under = [
        item
        for item in observation["spatial_relations"]
        if item["subject"] == "ball" and item["phrase"] == "under"
    ][0]
    assert under["passed"] is False
    assert "not under" in under["evidence"]


def test_specialist_feedback_marks_count_failure_from_existing_errors() -> None:
    constraints = extract_constraints(
        "Exactly four blue fish swim through one orange hoop, with no fifth fish "
        "and no extra hoop."
    )

    observation = _observation_from_existing_feedback(
        {
            "errors": [
                {
                    "type": "wrong_count",
                    "target": "fish",
                    "expected": "4",
                    "observed": "5",
                    "evidence": "There are five blue fish visible.",
                }
            ]
        },
        constraints,
    )

    fish = [item for item in observation["subjects"] if item["name"] == "fish"][0]
    assert fish["visible"] is True
    assert fish["count"] == 5
    assert "five blue fish" in fish["evidence"]


def test_attribute_feedback_targets_nearest_object_color_pair() -> None:
    constraints = extract_constraints(
        "A blue monkey touches the top of a green drum while holding a silver "
        "spoon in its other hand."
    )
    critique = {
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": (
                    "The spoon held by the monkey is not silver; it has a gold "
                    "handle with a silver bowl."
                ),
            }
        ]
    }

    observation = _observation_from_existing_feedback(critique, constraints)
    failed = [
        item
        for item in observation["attributes"]
        if item.get("passed") is False
    ]

    assert [item["object"] for item in failed] == ["spoon"]


def test_passed_question_check_downgrades_conflicting_top_level_relation_error() -> None:
    critique = {
        "score": 0.8,
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": (
                    "The yellow bird is perched on the bicycle's handlebar, "
                    "not above the bicycle seat as specified."
                ),
                "prompt_span": "",
            }
        ],
        "user_grounded": True,
    }
    constraint_check = {
        "source": "question_level_vqa",
        "passed": True,
        "score": 1.0,
        "errors": [],
        "checks": [
            {
                "category": "spatial_relation",
                "question_id": "relation:bird:bicycle_seat:above",
                "target": "bicycle seat",
                "expected": "yes",
                "passed": True,
            }
        ],
    }

    merged = _merge_constraint_check(critique, constraint_check)

    assert merged["errors"] == []
    assert merged["score"] == 1.0
    assert merged["user_grounded"] is False
    assert merged["soft_evaluation_errors"][0]["type"] == "wrong_relation"
    assert merged["judge_disagreements"][0]["resolution"] == (
        "question_level_hard_constraints_passed"
    )


def test_passed_negative_relation_check_downgrades_top_level_attachment_error() -> None:
    critique = {
        "score": 0.8,
        "errors": [
            {
                "type": "wrong_relation",
                "evidence": (
                    "The lantern is incorrectly depicted as attached to the orange barrel, "
                    "violating the instruction that it is not attached to the barrel."
                ),
                "prompt_span": "the lantern is not attached to the barrel",
            }
        ],
        "user_grounded": True,
    }
    constraint_check = {
        "source": "question_level_vqa",
        "passed": True,
        "score": 1.0,
        "errors": [],
        "checks": [
            {
                "category": "negative_relation",
                "question_id": "negative_relation:lantern:barrel:attached_to",
                "target": "barrel",
                "expected": "yes",
                "passed": True,
            }
        ],
    }

    merged = _merge_constraint_check(critique, constraint_check)

    assert merged["errors"] == []
    assert merged["score"] == 1.0
    assert merged["soft_evaluation_errors"][0]["type"] == "wrong_relation"


def test_merge_localized_repair_hint_adds_bbox_and_route_to_plan() -> None:
    plan = {
        "primary_action": "none",
        "tool_sequence": [],
        "repairable": False,
    }
    critique = {
        "constraint_check": {
            "localized_errors": [
                {
                    "repair_kind": "text_overlay",
                    "target_object": "top sign",
                    "target_attribute": "exact text",
                    "bbox": [105, 45, 305, 210],
                    "bbox_confidence": 0.82,
                    "expected": "yellow text 'NO'",
                    "repair_instruction": "render exact yellow text NO",
                }
            ]
        }
    }

    merged = _merge_localized_repair_hint(plan, critique)

    assert merged["primary_action"] == "efficient_repair"
    assert merged["tool_sequence"] == ["text_overlay"]
    assert merged["bbox"] == [105, 45, 305, 210]
    assert merged["target_object"] == "top sign"
    assert merged["localized_repair_hint"]["repair_kind"] == "text_overlay"


def test_completion_score_uses_hard_pass_when_only_soft_eval_errors_remain() -> None:
    critique = {
        "score": 0.8,
        "constraint_check": {
            "source": "question_level_vqa",
            "passed": True,
            "score": 1.0,
            "errors": [],
            "checks": [
                {
                    "category": "entity_existence",
                    "question_id": "existence:vase",
                    "target": "vase",
                    "expected": "yes",
                    "passed": True,
                }
            ],
        },
        "evaluation": {
            "passed": True,
            "score": 0.9,
            "errors": [
                {
                    "type": "wrong_attribute",
                    "severity": "minor",
                    "prompt_span": "smooth glass vase",
                    "evidence": "The vase has a slight tint.",
                }
            ],
        },
        "soft_evaluation_errors": [
            {
                "type": "wrong_attribute",
                "severity": "minor",
                "prompt_span": "smooth glass vase",
                "evidence": "The vase has a slight tint.",
            }
        ],
        "errors": [],
    }

    assert _completion_score(critique) == pytest.approx(0.9)


def test_count_evidence_labeled_attribute_stays_hard_under_question_pass() -> None:
    error = {
        "type": "wrong_attribute",
        "prompt_span": "amber glass vase",
        "evidence": (
            "The vase is amber glass, but the prompt specifies a single amber "
            "glass vase. The image contains two vases."
        ),
    }
    constraint_check = {
        "source": "question_level_vqa",
        "passed": True,
        "score": 1.0,
        "errors": [],
        "checks": [
            {
                "category": "count",
                "question_id": "count:glass_vase",
                "target": "glass vase",
                "expected": "1",
                "observed": "1",
                "passed": True,
            },
            {
                "category": "color_binding",
                "question_id": "color:glass_vase",
                "target": "glass vase",
                "expected": "amber",
                "observed": "amber",
                "passed": True,
            },
        ],
    }
    critique = {
        "score": 0.6,
        "constraint_check": constraint_check,
        "evaluation": {"passed": False, "score": 0.6, "errors": [error]},
        "errors": [error],
        "user_grounded": True,
    }
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=MockVLMClient(),
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        score_threshold=0.85,
    )

    gate = orchestrator.completion_gate(critique, 0)

    assert _semantic_evaluation_error_type(error) == "wrong_count"
    assert gate["passed"] is False
    assert {item["type"] for item in gate["blockers"]} >= {
        "evaluation_failed",
        "user_grounded_error",
    }


def test_low_hard_constraint_score_blocks_completion_even_without_errors(
    tmp_path: Path,
) -> None:
    critique = {
        "score": 0.9,
        "constraint_check": {
            "source": "question_level_vqa",
            "passed": True,
            "score": 0.44,
            "errors": [],
            "checks": [
                {
                    "category": "entity_existence",
                    "question_id": "existence:robot",
                    "target": "robot",
                    "expected": "yes",
                    "passed": True,
                }
            ],
        },
        "evaluation": {"passed": True, "score": 0.9, "errors": []},
        "errors": [],
    }
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=MockVLMClient(),
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        score_threshold=0.85,
    )

    gate = orchestrator.completion_gate(critique, 0)

    assert _question_level_constraints_passed(critique["constraint_check"]) is False
    assert gate["passed"] is False
    assert gate["score"] == pytest.approx(0.44)
    assert any(
        item["type"] == "constraint_score_below_threshold"
        for item in gate["blockers"]
    )


def test_orchestrator_runs_two_rounds_and_writes_artifacts(tmp_path: Path) -> None:
    llm = MockLLMClient(
        responses=[
            "not json belief",
            json.dumps({"prompts": ["a cinematic red robot holding a blue umbrella"]}),
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": (
                                "a cinematic red robot clearly holding a blue umbrella"
                            ),
                            "prompt": (
                                "a cinematic red robot clearly holding a blue umbrella, "
                                "rainy street, sharp focus"
                            ),
                            "fixes": ["missing_object"],
                            "expected_improvement": "Adds the umbrella explicitly.",
                        }
                    ]
                }
            ),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.7,
                                "attribute_binding": 0.7,
                                "object_relationship": 0.7,
                                "background_consistency": 0.6,
                                "aesthetic": 0.7,
                            },
                            "reason": "Good starting prompt.",
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.5}]}),
            json.dumps(
                {
                    "score": 0.45,
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The blue umbrella is missing.",
                            "prompt_span": "a cinematic red robot holding a blue umbrella",
                        }
                    ],
                    "strengths": ["Red robot is visible."],
                    "revision_hint": "Make the blue umbrella explicit and visible.",
                }
            ),
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.95,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                            "reason": "Umbrella is explicit.",
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.92,
                    "errors": [],
                    "strengths": ["Robot and umbrella match the prompt."],
                    "revision_hint": "No major revision needed.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(
            human_in_loop=False,
            creativity_level="high",
            n_images=1,
            max_rounds=2,
            seed=7,
        ),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run("a red robot holding a blue umbrella", run_id="m4-test")
    run_dir = Path(result.run_dir)

    assert result.status == "completed"
    assert result.mode == "mock"
    assert len(result.round_records) == 2
    assert "a cinematic red robot clearly holding a blue umbrella" in result.state[
        "refined_prompt"
    ]
    assert generator.calls[0]["prompt"] == "a cinematic red robot holding a blue umbrella"
    assert "a cinematic red robot clearly" in generator.calls[1]["prompt"]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "state_round_0.json").exists()
    assert (run_dir / "state_round_1.json").exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "final_report.md").exists()
    assert len(list((run_dir / "images").glob("*.txt"))) == 2

    run_log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert run_log["config"]["max_rounds"] == 2
    assert run_log["round_records"][0]["feedback"]["score"] == 0.45
    assert "Final score: 0.920" in (run_dir / "final_report.md").read_text(
        encoding="utf-8"
    )
    assert any(event["type"] == "prompt_optimized" for event in result.events)
    assert len(llm.calls) == 3
    assert len(vlm.calls) == 6


def test_orchestrator_specialist_patch_gate_blocks_relation_drift(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "A cyan cat holds a red umbrella handle while sitting beside a purple "
        "teapot, no color leakage between the objects."
    )
    drifted_prompt = (
        "the subject visibly holds the teapot, A cyan cat sitting beside a "
        "purple teapot, red umbrella handle"
    )
    llm = MockLLMClient(responses=[json.dumps({"prompts": [drifted_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.62,
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": (
                                "No umbrella handle is present; the cat is holding "
                                "a teapot handle instead."
                            ),
                            "prompt_span": "red umbrella handle",
                        }
                    ],
                    "strengths": ["The cyan cat and purple teapot are visible."],
                    "revision_hint": "Replace the teapot handle with a red umbrella handle.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(n_images=1, max_rounds=2, seed=7),
        runs_dir=tmp_path,
        enable_clarifier=False,
        score_threshold=0.85,
    )

    result = orchestrator.run(user_prompt, run_id="specialist-patch-gate")
    run_dir = Path(result.run_dir)
    specialist_path = run_dir / "specialist_reports_round_0.json"
    specialist = json.loads(specialist_path.read_text(encoding="utf-8"))
    revised_prompt = result.round_records[0]["revised_prompt"]

    assert specialist_path.exists()
    assert specialist["api_call_count"] == 0
    assert specialist["arbitration"]["dominant_failure"] == "interaction_relation"
    assert specialist["arbitration"]["selected_action"] == (
        "relation_repair_or_object_insertion"
    )
    assert "teapot handle" in specialist["arbitration"]["forbidden_phrases"]
    assert "separate red umbrella handle" in specialist["arbitration"]["prompt_patch"]
    assert "teapot handle" not in revised_prompt
    assert "separate red umbrella handle" in revised_prompt
    assert any(
        event["type"] == "prompt_specialist_patch_gated"
        for event in result.events
    )
    assert any(event["type"] == "specialist_reports" for event in result.events)


def test_orchestrator_can_use_one_structured_vlm_specialist_observation(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "A cyan cat holds a red umbrella handle while sitting beside a purple "
        "teapot, no color leakage between the objects."
    )
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    structured_observation = {
        "subjects": [
            {"name": "cat", "visible": True, "count": 1, "confidence": 0.9},
            {"name": "teapot", "visible": True, "count": 1, "confidence": 0.9},
            {
                "name": "umbrella handle",
                "visible": False,
                "count": 0,
                "confidence": 0.88,
                "evidence": "No separate umbrella handle is visible.",
            },
        ],
        "attributes": [
            {
                "object": "cat",
                "attribute": "color",
                "expected": "cyan",
                "observed": "cyan",
                "passed": True,
                "confidence": 0.9,
            },
            {
                "object": "teapot",
                "attribute": "color",
                "expected": "purple",
                "observed": "purple",
                "passed": True,
                "confidence": 0.9,
            },
            {
                "object": "umbrella handle",
                "attribute": "color",
                "expected": "red",
                "observed": "red teapot handle",
                "passed": False,
                "confidence": 0.9,
                "evidence": "The red handle is part of the teapot.",
            },
        ],
        "spatial_relations": [],
        "interaction_relations": [
            {
                "subject": "cat",
                "action": "holds",
                "object": "teapot handle",
                "passed": False,
                "confidence": 0.9,
                "evidence": "The cat holds a teapot handle rather than a separate umbrella handle.",
                "confused_with": "teapot handle",
            }
        ],
        "negative_constraints": [
            {
                "constraint": "no color leakage between the objects",
                "passed": False,
                "confidence": 0.8,
                "evidence": "The red handle is visually attached to the purple teapot.",
            }
        ],
        "summary": {
            "global_passed": False,
            "dominant_failure": "interaction",
            "repair_hint": "Make the cat hold a separate red umbrella handle.",
        },
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.72,
                                "attribute_binding": 0.72,
                                "object_relationship": 0.72,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.72}]}),
            json.dumps(
                {
                    "score": 0.74,
                    "errors": [],
                    "strengths": ["The cat and teapot are visible."],
                    "revision_hint": "No broad critique.",
                }
            ),
            json.dumps(structured_observation),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(n_images=1, max_rounds=2, seed=7),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_specialist_vlm_observation=True,
        score_threshold=0.85,
    )

    result = orchestrator.run(user_prompt, run_id="specialist-vlm-observation")
    run_dir = Path(result.run_dir)
    specialist = json.loads(
        (run_dir / "specialist_reports_round_0.json").read_text(encoding="utf-8")
    )

    assert specialist["source"] == "structured_vlm_observation"
    assert specialist["api_call_count"] == 1
    assert "structured visual observation module" in specialist["request"]
    assert specialist["arbitration"]["dominant_failure"] == "interaction_relation"
    assert "teapot handle" in specialist["arbitration"]["forbidden_phrases"]
    assert any(
        event["type"] == "specialist_reports" and event["api_call_count"] == 1
        for event in result.events
    )
    assert any(
        event["type"] == "prompt_specialist_patch_gated"
        for event in result.events
    )


def test_orchestrator_logs_fusion_candidates_and_agents(tmp_path: Path) -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps({"prompts": ["a cyan cat holding a red umbrella handle"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                            "reason": "Prompt is clear.",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "selected_index": 0,
                    "scores": [
                        {"index": 0, "score": 0.9},
                        {"index": 1, "score": 0.75},
                    ],
                }
            ),
            json.dumps(
                {
                    "score": 0.91,
                    "errors": [],
                    "strengths": ["The relation is visible."],
                    "revision_hint": "No revision needed.",
                }
            ),
        ]
    )
    generator = FusionImageGenerator(
        flux=MockImageGenerator(prefix="flux_mock"),
        sdxl=MockImageGenerator(prefix="sdxl_mock"),
        policy="parallel",
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=1, seed=7),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
    )

    result = orchestrator.run("a cyan cat holding a red umbrella handle", run_id="fusion-smoke")

    assert result.status == "completed"
    generated = next(item for item in result.events if item["type"] == "images_generated")
    assert generated["agent"] == "GenerationEngineAgent"
    assert len(generated["image_paths"]) == 2
    assert len(generated["image_prompts"]) == 2
    assert [item["backend"] for item in generated["generator_metadata"]] == ["flux", "sdxl"]
    assert any(item.get("agent") == "InputInterpreterAgent" for item in result.events)


def test_orchestrator_pauses_when_clarifier_needs_user_answer(tmp_path: Path) -> None:
    llm = MockLLMClient(default_response="not json")
    vlm = MockVLMClient()
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(creativity_level="medium", max_rounds=1),
        runs_dir=tmp_path,
    )

    result = orchestrator.run("a futuristic city", run_id="needs-clarification")

    assert result.status == "awaiting_clarification"
    assert result.round_records == []
    assert generator.calls == []
    assert len(llm.calls) == 2
    assert result.events[-1]["type"] == "clarification_decision"
    assert result.events[-1]["question"]
    run_dir = Path(result.run_dir)
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run.json").exists()
    assert not (run_dir / "state_round_0.json").exists()
    assert "No generation rounds were completed." in (
        run_dir / "final_report.md"
    ).read_text(encoding="utf-8")


def test_orchestrator_can_auto_merge_clarification_before_generation(
    tmp_path: Path,
) -> None:
    llm = MockLLMClient(
        responses=[
            "not json belief",
            "<question>What style and viewpoint should the city have?</question>",
            json.dumps(
                {
                    "merged_prompt": (
                        "a futuristic city, cinematic street-level view"
                    )
                }
            ),
            json.dumps({"prompts": ["cinematic street-level futuristic city"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.8,
                                "background_consistency": 0.9,
                                "aesthetic": 0.9,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 1.0}]}),
            json.dumps(
                {
                    "score": 0.91,
                    "errors": [],
                    "strengths": ["Clear style and viewpoint."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="medium", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        auto_merge_clarification="cinematic street-level view",
    )

    result = orchestrator.run("a futuristic city", run_id="auto-clarified")

    assert result.status == "completed"
    assert result.state["user_prompt"] == "a futuristic city, cinematic street-level view"
    assert result.round_records[0]["prompt"] == "cinematic street-level futuristic city"
    assert any(event["type"] == "clarification_merged" for event in result.events)


def test_create_run_dir_avoids_collisions(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path, run_id="same-id")
    second = create_run_dir(tmp_path, run_id="same-id")

    assert first.name == "same-id"
    assert second.name == "same-id-001"


def test_completed_when_critique_api_fails_but_verified_constraints_pass(
    tmp_path: Path,
) -> None:
    class FailingCritiqueVLM(MockVLMClient):
        def vision(self, prompt: str, image_paths: list[str]) -> str:
            if "Idea2Img-style refinement loop" in prompt:
                raise RuntimeError("API HTTP error 400: input length too large")
            return super().vision(prompt, image_paths)

    user_prompt = "a small red robot clearly gripping the handle of a blue umbrella"
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    vlm = FailingCritiqueVLM(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.95,
                                "attribute_binding": 0.95,
                                "object_relationship": 0.95,
                                "background_consistency": 0.9,
                                "aesthetic": 0.9,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.95}]}),
            json.dumps(
                {
                    "answers": [
                        {"id": "existence:robot", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:umbrella", "answer": "yes", "confidence": 1.0},
                        {"id": "count:robot", "answer": "1", "confidence": 1.0},
                        {"id": "count:umbrella", "answer": "1", "confidence": 1.0},
                        {"id": "color:robot", "answer": "red", "confidence": 1.0},
                        {"id": "color:umbrella", "answer": "blue", "confidence": 1.0},
                        {
                            "id": "part:umbrella_handle",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The umbrella handle is visible.",
                        },
                        {
                            "id": "relation:robot:umbrella_handle:gripping",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The robot claw wraps around the umbrella handle.",
                        },
                        {
                            "id": "relation:robot:umbrella:gripping",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The gripped handle belongs to the umbrella.",
                        },
                        {
                            "id": "action:robot:clearly_gripping",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The robot claw wraps around the handle.",
                        },
                        {
                            "id": "binding:robot:umbrella:gripping:blue",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The same blue umbrella handle is gripped by the robot claw.",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": True,
                    "criteria_scores": {
                        "alignment": 0.95,
                        "attribute_binding": 0.95,
                        "object_relationship": 0.95,
                    },
                    "errors": [],
                    "strengths": ["Hard user constraints are verified."],
                    "revision_hint": "No revision.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="critique-api-fail-verified-pass")
    feedback = result.round_records[0]["feedback"]

    assert result.status == "completed"
    assert any(event["type"] == "critique_failed" for event in result.events)
    assert feedback["failed"] is True
    assert feedback["score"] >= 0.95
    assert feedback["constraint_check"]["passed"] is True
    assert feedback["evaluation"]["passed"] is True
    assert feedback["completion_gate"]["passed"] is True
    assert feedback["completion_gate"]["blockers"] == []


def test_orchestrator_logs_m6_evaluation_and_reward_ranking(tmp_path: Path) -> None:
    llm = MockLLMClient(
        responses=[
            "not json belief",
            json.dumps({"prompts": ["a red robot holding a blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.8}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["VisualReflector accepts it."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "score": 0.55,
                    "criteria_scores": {"attribute_binding": 0.3},
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Fix the umbrella color.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        reward_reranker=RewardReranker(MockRewardBackend(default_score=0.88)),
        enable_reward_reranker=True,
    )

    result = orchestrator.run("a red robot holding a blue umbrella", run_id="m6-agent")
    run_dir = Path(result.run_dir)

    assert result.status == "max_rounds_reached"
    assert (run_dir / "reward_round_0.json").exists()
    assert (run_dir / "evaluation_round_0.json").exists()
    assert result.round_records[0]["feedback"]["score"] == 0.55
    assert any(event["type"] == "reward_reranked" for event in result.events)
    assert any(event["type"] == "evaluated" for event in result.events)


def test_orchestrator_uses_constraint_arbitration_before_reward_override(
    tmp_path: Path,
) -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "prompts": [
                        "a small red robot gripping the handle of a blue umbrella"
                    ]
                }
            )
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "selected_index": 0,
                    "scores": [
                        {"index": 0, "score": 0.95},
                        {"index": 1, "score": 0.9},
                        {"index": 2, "score": 0.85},
                    ],
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.6,
                    "checks": [
                        {
                            "type": "color",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "red",
                            "passed": False,
                        }
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Make the umbrella blue.",
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.92,
                    "checks": [
                        {
                            "type": "color",
                            "target": "robot",
                            "expected": "red",
                            "observed": "red",
                            "passed": True,
                        },
                        {
                            "type": "color",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "blue",
                            "passed": True,
                        },
                    ],
                    "errors": [],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.7,
                    "checks": [
                        {
                            "type": "relation",
                            "target": "robot-umbrella",
                            "expected": "gripping handle",
                            "observed": "not touching",
                            "passed": False,
                        }
                    ],
                    "errors": [],
                    "revision_hint": "Show contact with the handle.",
                }
            ),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["The second image preserves the color binding."],
                    "revision_hint": "No change.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.92,
                    "checks": [
                        {
                            "type": "color",
                            "target": "umbrella",
                            "passed": True,
                        }
                    ],
                    "errors": [],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(creativity_level="high", n_images=3, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        reward_reranker=RewardReranker(MockRewardBackend(default_score=0.95)),
        enable_reward_reranker=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(
        "a small red robot gripping the handle of a blue umbrella",
        run_id="m68-arbitration",
    )
    run_dir = Path(result.run_dir)
    constraint_event = next(
        event for event in result.events if event["type"] == "constraint_aware_selection"
    )
    candidate_log = json.loads(
        (run_dir / "candidate_constraints_round_0.json").read_text(encoding="utf-8")
    )
    question_log = json.loads(
        (run_dir / "candidate_questions_round_0.json").read_text(encoding="utf-8")
    )
    selection_trace = json.loads(
        (run_dir / "selection_trace_round_0.json").read_text(encoding="utf-8")
    )

    assert result.status == "completed"
    assert result.round_records[0]["selected_image"].endswith("mock_image_0001.txt")
    assert constraint_event["selected_index"] == 1
    assert constraint_event["reward_selected_index"] == 0
    assert constraint_event["overrode_reward_selection"] is True
    assert candidate_log["selected_index"] == 1
    assert Path(candidate_log["selection_trace_path"]).name == "selection_trace_round_0.json"
    assert len(candidate_log["candidate_checks"]) == 3
    assert question_log["selected_index"] == 1
    assert Path(question_log["selection_trace_path"]).name == "selection_trace_round_0.json"
    assert len(question_log["candidate_questions"]) == 3
    assert question_log["candidate_questions"][1]["source"] == "legacy_constraint_check"
    assert selection_trace["selected_index"] == 1
    assert selection_trace["policy"].startswith("VLM/reward are evidence providers")
    assert result.round_records[0]["feedback"]["constraint_arbitration"][
        "overrode_reward_selection"
    ] is True


def test_candidate_question_log_preserves_object_state_geometry(
    tmp_path: Path,
) -> None:
    prompt = "a white cup to the left of three red apples on a wooden table"
    vqa_response = {
        "answers": [
            {"id": "existence:cup", "answer": "yes", "confidence": 1.0},
            {"id": "existence:apples", "answer": "yes", "confidence": 1.0},
            {"id": "existence:table", "answer": "yes", "confidence": 1.0},
            {"id": "count:apples", "answer": "3", "confidence": 1.0},
            {"id": "color:cup", "answer": "white", "confidence": 1.0},
            {"id": "color:apples", "answer": "red", "confidence": 1.0},
            {"id": "relation:cup:apples:left_of", "answer": "yes", "confidence": 1.0},
            {"id": "relation:apples:table:on", "answer": "yes", "confidence": 1.0},
        ]
    }
    bbox_response = {
        "objects": [
            {"name": "cup", "visible": True, "bbox": [100, 100, 80, 80]},
            {"name": "apples", "visible": True, "bbox": [240, 100, 120, 100]},
        ]
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps(vqa_response),
            json.dumps(bbox_response),
            json.dumps(vqa_response),
            json.dumps(bbox_response),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(n_images=2, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
    )
    image_paths = [str(tmp_path / "candidate0.png"), str(tmp_path / "candidate1.png")]

    decision = orchestrator._maybe_arbitrate_image_selection(
        AgentState(user_prompt=prompt),
        [prompt, prompt],
        image_paths,
        {"selected_index": 0, "selected_image": image_paths[0]},
        None,
        extract_constraints(prompt),
        tmp_path,
        [],
        0,
    )
    question_log = json.loads(
        (tmp_path / "candidate_questions_round_0.json").read_text(encoding="utf-8")
    )
    first_record = question_log["candidate_questions"][0]

    assert decision is not None
    assert first_record["object_state"]["available"] is True
    assert first_record["object_state"]["objects"][0]["name"] == "cup"
    assert first_record["geometry_verification"]["passed"] is True
    assert first_record["evidence_chain"]
    assert any(step["question_id"] == "relation:cup:apples:left_of" for step in first_record["evidence_chain"])
    assert any(
        check["category"] == "spatial_geometry"
        for check in first_record["constraint_check"]["checks"]
    )


def test_orchestrator_does_not_complete_when_evaluator_fails_hard_constraints(
    tmp_path: Path,
) -> None:
    llm = MockLLMClient(
        responses=[json.dumps({"prompts": ["a red robot holding a blue umbrella"]})]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.95}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["The image is cinematic."],
                    "revision_hint": "No ordinary revision.",
                }
            ),
            json.dumps(
                {
                    "score": 0.85,
                    "passed": False,
                    "criteria_scores": {"attribute_binding": 0.6},
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red, but the user specified blue.",
                            "prompt_span": "blue umbrella",
                            "severity": "major",
                        }
                    ],
                    "revision_hint": "Make the umbrella blue.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a red robot holding a blue umbrella", run_id="m65-gate")
    feedback = result.round_records[0]["feedback"]
    stop_event = next(event for event in result.events if event["type"] == "stop")

    assert result.status == "max_rounds_reached"
    assert feedback["score"] == pytest.approx(0.85)
    assert feedback["evaluation"]["passed"] is False
    assert feedback["completion_gate"]["passed"] is False
    assert feedback["completion_gate"]["score_passed"] is True
    assert {item["type"] for item in feedback["completion_gate"]["blockers"]} >= {
        "evaluation_failed",
        "user_grounded_error",
    }
    assert stop_event["reason"] == "max_rounds_reached"
    assert stop_event["completion_gate"]["passed"] is False


def test_question_level_hard_pass_downgrades_non_contradictory_evaluator_errors(
    tmp_path: Path,
) -> None:
    user_prompt = "a woman wearing a yellow hat holding a black cat"
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.95,
                                "attribute_binding": 0.95,
                                "object_relationship": 0.95,
                                "background_consistency": 0.9,
                                "aesthetic": 0.9,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.95}]}),
            json.dumps(
                {
                    "score": 0.95,
                    "errors": [],
                    "strengths": ["Woman, hat, and cat are visible."],
                    "revision_hint": "No hard revision.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "answers": [
                        {"id": "existence:woman", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:hat", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:cat", "answer": "yes", "confidence": 1.0},
                        {"id": "count:woman", "answer": "1", "confidence": 1.0},
                        {"id": "count:hat", "answer": "1", "confidence": 1.0},
                        {"id": "count:cat", "answer": "1", "confidence": 1.0},
                        {"id": "color:hat", "answer": "yellow", "confidence": 1.0},
                        {"id": "color:cat", "answer": "black", "confidence": 1.0},
                        {
                            "id": "relation:woman:cat:holding",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                            {
                                "id": "relation:woman:hat:wearing",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                            {
                                "id": "binding:woman:cat:holding:black",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                            {
                                "id": "binding:woman:hat:wearing:yellow",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                        ]
                    }
                ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": False,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The cat's eyes appear slightly lighter than expected for a black cat.",
                            "prompt_span": "black cat",
                        },
                        {
                            "type": "wrong_attribute",
                            "evidence": "The cat's fur appears glossy rather than matte.",
                            "prompt_span": "black cat",
                        },
                    ],
                    "revision_hint": "Make the cat eyes darker and fur matte.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="soft-evaluator-errors")
    feedback = result.round_records[0]["feedback"]

    assert result.status == "completed"
    assert feedback["constraint_check"]["passed"] is True
    assert feedback["evaluation"]["passed"] is False
    assert feedback["completion_gate"]["passed"] is True
    assert feedback.get("errors", []) == []
    assert len(feedback["soft_evaluation_errors"]) == 2


def test_question_level_hard_pass_stops_before_repair_and_specialist_regression(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "A purple astronaut grips a gold rope while standing near a white ladder; "
        "the rope is not attached to the ladder."
    )
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.92,
                                "attribute_binding": 0.92,
                                "object_relationship": 0.92,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["Astronaut, rope, and ladder are visible."],
                    "revision_hint": "No ordinary critique.",
                }
            ),
            json.dumps(
                {
                    "answers": [
                        {"id": "existence:astronaut", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:rope", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:ladder", "answer": "yes", "confidence": 1.0},
                        {"id": "count:rope", "answer": "1", "confidence": 1.0},
                        {"id": "count:ladder", "answer": "1", "confidence": 1.0},
                        {"id": "color:astronaut", "answer": "purple", "confidence": 1.0},
                        {"id": "color:rope", "answer": "gold", "confidence": 1.0},
                        {"id": "color:ladder", "answer": "white", "confidence": 1.0},
                        {
                            "id": "relation:astronaut:rope:grips",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                        {
                            "id": "relation:rope:ladder:near",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                        {"id": "action:astronaut:standing", "answer": "yes", "confidence": 1.0},
                        {
                            "id": "negative_relation:rope:ladder:attached_to",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                        {
                            "id": "binding:astronaut:rope:grips:gold",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "score": 0.3,
                    "passed": False,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The rope is depicted as a mix of gold and purple, violating the specified gold color.",
                            "prompt_span": "gold rope",
                        }
                    ],
                    "revision_hint": "Refine the rope to be entirely gold-colored.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=2),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="hard-pass-no-regression")
    feedback = result.round_records[0]["feedback"]
    run_payload = json.loads((Path(result.run_dir) / "run.json").read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert len(result.round_records) == 1
    assert feedback["constraint_check"]["passed"] is True
    assert feedback["evaluation"]["passed"] is False
    assert feedback["hard_pass_guard"]["reason"] == "question_level_hard_constraints_passed"
    assert feedback["completion_gate"]["passed"] is True
    assert feedback["completion_gate"]["blockers"] == []
    assert feedback["errors"] == []
    assert not (Path(result.run_dir) / "repair_plan_round_0.json").exists()
    assert not (Path(result.run_dir) / "specialist_reports_round_0.json").exists()
    assert any(event["type"] == "hard_pass_guard" for event in result.events)
    assert any(event["type"] == "specialist_reports_skipped" for event in result.events)
    assert run_payload["final_selection"]["hard_pass"] is True


def test_question_level_hard_pass_does_not_hide_missing_occluder_failure(
    tmp_path: Path,
) -> None:
    critique = {
        "score": 0.2,
        "constraint_check": {
            "source": "question_level_vqa",
            "passed": True,
            "score": 1.0,
            "errors": [],
            "checks": [
                {
                    "category": "occlusion_relation",
                    "question_id": "relation:screen:suitcase:hides_lower_half",
                    "target": "screen",
                    "expected": "yes",
                    "observed": "yes",
                    "passed": True,
                }
            ],
        },
        "evaluation": {
            "passed": False,
            "score": 0.2,
            "errors": [
                {
                    "type": "missing_object",
                    "evidence": (
                        "There is no visible red screen hiding the lower half "
                        "of the green suitcase."
                    ),
                    "prompt_span": "red screen",
                }
            ],
            "revision_hint": "Add the missing red screen occluder.",
        },
        "errors": [
            {
                "type": "missing_object",
                "evidence": (
                    "There is no visible red screen hiding the lower half "
                    "of the green suitcase."
                ),
                "prompt_span": "red screen",
            }
        ],
        "user_grounded": True,
    }
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=MockVLMClient(),
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        score_threshold=0.85,
    )

    gate = orchestrator.completion_gate(critique, 0)

    assert _hard_pass_guard(critique) is None
    assert gate["passed"] is False
    assert {item["type"] for item in gate["blockers"]} >= {
        "evaluation_failed",
        "user_grounded_error",
    }


def test_question_level_hard_pass_does_not_hide_forbidden_object_failure(
    tmp_path: Path,
) -> None:
    critique = {
        "score": 0.2,
        "constraint_check": {
            "source": "question_level_vqa",
            "passed": True,
            "score": 1.0,
            "errors": [],
            "checks": [
                {
                    "category": "negative_relation",
                    "question_id": "negative_relation:scene:bowl:absent",
                    "target": "bowl",
                    "expected": "absent",
                    "observed": "absent",
                    "passed": True,
                },
                {
                    "category": "negative_relation",
                    "question_id": "negative_relation:scene:spoon:absent",
                    "target": "spoon",
                    "expected": "absent",
                    "observed": "absent",
                    "passed": True,
                },
            ],
        },
        "evaluation": {
            "passed": False,
            "score": 0.2,
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": (
                        "The image contains a yellow bowl and spoon, which "
                        "violates the user's instruction that there should be "
                        "no bowl and no spoon nearby."
                    ),
                    "prompt_span": "no bowl and no spoon nearby",
                }
            ],
            "revision_hint": "Remove the forbidden bowl and spoon.",
        },
        "errors": [
            {
                "type": "wrong_attribute",
                "evidence": (
                    "The image contains a yellow bowl and spoon, which "
                    "violates the user's instruction that there should be "
                    "no bowl and no spoon nearby."
                ),
                "prompt_span": "no bowl and no spoon nearby",
            }
        ],
        "user_grounded": True,
    }
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=MockVLMClient(),
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        score_threshold=0.85,
    )

    gate = orchestrator.completion_gate(critique, 0)

    assert _semantic_evaluation_error_type(critique["evaluation"]["errors"][0]) == (
        "forbidden_object_present"
    )
    assert _hard_pass_guard(critique) is None
    assert gate["passed"] is False
    assert {item["type"] for item in gate["blockers"]} >= {
        "evaluation_failed",
        "user_grounded_error",
    }


def test_question_level_failures_filter_positive_evaluator_pseudo_errors(
    tmp_path: Path,
) -> None:
    user_prompt = "two yellow birds sitting on a black bicycle near a white dog"
    llm = MockLLMClient(
        responses=[
            json.dumps({"prompts": [user_prompt]}),
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": "two yellow birds clearly visible sitting on a black bicycle near a white dog",
                            "prompt": "two yellow birds clearly visible sitting on a black bicycle near a white dog",
                            "fixes": ["missing_object", "wrong_count"],
                            "expected_improvement": "Restore missing birds before tuning pose.",
                        }
                    ]
                }
            ),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.75,
                                "attribute_binding": 0.7,
                                "object_relationship": 0.65,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.7,
                    "errors": [],
                    "strengths": ["Bicycle and dog are visible."],
                    "revision_hint": "No ordinary critique.",
                }
            ),
            json.dumps(
                {
                    "answers": [
                        {
                            "id": "existence:birds",
                            "answer": "no",
                            "confidence": 1.0,
                            "evidence": "There are no visible birds.",
                        },
                        {
                            "id": "existence:bicycle",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "A bicycle is visible.",
                        },
                        {
                            "id": "existence:dog",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "A dog is visible.",
                        },
                        {
                            "id": "color:bicycle",
                            "answer": "black",
                            "confidence": 1.0,
                            "evidence": "The bicycle is black.",
                        },
                        {
                            "id": "color:dog",
                            "answer": "white",
                            "confidence": 1.0,
                            "evidence": "The dog is white.",
                        },
                        {
                            "id": "relation:bicycle:dog:near",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "The bicycle is near the dog.",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "score": 0.62,
                    "passed": False,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The bicycle is black, which matches the prompt. No error here.",
                            "prompt_span": "black bicycle",
                        },
                        {
                            "type": "wrong_attribute",
                            "evidence": "The dog is white as specified.",
                            "prompt_span": "white dog",
                        },
                        {
                            "type": "missing_object",
                            "evidence": "There are no visible birds.",
                            "prompt_span": "birds",
                        },
                    ],
                    "revision_hint": "Add the missing birds.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=2),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="positive-pseudo-errors")
    feedback = result.round_records[0]["feedback"]
    error_text = " ".join(str(item.get("evidence", "")) for item in feedback["errors"])

    assert result.status == "max_rounds_reached"
    assert feedback["constraint_check"]["passed"] is False
    assert "No error here" not in error_text
    assert "as specified" not in error_text
    assert any(item["type"] == "missing_object" for item in feedback["errors"])
    stop_event = next(event for event in reversed(result.events) if event["type"] == "stop")
    assert stop_event["completion_gate"]["passed"] is False
    assert "two yellow birds clearly visible" in result.round_records[0]["revised_prompt"]


def test_question_level_hard_pass_overrides_matching_count_pseudo_error(
    tmp_path: Path,
) -> None:
    user_prompt = "two yellow birds sitting on a black bicycle near a white dog"
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["Required subjects are visible."],
                    "revision_hint": "No ordinary critique.",
                }
            ),
            json.dumps(
                {
                    "answers": [
                        {"id": "existence:birds", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:bicycle", "answer": "yes", "confidence": 1.0},
                        {"id": "existence:dog", "answer": "yes", "confidence": 1.0},
                        {
                            "id": "count:birds",
                            "answer": "2",
                            "confidence": 1.0,
                            "evidence": "Two yellow birds are visible.",
                        },
                        {"id": "count:bicycle", "answer": "1", "confidence": 1.0},
                        {"id": "count:dog", "answer": "1", "confidence": 1.0},
                        {"id": "color:birds", "answer": "yellow", "confidence": 1.0},
                        {"id": "color:bicycle", "answer": "black", "confidence": 1.0},
                        {"id": "color:dog", "answer": "white", "confidence": 1.0},
                        {
                            "id": "relation:birds:bicycle:on",
                            "answer": "yes",
                            "confidence": 1.0,
                        },
                            {
                                "id": "relation:bicycle:dog:near",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                            {
                                "id": "binding:birds:bicycle:on:black",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                            {
                                "id": "binding:bicycle:dog:near:white",
                                "answer": "yes",
                                "confidence": 1.0,
                            },
                            {"id": "action:birds:sitting", "answer": "yes", "confidence": 1.0},
                        ]
                    }
                ),
            json.dumps(
                {
                    "score": 0.85,
                    "passed": False,
                    "errors": [
                        {
                            "type": "wrong_count",
                            "evidence": "There are only two yellow birds visible, but the prompt specifies two yellow birds.",
                            "prompt_span": "two yellow birds",
                        }
                    ],
                    "revision_hint": "No meaningful revision.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="matching-count-pseudo-error")
    feedback = result.round_records[0]["feedback"]

    assert result.status == "completed"
    assert feedback["constraint_check"]["passed"] is True
    assert feedback["errors"] == []
    assert feedback["completion_gate"]["passed"] is True
    assert len(feedback["soft_evaluation_errors"]) == 1


def test_question_level_errors_drive_prompt_optimization_priority(
    tmp_path: Path,
) -> None:
    user_prompt = "two yellow birds sitting on a black bicycle near a white dog"
    llm = MockLLMClient(
        responses=[
            json.dumps({"prompts": [user_prompt]}),
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": "two yellow birds clearly visible sitting on a black bicycle near a white dog",
                            "prompt": "two yellow birds clearly visible sitting on a black bicycle near a white dog",
                            "fixes": ["missing_object"],
                            "expected_improvement": "Restore the missing required subject first.",
                        }
                    ]
                }
            ),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.8,
                                "attribute_binding": 0.7,
                                "object_relationship": 0.6,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.7,
                    "errors": [
                        {
                            "type": "wrong_relation",
                            "evidence": "The scene should use the rear bicycle seat.",
                            "prompt_span": "birds sitting on a black bicycle",
                        }
                    ],
                    "strengths": ["The dog and bicycle are visible."],
                    "revision_hint": "Specify rear bicycle seat.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "answers": [
                        {
                            "id": "existence:birds",
                            "answer": "no",
                            "confidence": 1.0,
                            "evidence": "There are no visible birds.",
                        },
                        {
                            "id": "existence:bicycle",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "A black bicycle is visible.",
                        },
                        {
                            "id": "existence:dog",
                            "answer": "yes",
                            "confidence": 1.0,
                            "evidence": "A white dog is visible.",
                        },
                    ]
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=2),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="qa-priority")

    optimizer_request = llm.calls[-1]
    assert "Error type: missing_object" in optimizer_request
    assert "There are no visible birds" in optimizer_request
    assert "rear bicycle seat" not in optimizer_request
    assert "two yellow birds clearly visible" in result.round_records[0]["revised_prompt"]


def test_orchestrator_can_apply_m6_local_repair(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (96, 96), (0, 0, 0))
    for y in range(30, 54):
        for x in range(24, 72):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
                "canvas_size": [96, 96],
            "background": {"description": "rainy street", "viewpoint": "front view"},
            "objects": [
                    {
                        "name": "blue umbrella",
                        "description": "blue umbrella canopy",
                            "bbox": [24, 30, 48, 24],
                        "order": 1,
                        "relations": ["held by red robot"],
                        "requires_reference": False,
                }
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["red robot, blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.75,
                                "attribute_binding": 0.8,
                                "object_relationship": 0.45,
                                "background_consistency": 0.8,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [{"type": "wrong_attribute", "evidence": "The blue umbrella appears as a red umbrella."}],
                    "revision_hint": "Make the umbrella blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [{"type": "wrong_attribute", "evidence": "red umbrella, not blue"}],
                    "revision_hint": "Fix color.",
                }
            ),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["The umbrella is now blue."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.95,
                    "checks": [],
                    "errors": [],
                    "revision_hint": "Post-repair constraints pass.",
                }
            ),
        ]
    )
    generator = MockImageGenerator(existing_paths=[source])
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(96, 96),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        mask_refiner=MockMaskRefiner(),
        enable_mask_refiner=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a red robot holding a blue umbrella", run_id="m6-repair")
    run_dir = Path(result.run_dir)

    assert result.status == "completed"
    assert (run_dir / "local_repair_round_0.json").exists()
    assert (run_dir / "mask_refine_round_0.json").exists()
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))
    mask_refine = json.loads((run_dir / "mask_refine_round_0.json").read_text(encoding="utf-8"))
    assert mask_refine["result"]["method"] == "mock_mask_refiner"
    assert Path(mask_refine["result"]["mask_path"]).exists()
    assert repair["target_evidence"]["mask_refinement"]["result"]["mask_path"] == mask_refine["result"]["mask_path"]
    assert result.round_records[0]["feedback"]["repaired"] is True
    assert result.round_records[0]["selected_image"].endswith("recolor_image_0000.png")
    assert any(event["type"] == "local_repair" and event["accepted"] for event in result.events)
    assert any(event["type"] == "mask_refined" for event in result.events)


def test_orchestrator_can_use_vlm_target_locator_for_recolor_mask(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    for y in range(44, 58):
        for x in range(12, 44):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [80, 80],
            "background": {"description": "plain scene", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue umbrella",
                    "description": "planned umbrella bbox is stale",
                    "bbox": [45, 12, 24, 16],
                    "order": 1,
                    "relations": [],
                    "requires_reference": False,
                }
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.5,
                                "attribute_binding": 0.4,
                                "object_relationship": 0.7,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.5}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Make the umbrella blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Fix umbrella color.",
                }
            ),
            json.dumps(
                {
                    "found": True,
                    "bbox": [12, 44, 32, 14],
                    "confidence": 0.93,
                    "reason": "The visible red umbrella is in the lower-left image region.",
                }
            ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": True,
                    "errors": [],
                    "strengths": ["The edited target is blue."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.95,
                    "checks": [],
                    "errors": [],
                    "revision_hint": "Post-repair constraints pass.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(80, 80),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        enable_vlm_target_locator=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a blue umbrella", run_id="vlm-target-locator")
    run_dir = Path(result.run_dir)
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert repair["detection"]["target_localization"]["applied"] is True
    assert repair["detection"]["target_localization"]["bbox"] == [12, 44, 32, 14]
    assert repair["detection"]["constrained_bbox"] == [12, 44, 32, 14]
    assert repair["detection"]["mask_mode"] == "object_region"


def test_orchestrator_switches_to_repairable_base_candidate_for_recolor(
    tmp_path: Path,
) -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "prompts": [
                        "a small red robot gripping the handle of a blue umbrella"
                    ]
                }
            )
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.85,
                                "attribute_binding": 0.7,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 1, "scores": [{"index": 1, "score": 0.95}]}),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.62,
                    "checks": [
                        {
                            "type": "color",
                            "target": "robot",
                            "expected": "red",
                            "observed": "red",
                            "passed": True,
                        },
                        {
                            "type": "color",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "red",
                            "passed": False,
                        },
                        {
                            "type": "subject",
                            "target": "robot",
                            "expected": "small red robot",
                            "observed": "small red robot",
                            "passed": True,
                        },
                        {
                            "type": "subject",
                            "target": "handle",
                            "expected": "handle of a blue umbrella",
                            "observed": "handle of a red umbrella",
                            "passed": False,
                        },
                        {
                            "type": "action",
                            "target": "gripping",
                            "expected": "clearly gripping",
                            "observed": "clearly gripping",
                            "passed": True,
                        },
                        {
                            "type": "relation",
                            "target": "robot-umbrella",
                            "expected": "robot gripping the umbrella handle",
                            "observed": "robot gripping the umbrella handle",
                            "passed": True,
                        },
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "umbrella",
                        }
                    ],
                    "revision_hint": "Recolor the umbrella blue.",
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 1.0,
                    "checks": [
                        {
                            "type": "color-object binding",
                            "target": "robot",
                            "expected": "red",
                            "observed": "red",
                            "passed": True,
                        },
                        {
                            "type": "color-object binding",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "blue",
                            "passed": True,
                        },
                    ],
                    "errors": [],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "score": 0.8,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "strengths": ["The robot is red."],
                    "revision_hint": "Make the umbrella blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.8,
                    "passed": False,
                    "criteria_scores": {"attribute_binding": 0.6},
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Make the umbrella blue.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=2, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella",
        run_id="repairability-base-selection",
    )
    run_dir = Path(result.run_dir)
    base_log = json.loads(
        (run_dir / "repair_base_round_0.json").read_text(encoding="utf-8")
    )

    assert result.status == "max_rounds_reached"
    assert base_log["current_index"] == 1
    assert base_log["selected_index"] == 0
    assert base_log["overrode_current_selection"] is True
    assert base_log["rejected_candidates"][0]["index"] == 1
    assert result.round_records[0]["feedback"]["repair_base_selection"][
        "selected_index"
    ] == 0
    assert not (run_dir / "repairable_candidate_round_0.json").exists()
    assert result.round_records[0]["selected_image"].endswith("mock_image_0000.txt")
    assert any(
        event["type"] == "repair_base_selected" and event["selected_index"] == 0
        for event in result.events
    )


def test_non_local_regenerate_plan_does_not_switch_final_image_via_repair_base(
    tmp_path: Path,
) -> None:
    user_prompt = "two yellow birds sitting on a black bicycle near a white dog"
    llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    candidate0 = {
        "passed": False,
        "score": 0.3,
        "checks": [
            {"type": "subject", "target": "birds", "expected": "yes", "passed": False},
            {"type": "subject", "target": "bicycle", "expected": "yes", "passed": True},
            {"type": "subject", "target": "dog", "expected": "yes", "passed": True},
        ],
        "errors": [
            {
                "type": "missing_object",
                "evidence": "There are no visible birds.",
                "prompt_span": "birds",
            }
        ],
        "revision_hint": "Regenerate with birds.",
    }
    candidate1 = {
        "passed": False,
        "score": 0.5,
        "checks": [
            {"type": "subject", "target": "birds", "expected": "yes", "passed": True},
            {"type": "wrong_count", "target": "birds", "expected": "2", "passed": False},
            {"type": "subject", "target": "dog", "expected": "yes", "passed": True},
        ],
        "errors": [
            {
                "type": "wrong_count",
                "evidence": "Only one bird is visible.",
                "prompt_span": "birds",
            }
        ],
        "revision_hint": "Regenerate with exactly two birds.",
    }
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.8,
                                "object_relationship": 0.8,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.8}]}),
            json.dumps(candidate0),
            json.dumps(candidate1),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "There are no visible birds.",
                            "prompt_span": "birds",
                        }
                    ],
                    "strengths": ["Bicycle and dog are visible."],
                    "revision_hint": "Regenerate with two yellow birds.",
                    "user_grounded": True,
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(creativity_level="high", n_images=2, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_constraint_check=True,
        enable_repair_planner=True,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
    )

    result = orchestrator.run(user_prompt, run_id="regenerate-no-base-switch")
    run_dir = Path(result.run_dir)

    assert result.status == "max_rounds_reached"
    assert not (run_dir / "repair_base_round_0.json").exists()
    arbitration_event = next(
        event for event in result.events if event["type"] == "constraint_aware_selection"
    )
    assert result.round_records[0]["selected_image"] == arbitration_event["selected_image"]
    assert not any(event["type"] == "repair_base_selected" for event in result.events)


def test_orchestrator_rechecks_constraints_after_accepted_recolor(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    for y in range(18, 34):
        for x in range(16, 62):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [80, 80],
            "background": {"description": "plain scene", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue canopy",
                    "description": "blue canopy",
                    "bbox": [16, 18, 46, 28],
                    "order": 1,
                    "relations": [],
                    "requires_reference": False,
                }
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a blue canopy"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.5,
                                "attribute_binding": 0.4,
                                "object_relationship": 0.6,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.4}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The blue canopy appears red.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Make the canopy blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The canopy is red instead of blue.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Fix canopy color.",
                }
            ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": True,
                    "errors": [],
                    "strengths": ["The edited image looks blue."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.3,
                    "checks": [
                        {
                            "type": "color-object binding",
                            "target": "canopy",
                            "expected": "blue",
                            "observed": "background patch edited, target still wrong",
                            "passed": False,
                            "description": "The local edit is not verified on the target object.",
                        }
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The local edit is not verified on the target object.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Reject this local repair.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(80, 80),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a blue canopy", run_id="post-repair-check")
    run_dir = Path(result.run_dir)
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))

    assert repair["accepted"] is False
    assert repair["post_repair_constraint_check"]["passed"] is False
    failure_types = {
        item["type"] for item in repair["acceptance"]["hard_gate_failures"]
    }
    assert "post_repair_constraint_failed" in failure_types
    assert result.round_records[0]["selected_image"] == str(source)
    assert any(
        event["type"] == "local_repair" and event["accepted"] is False
        for event in result.events
    )


def test_orchestrator_keeps_repaired_image_when_post_check_api_fails(
    tmp_path: Path,
) -> None:
    from PIL import Image

    class FailingPostCheckVLM(MockVLMClient):
        def __init__(self, responses):
            super().__init__(responses=responses)
            self._vision_calls = 0

        def vision(self, prompt, image_paths):
            self._vision_calls += 1
            if self._vision_calls == 6:
                raise RuntimeError(
                    "API HTTP error 400: Range of input length should be [1, 129024]"
                )
            return super().vision(prompt, image_paths)

    source = tmp_path / "source.png"
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    for y in range(18, 34):
        for x in range(16, 62):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [80, 80],
            "background": {"description": "plain scene", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue canopy",
                    "description": "blue canopy",
                    "bbox": [16, 18, 46, 28],
                    "order": 1,
                    "relations": [],
                    "requires_reference": False,
                }
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a blue canopy"]}),
        ]
    )
    vlm = FailingPostCheckVLM(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.5,
                                "attribute_binding": 0.4,
                                "object_relationship": 0.6,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.4}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The blue canopy appears red.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Make the canopy blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The canopy is red instead of blue.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Fix canopy color.",
                }
            ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": True,
                    "errors": [],
                    "strengths": ["The edited image looks blue."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(80, 80),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a blue canopy", run_id="post-check-api-failed")
    run_dir = Path(result.run_dir)
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))

    assert result.status == "max_rounds_reached"
    assert repair["accepted"] is True
    assert repair["acceptance"]["verification_unavailable"] is True
    assert repair["post_repair_constraint_check"]["failed"] is True
    assert result.round_records[0]["selected_image"].endswith("recolor_image_0000.png")
    gate = result.round_records[0]["feedback"]["completion_gate"]
    assert gate["passed"] is False
    assert any(item["type"] == "constraint_check_unavailable" for item in gate["blockers"])


def test_orchestrator_rejects_recolor_that_breaks_other_user_color(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (64, 64), (0, 0, 0))
    for y in range(12, 32):
        for x in range(12, 52):
            image.putpixel((x, y), (220, 20, 20))
    for y in range(30, 54):
        for x in range(24, 42):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [64, 64],
            "background": {"description": "rainy street", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue canopy",
                    "description": "blue canopy",
                    "bbox": [12, 12, 40, 40],
                    "order": 1,
                    "relations": ["held by red figure"],
                    "requires_reference": False,
                },
                {
                    "name": "red figure",
                    "description": "small red figure",
                    "bbox": [24, 30, 18, 24],
                    "order": 2,
                    "relations": ["holding canopy"],
                    "requires_reference": False,
                },
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a red figure holding a blue canopy"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.7,
                                "attribute_binding": 0.4,
                                "object_relationship": 0.7,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The blue canopy appears red.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Make the canopy blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The canopy is red instead of blue.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Fix canopy color.",
                }
            ),
            json.dumps(
                {
                    "score": 0.91,
                    "passed": False,
                    "checks": [
                        {
                            "type": "color-object binding",
                            "target": "figure",
                            "expected": "red",
                            "observed": "blue",
                            "passed": False,
                            "description": "The red figure became blue during the repair.",
                        }
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The red figure became blue during the repair.",
                            "prompt_span": "red figure",
                        }
                    ],
                    "revision_hint": "Preserve the red figure while changing only the canopy.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(64, 64),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a red figure holding a blue canopy", run_id="reject-repair")
    run_dir = Path(result.run_dir)
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))

    assert result.status == "max_rounds_reached"
    assert repair["accepted"] is False
    assert repair["acceptance"]["color_preservation_errors"]
    assert result.round_records[0]["selected_image"] == str(source)
    assert any(event["type"] == "local_repair" and not event["accepted"] for event in result.events)


def test_orchestrator_rejects_recolor_with_low_layout_overlap(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    for y in range(42, 70):
        for x in range(8, 38):
            image.putpixel((x, y), (220, 20, 20))
    image.save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [80, 80],
            "background": {"description": "plain scene", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue canopy",
                    "description": "blue canopy",
                    "bbox": [40, 20, 32, 32],
                    "order": 1,
                    "relations": [],
                    "requires_reference": False,
                }
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a blue canopy"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.4,
                                "attribute_binding": 0.4,
                                "object_relationship": 0.7,
                                "background_consistency": 0.7,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.4}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The blue canopy appears red.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Make the canopy blue.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The canopy is red instead of blue.",
                            "prompt_span": "blue canopy",
                        }
                    ],
                    "revision_hint": "Fix canopy color.",
                }
            ),
            json.dumps(
                {
                    "score": 0.95,
                    "passed": True,
                    "errors": [],
                    "strengths": ["The edited image looks blue."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(80, 80),
        evaluator=VLMJudgeEvaluator(vlm),
        enable_evaluator=True,
        enable_local_repair=True,
        score_threshold=0.85,
    )

    result = orchestrator.run("a blue canopy", run_id="low-overlap-repair")
    run_dir = Path(result.run_dir)
    repair = json.loads((run_dir / "local_repair_round_0.json").read_text(encoding="utf-8"))

    assert repair["accepted"] is False
    assert repair["acceptance"]["hard_gate_failures"]
    failure_types = {
        item["type"] for item in repair["acceptance"]["hard_gate_failures"]
    }
    assert (
        {"layout_only_bbox_not_on_source_target", "low_target_color_gain", "low_component_target_color_gain"}
        & failure_types
    )
    assert repair["detection"]["mask_mode"] == "object_region"
    assert repair["detection"]["color_geometry"]["failures"]
    assert result.round_records[0]["selected_image"] == str(source)


def test_orchestrator_can_apply_relation_action_repair(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (12, 12, 12)).save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [64, 64],
            "background": {"description": "rainy street", "viewpoint": "back view"},
            "objects": [
                {
                    "name": "small red robot",
                    "description": "robot body and claw",
                    "bbox": [22, 30, 20, 24],
                    "order": 1,
                    "relations": ["gripping umbrella handle"],
                },
                {
                    "name": "blue umbrella",
                    "description": "blue umbrella and black handle",
                    "bbox": [12, 8, 44, 30],
                    "order": 2,
                    "relations": ["held by robot"],
                },
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a small red robot gripping a blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.75,
                                "attribute_binding": 0.8,
                                "object_relationship": 0.45,
                                "background_consistency": 0.8,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.46,
                    "errors": [
                        {
                            "type": "wrong_relation",
                            "evidence": "The hand is detached and not gripping the umbrella handle.",
                            "prompt_span": "gripping the handle",
                        }
                    ],
                    "revision_hint": "Make visible physical contact with the handle.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "found": True,
                    "bbox": [22, 32, 20, 20],
                    "confidence": 0.9,
                    "reason": "The visible claw and handle contact should be repaired here.",
                }
            ),
            json.dumps(
                {
                    "score": 0.91,
                    "passed": True,
                    "checks": {
                        "handle_visible": True,
                        "hand_or_claw_visible": True,
                        "visible_grip": True,
                        "physical_contact": True,
                        "handle_connected_to_umbrella": True,
                        "not_merely_near_or_supported": True,
                        "user_colors_preserved": True,
                    },
                    "strengths": ["The claw visibly grips the umbrella handle."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.91,
                    "checks": [
                        {
                            "type": "color",
                            "target": "robot",
                            "expected": "red",
                            "observed": "red",
                            "passed": True,
                        },
                        {
                            "type": "color",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "blue",
                            "passed": True,
                        },
                        {
                            "type": "relation",
                            "target": "robot and umbrella handle",
                            "expected": "clearly gripping",
                            "observed": "visible contact",
                            "passed": True,
                        },
                    ],
                    "errors": [],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    relation_repairer = RelationActionRepairer(
        vlm,
        MockInpaintEditor(prefix="relation_test"),
        candidates=1,
        pass_threshold=0.82,
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(64, 64),
        enable_relation_repair=True,
        relation_repairer=relation_repairer,
        score_threshold=0.85,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella",
        run_id="relation-repair",
    )
    run_dir = Path(result.run_dir)

    assert result.status == "completed"
    assert (run_dir / "relation_repair_round_0.json").exists()
    assert result.round_records[0]["feedback"]["relation_repaired"] is True
    assert result.round_records[0]["feedback"]["score"] == pytest.approx(0.91)
    assert result.round_records[0]["feedback"]["constraint_check"]["source"] == (
        "post_relation_repair_constraint_check"
    )
    assert result.round_records[0]["selected_image"].endswith("relation_test_image_0000.txt")
    assert any(
        event["type"] == "relation_action_repair" and event["accepted"]
        for event in result.events
    )
    assert any(
        event["type"] == "post_relation_repair_constraint_check" and event["passed"]
        for event in result.events
    )


def test_repair_planner_routes_missing_subject_to_object_insertion_not_relation(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (12, 12, 12)).save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [64, 64],
            "background": {"description": "rainy street", "viewpoint": "front view"},
            "objects": [
                {
                    "name": "blue umbrella",
                    "description": "blue umbrella with visible handle",
                    "bbox": [12, 8, 44, 30],
                    "order": 1,
                    "relations": ["should be held by robot"],
                },
                {
                    "name": "small red robot",
                    "description": "missing red robot target",
                    "bbox": [22, 30, 20, 24],
                    "order": 2,
                    "relations": ["gripping blue umbrella handle"],
                },
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a small red robot gripping a blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.65,
                                "attribute_binding": 0.6,
                                "object_relationship": 0.25,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.6}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The umbrella is visible, but no robot is visible.",
                            "prompt_span": "small red robot",
                        }
                    ],
                    "revision_hint": "Add the missing small red robot.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.9,
                    "checks": [
                        {
                            "type": "subject",
                            "target": "robot",
                            "expected": "small red robot",
                            "observed": "small red robot",
                            "passed": True,
                        },
                        {
                            "type": "subject",
                            "target": "umbrella",
                            "expected": "blue umbrella",
                            "observed": "blue umbrella",
                            "passed": True,
                        },
                    ],
                    "errors": [],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    relation_repairer = RelationActionRepairer(
        vlm,
        MockInpaintEditor(prefix="object_test"),
        candidates=1,
        pass_threshold=0.82,
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(64, 64),
        enable_relation_repair=False,
        enable_object_insertion_repair=True,
        relation_repairer=relation_repairer,
        score_threshold=0.85,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella",
        run_id="object-insertion-routing",
    )
    run_dir = Path(result.run_dir)

    assert result.status == "completed"
    assert (run_dir / "repair_plan_round_0.json").exists()
    assert (run_dir / "object_repair_round_0.json").exists()
    assert not (run_dir / "relation_repair_round_0.json").exists()
    repair_plan = json.loads(
        (run_dir / "repair_plan_round_0.json").read_text(encoding="utf-8")
    )
    assert repair_plan["primary_action"] == "object_insertion"
    object_repair = json.loads(
        (run_dir / "object_repair_round_0.json").read_text(encoding="utf-8")
    )
    assert object_repair["accepted"] is True
    assert object_repair["repair_plan"]["target_object"] == "robot"
    assert result.round_records[0]["selected_image"].endswith("object_test_image_0000.txt")
    assert any(event["type"] == "object_insertion_repair" for event in result.events)


def test_orchestrator_triggers_efficient_typed_occlusion_repair_without_layout(
    tmp_path: Path,
) -> None:
    from PIL import Image

    user_prompt = (
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (220, 20, 20)).save(source)
    llm = MockLLMClient(
        responses=[
            json.dumps({"prompts": [user_prompt]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.45}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "missing_object",
                            "target": "screen",
                            "evidence": "There is no visible red screen occluding the suitcase.",
                        }
                    ],
                    "revision_hint": "Add a red screen over the lower half.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.92,
                    "checks": [
                        {
                            "category": "occlusion_relation",
                            "target": "screen",
                            "expected": "yes",
                            "observed": "yes",
                            "passed": True,
                        }
                    ],
                    "errors": [],
                }
            ),
        ]
    )
    base_editor = MockInpaintEditor(prefix="should_not_run_efficient_occlusion")
    efficient_agent = EfficientRepairAgent(
        inpaint_agent=GroundedSAM2PowerPaintEditingAgent(editor=base_editor)
    )
    repair_plan = {
        "primary_action": "object_insertion",
        "tool_sequence": ["object_insertion"],
        "repairable": True,
        "typed_route": "occlusion_object_insertion",
        "target_object": "screen",
        "target_attribute": "occlusion",
        "target_region": "lower_half",
        "reason": "The red screen is missing.",
        "occlusion_spec": {
            "occluder": "screen",
            "target": "suitcase",
            "hidden_part": "lower half",
            "visible_part": "suitcase handle",
        },
    }
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_evaluator=False,
        enable_constraint_check=False,
        enable_object_insertion_repair=True,
        relation_repairer=RelationActionRepairer(
            vlm,
            MockInpaintEditor(prefix="should_not_run_object"),
            candidates=1,
        ),
        enable_efficient_repair_agent=True,
        efficient_repair_agent=efficient_agent,
        repair_planner=FixedRepairPlanner(repair_plan),
        score_threshold=0.85,
    )

    result = orchestrator.run(user_prompt, run_id="efficient-occlusion-routing")
    run_dir = Path(result.run_dir)

    assert result.status == "completed"
    assert len(result.round_records) == 1
    assert (run_dir / "efficient_repair_round_0.json").exists()
    assert not (run_dir / "object_repair_round_0.json").exists()
    repair = json.loads(
        (run_dir / "efficient_repair_round_0.json").read_text(encoding="utf-8")
    )
    assert repair["route"] == "shape_overlay"
    assert repair["repair_plan"]["typed_route"] == "occlusion_object_insertion"
    assert repair["accepted"] is True
    assert repair["post_repair_constraint_check"]["passed"] is True
    assert repair["gpu_used"] is False
    assert repair["powerpaint_used"] is False
    assert base_editor.calls == []
    assert result.round_records[0]["selected_image"].endswith(
        "shape_overlay_repair.png"
    )
    assert result.round_records[0]["feedback"]["completion_gate"]["passed"] is True
    assert result.round_records[0]["feedback"]["object_repaired"] is True
    assert any(event["type"] == "efficient_repair" for event in result.events)


def test_efficient_repair_attempted_editing_backend_only_for_inpaint_routes() -> None:
    assert _efficient_repair_attempted_editing_backend(None) is False
    assert _efficient_repair_attempted_editing_backend({"route": "text_overlay", "ok": True}) is False
    assert _efficient_repair_attempted_editing_backend(
        {"route": "shape_overlay", "ok": False, "error": "post-check failed"}
    ) is True
    assert _efficient_repair_attempted_editing_backend(
        {"route": "bbox_shape_inpaint", "ok": True}
    ) is True
    assert _efficient_repair_attempted_editing_backend(
        {"route": "existing_object_inpaint", "powerpaint_used": True}
    ) is True
    assert _efficient_repair_attempted_editing_backend(
        {"route": "bbox_shape_inpaint", "ok": False, "error": "mask path missing"}
    ) is True


def test_accepted_efficient_edit_clears_stale_pre_edit_evaluator_failure() -> None:
    critique = {
        "score": 0.3,
        "errors": [
            {
                "type": "missing_object",
                "evidence": "The red screen is missing.",
                "prompt_span": "red screen",
            }
        ],
        "evaluation": {
            "passed": False,
            "score": 0.3,
            "errors": [
                {
                    "type": "missing_object",
                    "evidence": "The red screen is missing.",
                    "prompt_span": "red screen",
                }
            ],
        },
        "constraint_check": {
            "passed": True,
            "score": 1.0,
            "errors": [],
        },
        "user_grounded": True,
    }

    repaired = _mark_accepted_local_edit(
        critique,
        {
            "route": "shape_overlay",
            "edited_image": "shape_overlay_repair.png",
            "repair_kind": "occlusion_object_insertion",
        },
    )
    gate = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=MockVLMClient(),
        image_generator=MockImageGenerator(),
        score_threshold=0.85,
    ).completion_gate(repaired, 0)

    assert repaired["evaluation"]["passed"] is True
    assert repaired["errors"] == []
    assert repaired["user_grounded"] is False
    assert gate["passed"] is True
    assert gate["blockers"] == []


def test_best_final_selection_prefers_completion_pass_over_hard_only_old_image() -> None:
    selected = _best_final_selection(
        [
            {
                "round": 0,
                "prompt": "original",
                "selected_image": "old.png",
                "feedback": {
                    "score": 1.0,
                    "completion_gate": {
                        "passed": False,
                        "score": 1.0,
                        "blockers": [{"type": "evaluation_failed"}],
                    },
                    "constraint_check": {
                        "passed": True,
                        "score": 1.0,
                        "question_summary": {"passed": True, "hard_failures": 0},
                    },
                },
            },
            {
                "round": 1,
                "prompt": "edited",
                "selected_image": "edit.png",
                "feedback": {
                    "score": 1.0,
                    "completion_gate": {"passed": True, "score": 1.0, "blockers": []},
                    "constraint_check": {"passed": True, "score": 1.0},
                },
            },
        ]
    )

    assert selected is not None
    assert selected["selected_image"] == "edit.png"
    assert selected["completion_passed"] is True


def test_object_insertion_hard_gate_rejects_large_local_edit_region() -> None:
    region = InpaintRegion(
        name="blue umbrella",
        bbox=[375, 116, 410, 588],
        prompt="add blue umbrella",
        canvas_size=[1024, 1024],
    )
    failures = _object_insertion_hard_gate_failures(
        region,
        {
            "scaled_bbox": [281, 87, 320, 441],
            "source_image_size": [768, 768],
        },
    )

    failure_types = {item["type"] for item in failures}
    assert "object_insertion_region_too_large" in failure_types


def test_object_insertion_region_allows_occlusion_without_layout() -> None:
    region = _object_insertion_region(
        None,
        "screen",
        {
            "typed_route": "occlusion_object_insertion",
            "target_region": "lower_half",
            "occlusion_spec": {
                "target": "suitcase",
                "hidden_part": "lower half",
                "visible_part": "suitcase handle",
            },
        },
        prompt="add the missing red screen",
        negative_prompt="changed suitcase handle",
        canvas_size=(1024, 1024),
    )

    assert region.name == "screen"
    assert region.bbox == [225, 593, 573, 225]
    assert "lower half of the suitcase" in region.prompt
    assert "suitcase handle" in region.prompt


def test_object_insertion_region_requires_layout_for_non_occlusion() -> None:
    with pytest.raises(ValueError, match="requires layout_context"):
        _object_insertion_region(
            None,
            "robot",
            {"typed_route": "missing_required_object"},
            prompt="add the missing robot",
            negative_prompt="",
            canvas_size=(1024, 1024),
        )


def test_local_repair_pre_edit_gate_rejects_layout_only_off_target_bbox() -> None:
    failures = _local_repair_pre_edit_gate_failures(
        {
            "method": "object_region_near_layout",
            "target_region": "canopy",
            "constrained_bbox": [315, 135, 240, 200],
            "mask_refinement": {
                "result": {
                    "method": "bbox_fallback",
                    "prompt_bbox": [307, 127, 256, 216],
                }
            },
        },
        {
            "source_color": "red",
            "source_coverage": 0.005,
            "target_coverage": 0.05,
        },
        {
            "target_name": "umbrella",
            "source_color": "red",
            "target_color": "#1d63d9",
        },
    )

    assert failures
    assert failures[0]["type"] == "layout_only_bbox_not_on_source_target"
    assert failures[0]["bbox_provenance"]["bbox_source"] == "layout_prior"
    assert failures[0]["bbox_provenance"]["image_grounded"] is False


def test_local_repair_pre_edit_gate_allows_image_grounded_locator_bbox() -> None:
    failures = _local_repair_pre_edit_gate_failures(
        {
            "method": "object_region_near_layout",
            "target_localization": {
                "applied": True,
                "bbox": [150, 320, 300, 160],
            },
            "mask_refinement": {
                "result": {
                    "method": "bbox_fallback",
                    "prompt_bbox": [150, 320, 300, 160],
                }
            },
        },
        {
            "source_color": "red",
            "source_coverage": 0.005,
        },
        {
            "target_name": "umbrella",
            "source_color": "red",
            "target_color": "#1d63d9",
        },
    )

    assert failures == []


def test_local_repair_pre_edit_gate_rejects_failed_target_locator() -> None:
    failures = _local_repair_pre_edit_gate_failures(
        {
            "method": "object_region_near_layout",
            "target_localization": {
                "applied": False,
                "error": "API HTTP error 400",
            },
            "bbox_provenance": {
                "bbox_source": "layout_prior",
                "image_grounded": False,
                "selected_bbox": [307, 127, 256, 216],
            },
        },
        {
            "source_color": "red",
            "source_coverage": 0.2,
        },
        {
            "target_name": "umbrella",
            "source_color": "red",
            "target_color": "#1d63d9",
        },
    )

    assert any(
        item["type"] == "target_locator_not_applied_pre_edit"
        for item in failures
    )


def test_local_recolor_hard_gate_rejects_full_object_source_residual() -> None:
    failures = _local_recolor_hard_gate_failures(
        {
            "selected_overlap_ratio": 1.0,
            "geometry_failures": [],
        },
        {
            "target_coverage": 0.0,
            "source_coverage": 0.95,
        },
        {
            "target_coverage": 0.92,
            "source_coverage": 0.0,
        },
        component_coverage_before={
            "target_coverage": 0.0,
            "source_coverage": 0.95,
        },
        component_coverage_after={
            "target_coverage": 0.92,
            "source_coverage": 0.0,
        },
        full_object_coverage_before={
            "target_coverage": 0.0,
            "source_coverage": 0.9,
        },
        full_object_coverage_after={
            "target_coverage": 0.5,
            "source_coverage": 0.52,
        },
    )

    failure_types = {item["type"] for item in failures}
    assert "high_full_object_source_color_residual" in failure_types


def test_local_recolor_hard_gate_rejects_partial_full_object_target_coverage() -> None:
    failures = _local_recolor_hard_gate_failures(
        {
            "selected_overlap_ratio": 1.0,
            "geometry_failures": [],
        },
        {
            "target_coverage": 0.05,
            "source_coverage": 0.36,
        },
        {
            "target_coverage": 0.89,
            "source_coverage": 0.0,
        },
        component_coverage_before={
            "target_coverage": 0.05,
            "source_coverage": 0.36,
        },
        component_coverage_after={
            "target_coverage": 0.89,
            "source_coverage": 0.0,
        },
        full_object_coverage_before={
            "target_coverage": 0.03,
            "source_coverage": 0.19,
        },
        full_object_coverage_after={
            "target_coverage": 0.47,
            "source_coverage": 0.0,
        },
    )

    failure_types = {item["type"] for item in failures}
    assert "low_full_object_target_color_coverage" in failure_types


def test_mask_refinement_geometry_rejects_large_bbox_fallback_mask() -> None:
    check = _mask_refinement_geometry_checks(
        {
            "method": "bbox_fallback",
            "fallback_used": True,
            "selected_pixel_count": 120142,
            "area_ratio": 0.203691,
            "geometry": {"mask_to_bbox_ratio": 1.0},
        }
    )

    assert check["passed"] is False
    assert any(
        item["type"] == "bbox_fallback_mask_too_large" and item["threshold"] == 0.12
        for item in check["failures"]
    )


def test_mask_refinement_geometry_allows_large_sam_object_mask() -> None:
    check = _mask_refinement_geometry_checks(
        {
            "method": "sam_v1_bbox_prompt",
            "fallback_used": False,
            "selected_pixel_count": 105000,
            "area_ratio": 0.178,
            "geometry": {"mask_to_bbox_ratio": 0.62},
            "protected_overlap": {"overlap_ratio": 0.01},
        }
    )

    assert check["passed"] is True


def test_mask_refinement_geometry_rejects_degenerate_sam_bbox_mask() -> None:
    check = _mask_refinement_geometry_checks(
        {
            "method": "sam_v1_bbox_prompt",
            "fallback_used": False,
            "selected_pixel_count": 120142,
            "area_ratio": 0.203691,
            "geometry": {"mask_to_bbox_ratio": 0.97},
            "protected_overlap": {"overlap_ratio": 0.01},
        }
    )

    assert check["passed"] is False
    assert any(
        item["type"] == "shape_refined_mask_degenerated_to_bbox"
        for item in check["failures"]
    )


def test_mask_refinement_geometry_rejects_empty_target_prior_overlap() -> None:
    check = _mask_refinement_geometry_checks(
        {
            "method": "sam_v1_bbox_prompt",
            "fallback_used": False,
            "selected_pixel_count": 0,
            "area_ratio": 0.0,
            "geometry": {"mask_to_bbox_ratio": 0.0},
            "protected_overlap": {"overlap_ratio": 0.0},
            "prior_constraint": {
                "applied": True,
                "prior_mask_path": "target_prior.png",
                "raw_mask_path": "sam_mask.png",
                "constrained_pixel_count": 0,
            },
        }
    )

    assert check["passed"] is False
    failure_types = {item["type"] for item in check["failures"]}
    assert "target_prior_constrained_mask_empty" in failure_types


def test_recolor_plan_uses_repair_planner_target_object() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )
    plan = _first_recolor_repair_plan(
        constraints,
        {
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The umbrella is brown instead of blue, while the robot remains red.",
                    "prompt_span": "umbrella",
                }
            ]
        },
        preferred_object="umbrella",
    )

    assert plan is not None
    assert plan["target_name"] == "umbrella"
    assert plan["target_color_name"] == "blue"
    assert plan["target_color"] == "#1d63d9"


def test_recolor_plan_prefers_explicit_wrong_color_over_low_saturation_terms() -> None:
    constraints = extract_constraints(
        "a small red robot holding a blue umbrella, rainy street photo"
    )
    plan = _first_recolor_repair_plan(
        constraints,
        {
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": (
                        "The umbrella is red instead of blue, with glistening "
                        "raindrops and metallic reflections nearby."
                    ),
                    "prompt_span": "blue umbrella",
                }
            ]
        },
        preferred_object="umbrella",
    )

    assert plan is not None
    assert plan["source_color"] == "red"


def test_recolor_plan_prefers_structured_observed_color_over_scene_words() -> None:
    constraints = extract_constraints(
        "a yellow bird sitting on a black bicycle near a white dog"
    )
    plan = _first_recolor_repair_plan(
        constraints,
        {
            "constraint_check": {
                "checks": [
                    {
                        "question_id": "color:bicycle",
                        "category": "color_binding",
                        "target": "bicycle",
                        "passed": False,
                        "observed": "red",
                        "evidence": (
                            "The bicycle frame appears red, while the wet street "
                            "has white reflective highlights."
                        ),
                    }
                ]
            },
            "errors": [
                {
                    "type": "wrong_attribute",
                    "evidence": "The reflective scene has white highlights.",
                    "prompt_span": "black bicycle",
                }
            ],
        },
        preferred_object="bicycle",
    )

    assert plan is not None
    assert plan["target_name"] == "bicycle"
    assert plan["target_color_name"] == "black"
    assert plan["source_color"] == "red"


def test_relation_repair_final_image_must_preserve_user_constraints(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (12, 12, 12)).save(source)
    layout_response = json.dumps(
        {
            "canvas_size": [64, 64],
            "background": {"description": "rainy street", "viewpoint": "back view"},
            "objects": [
                {
                    "name": "small red robot",
                    "description": "robot body and claw",
                    "bbox": [22, 30, 20, 24],
                    "order": 1,
                    "relations": ["gripping umbrella handle"],
                },
                {
                    "name": "blue umbrella",
                    "description": "blue umbrella and black handle",
                    "bbox": [12, 8, 44, 30],
                    "order": 2,
                    "relations": ["held by robot"],
                },
            ],
        }
    )
    llm = MockLLMClient(
        responses=[
            layout_response,
            json.dumps({"prompts": ["a small red robot gripping a blue umbrella"]}),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.75,
                                "attribute_binding": 0.8,
                                "object_relationship": 0.45,
                                "background_consistency": 0.8,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.7}]}),
            json.dumps(
                {
                    "score": 0.46,
                    "errors": [
                        {
                            "type": "wrong_relation",
                            "evidence": "The hand is detached and not gripping the umbrella handle.",
                            "prompt_span": "gripping the handle",
                        }
                    ],
                    "revision_hint": "Make visible physical contact with the handle.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "found": True,
                    "bbox": [22, 32, 20, 20],
                    "confidence": 0.9,
                    "reason": "The visible claw and handle contact should be repaired here.",
                }
            ),
            json.dumps(
                {
                    "score": 0.91,
                    "passed": True,
                    "checks": {
                        "handle_visible": True,
                        "hand_or_claw_visible": True,
                        "visible_grip": True,
                        "physical_contact": True,
                        "handle_connected_to_umbrella": True,
                        "not_merely_near_or_supported": True,
                        "user_colors_preserved": True,
                    },
                    "strengths": ["The claw visibly grips the umbrella handle."],
                    "revision_hint": "No change.",
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.58,
                    "checks": [
                        {
                            "type": "color",
                            "target": "robot",
                            "expected": "red",
                            "observed": "red",
                            "passed": True,
                        },
                        {
                            "type": "color",
                            "target": "umbrella",
                            "expected": "blue",
                            "observed": "red",
                            "passed": False,
                            "description": "The relation edit fixed contact but the umbrella is red.",
                        },
                        {
                            "type": "relation",
                            "target": "robot and umbrella handle",
                            "expected": "clearly gripping",
                            "observed": "visible contact",
                            "passed": True,
                        },
                    ],
                    "errors": [
                        {
                            "type": "wrong_attribute",
                            "evidence": "The umbrella is red instead of blue after relation repair.",
                            "prompt_span": "blue umbrella",
                        }
                    ],
                    "revision_hint": "Preserve the blue umbrella while repairing contact.",
                }
            ),
        ]
    )
    relation_repairer = RelationActionRepairer(
        vlm,
        MockInpaintEditor(prefix="relation_test"),
        candidates=1,
        pass_threshold=0.82,
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(existing_paths=[source]),
        config=AgentConfig(creativity_level="high", n_images=1, max_rounds=1),
        runs_dir=tmp_path,
        enable_clarifier=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(llm),
        layout_canvas_size=(64, 64),
        enable_relation_repair=True,
        relation_repairer=relation_repairer,
        score_threshold=0.85,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella",
        run_id="relation-repair-color-regression",
    )
    feedback = result.round_records[0]["feedback"]
    stop_event = next(event for event in result.events if event["type"] == "stop")
    repair_log = json.loads(
        (Path(result.run_dir) / "relation_repair_round_0.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.status == "max_rounds_reached"
    assert "relation_repaired" not in feedback
    assert result.round_records[0]["selected_image"] == str(source)
    assert repair_log["accepted"] is False
    assert result.round_records[0]["selected_image"] == str(source)
    assert repair_log["rejected_image"].endswith("relation_test_image_0000.txt")
    assert repair_log["acceptance"]["post_repair_constraint_failures"]
    assert feedback["score"] == pytest.approx(0.46)
    assert feedback["completion_gate"]["passed"] is False
    assert {item["type"] for item in feedback["completion_gate"]["blockers"]} >= {
        "relation_repair_rejected",
        "user_grounded_error",
    }
    assert stop_event["reason"] == "max_rounds_reached"
    assert any(
        event["type"] == "post_relation_repair_constraint_check"
        and event["passed"] is False
        and event["accepted"] is False
        for event in result.events
    )


def test_build_final_report_handles_empty_rounds() -> None:
    report = build_final_report(
        run_id="run-1",
        status="awaiting_clarification",
        mode="mock",
        user_prompt="a city",
        final_prompt="a city",
        final_score=None,
        selected_image=None,
        round_records=[],
    )

    assert "Status: `awaiting_clarification`" in report
    assert "No generation rounds were completed." in report


def test_orchestrator_locks_user_constraints_and_prompt_budget(tmp_path: Path) -> None:
    too_long_wrong_prompt = (
        "a small red robot holding a red umbrella on a rainy street, cinematic shallow depth of field, "
        "moody indigo twilight, volumetric rain mist, hyperrealistic detail, 35mm film grain, "
        "extra ornate signage, complex reflections, dramatic lens flare"
    )
    llm = MockLLMClient(responses=[json.dumps({"prompts": [too_long_wrong_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["Looks aligned."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(
            human_in_loop=False,
            creativity_level="high",
            n_images=1,
            max_rounds=1,
        ),
        runs_dir=tmp_path,
        enable_clarifier=False,
        prompt_candidates_per_round=1,
        clip_token_budget=35,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella",
        run_id="constraint-lock",
    )

    generated_prompt = generator.calls[0]["prompt"]
    assert "blue umbrella" in generated_prompt.lower()
    assert "red umbrella" not in generated_prompt.lower()
    assert approx_clip_token_count(generated_prompt) <= 35
    assert any(event["type"] == "prompt_constraints_applied" for event in result.events)
    run_log = json.loads((Path(result.run_dir) / "run.json").read_text(encoding="utf-8"))
    assert run_log["prompt_constraints"]["colors"]["umbrella"] == "blue"


def test_orchestrator_m42_constraint_check_drives_binding_retry(
    tmp_path: Path,
) -> None:
    llm = MockLLMClient(
        responses=[
            json.dumps(
                {
                    "prompts": [
                        "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo"
                    ]
                }
            ),
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": "blue umbrella and visible handle grip",
                            "prompt": (
                                "a small red robot clearly gripping a blue umbrella, "
                                "cinematic rainy street photo"
                            ),
                            "fixes": ["wrong_attribute", "wrong_relation"],
                            "expected_improvement": "Keeps the user colors and relation explicit.",
                        }
                    ]
                }
            ),
        ]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.8,
                                "attribute_binding": 0.7,
                                "object_relationship": 0.7,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.8}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["The scene looks cinematic."],
                    "revision_hint": "No ordinary critique.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.42,
                    "checks": [
                        {
                            "type": "wrong_attribute",
                            "target": "umbrella",
                            "expected": "blue umbrella",
                            "observed": "red umbrella",
                            "passed": False,
                            "description": "The umbrella appears red, but the user asked for blue.",
                        },
                        {
                            "type": "wrong_relation",
                            "target": "robot hand and umbrella handle",
                            "expected": "clearly gripping",
                            "observed": "handle hidden",
                            "passed": False,
                            "description": "The hand is not clearly gripping the handle.",
                        },
                    ],
                    "revision_hint": "Use a vivid blue umbrella and show the hand gripping the handle.",
                }
            ),
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.91,
                    "errors": [],
                    "strengths": ["The visible blue umbrella is improved."],
                    "revision_hint": "No change.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": True,
                    "score": 0.91,
                    "checks": [],
                    "errors": [],
                    "revision_hint": "Constraints pass.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=2, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=True,
    )

    result = orchestrator.run(
        "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo",
        run_id="m42-binding",
    )

    assert result.status == "completed"
    assert result.round_records[0]["feedback"]["score"] == pytest.approx(0.42)
    assert result.round_records[0]["feedback"]["constraint_check"]["passed"] is False
    revised_prompt = result.round_records[0]["revised_prompt"].lower()
    assert "blue umbrella" in revised_prompt
    assert "visibly gripping" in revised_prompt
    assert "umbrella handle" in revised_prompt
    assert "red umbrella" not in revised_prompt
    assert "red umbrella" in generator.calls[0]["negative_prompt"]
    assert any(event["type"] == "constraint_check" for event in result.events)
    assert any(event["type"] == "binding_retry_prompt" for event in result.events)


def test_orchestrator_continues_when_constraint_check_api_fails(
    tmp_path: Path,
) -> None:
    class FailingConstraintVLM(MockVLMClient):
        def vision(self, prompt: str, image_paths: list[str]) -> str:
            if "strict visual constraint checker" in prompt:
                raise RuntimeError("API connection error: SSL EOF")
            return super().vision(prompt, image_paths)

    llm = MockLLMClient(
        responses=[json.dumps({"prompts": ["a red figure holding a blue canopy"]})]
    )
    vlm = FailingConstraintVLM(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.8,
                                "background_consistency": 0.8,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.9,
                    "errors": [],
                    "strengths": ["The mock image is acceptable."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    orchestrator = OrchestratorAgent(
        llm=llm,
        vlm=vlm,
        image_generator=MockImageGenerator(),
        config=AgentConfig(n_images=1, max_rounds=1, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=True,
    )

    result = orchestrator.run("a red figure holding a blue canopy", run_id="constraint-api-fail")

    assert result.status == "max_rounds_reached"
    assert any(event["type"] == "constraint_check_failed" for event in result.events)
    assert result.round_records[0]["feedback"]["constraint_check"]["failed"] is True
    gate = result.round_records[0]["feedback"]["completion_gate"]
    assert gate["passed"] is False
    assert any(
        item["type"] == "constraint_check_unavailable"
        for item in gate["blockers"]
    )
    assert (Path(result.run_dir) / "run.json").exists()


def test_orchestrator_m51_applies_layout_guidance_to_generation_prompt(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    prompt_llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    layout_llm = MockLLMClient(
        responses=[build_mock_layout_response(user_prompt, canvas_size=(512, 512))]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.92,
                    "errors": [],
                    "strengths": ["Layout guided mock image is aligned."],
                    "revision_hint": "No change.",
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=prompt_llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=1, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(layout_llm),
        layout_canvas_size=(512, 512),
    )

    result = orchestrator.run(user_prompt, run_id="m51-layout")
    run_dir = Path(result.run_dir)
    generated_prompt = generator.calls[0]["prompt"]
    run_log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    layout_log = json.loads((run_dir / "layout.json").read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert "cinematic composition" in generated_prompt
    assert "blue umbrella" in generated_prompt
    assert "positioned" in generated_prompt
    assert (run_dir / "layout.json").exists()
    assert run_log["layout"]["prompt_package"]["generation_order"] == [
        "robot",
        "umbrella",
    ]
    assert layout_log["prompt_package"]["objects"][0]["bbox"] == [204, 225, 112, 194]
    assert any(event["type"] == "layout_planned" for event in result.events)
    assert any(event["type"] == "layout_prompt_applied" for event in result.events)
    assert len(prompt_llm.calls) == 1
    assert len(layout_llm.calls) == 1


def test_orchestrator_m51_does_not_reapply_layout_after_visual_revision(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    first_prompt = (
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    revised_prompt = (
        "red robot, blue umbrella, robot hand visibly gripping black umbrella handle, "
        "cinematic rainy street photo"
    )
    prompt_llm = MockLLMClient(
        responses=[
            json.dumps({"prompts": [first_prompt]}),
            json.dumps(
                {
                    "candidates": [
                        {
                            "modified_sentence": "cinematic rainy street photo",
                            "prompt": revised_prompt,
                            "fixes": ["style_mismatch"],
                            "expected_improvement": "Restore the user-requested scene style.",
                        }
                    ]
                }
            ),
        ]
    )
    layout_llm = MockLLMClient(
        responses=[build_mock_layout_response(user_prompt, canvas_size=(512, 512))]
    )
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.8,
                                "attribute_binding": 0.8,
                                "object_relationship": 0.8,
                                "background_consistency": 0.4,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.4}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "style_mismatch",
                            "evidence": "The rainy street is missing.",
                            "prompt_span": "",
                        }
                    ],
                    "strengths": ["Robot and umbrella are visible."],
                    "revision_hint": "Restore the rainy street setting.",
                    "user_grounded": False,
                }
            ),
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.9,
                                "attribute_binding": 0.9,
                                "object_relationship": 0.9,
                                "background_consistency": 0.9,
                                "aesthetic": 0.8,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.95}]}),
            json.dumps(
                {
                    "score": 0.93,
                    "errors": [],
                    "strengths": ["The revised prompt is aligned."],
                    "revision_hint": "No change.",
                    "user_grounded": True,
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=prompt_llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=2, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=False,
        enable_layout_planner=True,
        layout_planner=LayoutPlanner(layout_llm),
        layout_canvas_size=(512, 512),
    )

    result = orchestrator.run(user_prompt, run_id="m51-layout-no-repeat")

    assert result.status == "completed"
    assert len(generator.calls) == 2
    assert "cinematic composition" in generator.calls[0]["prompt"]
    assert "robot hand visibly gripping" in generator.calls[1]["prompt"]
    assert "cinematic rainy street photo" in generator.calls[1]["prompt"]
    assert "cinematic composition" not in generator.calls[1]["prompt"]
    assert "positioned" not in generator.calls[1]["prompt"]
    layout_events = [
        event for event in result.events if event["type"] == "layout_prompt_applied"
    ]
    assert [event["round"] for event in layout_events] == [0]


def test_orchestrator_m52_generates_binding_variants_for_selection(
    tmp_path: Path,
) -> None:
    user_prompt = (
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    prompt_llm = MockLLMClient(responses=[json.dumps({"prompts": [user_prompt]})])
    vlm = MockVLMClient(
        responses=[
            json.dumps(
                {
                    "scores": [
                        {
                            "index": 0,
                            "subscores": {
                                "alignment": 0.7,
                                "attribute_binding": 0.6,
                                "object_relationship": 0.7,
                                "background_consistency": 0.8,
                                "aesthetic": 0.7,
                            },
                        }
                    ]
                }
            ),
            json.dumps({"selected_index": 1, "scores": [{"index": 1, "score": 0.9}]}),
            json.dumps(
                {
                    "score": 0.91,
                    "errors": [],
                    "strengths": ["The blue umbrella variant is selected."],
                    "revision_hint": "No change.",
                    "user_grounded": True,
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=prompt_llm,
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=3, max_rounds=1, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=False,
        enable_binding_variants=True,
    )

    result = orchestrator.run(user_prompt, run_id="m52-binding-variants")
    run_log = json.loads((Path(result.run_dir) / "run.json").read_text(encoding="utf-8"))
    image_event = next(event for event in result.events if event["type"] == "images_generated")
    variant_event = next(
        event for event in result.events if event["type"] == "binding_prompt_variants"
    )

    assert result.status == "completed"
    assert len(generator.calls[0]["prompts"]) == 3
    assert len(image_event["prompts"]) == 3
    assert variant_event["strategies"] == ["base", "color_first", "object_separation"]
    assert result.round_records[0]["selected_image"].endswith("mock_image_0001.txt")
    assert result.round_records[0]["prompt"] == generator.calls[0]["prompts"][1]
    assert result.state["refined_prompt"] == generator.calls[0]["prompts"][1]
    assert "not the same color" in generator.calls[0]["prompts"][2].lower()
    assert run_log["events"][0]["type"] == "state_initialized"


def test_typed_action_backend_generates_candidates_and_accepts_passing_one(tmp_path: Path) -> None:
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.5}]}),
            json.dumps(
                {
                    "score": 0.4,
                    "errors": [
                        {
                            "type": "wrong_count",
                            "evidence": "The holder has more pencils than pens.",
                            "prompt_span": "more pens than pencils",
                        }
                    ],
                    "revision_hint": "Make more pens than pencils.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.4,
                    "checks": [],
                    "errors": [
                        {
                            "type": "wrong_count",
                            "evidence": "The holder has more pencils than pens.",
                            "prompt_span": "more pens than pencils",
                        }
                    ],
                }
            ),
            json.dumps({"passed": False, "score": 0.3, "checks": [], "errors": [{"type": "wrong_count"}]}),
            json.dumps({"passed": True, "score": 0.95, "checks": [], "errors": []}),
            json.dumps({"passed": False, "score": 0.6, "checks": [], "errors": [{"type": "wrong_count"}]}),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=2, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=True,
        enable_specialist_reports=False,
        enable_typed_action_backend=True,
        typed_action_candidates=3,
        typed_action_max_candidates=4,
    )

    result = orchestrator.run(
        "A pencil holder with more pens than pencils.",
        run_id="typed-action-comparison",
    )
    run_dir = Path(result.run_dir)
    typed_payload = json.loads(
        (run_dir / "typed_action_round_0.json").read_text(encoding="utf-8")
    )

    assert typed_payload["route"] == "comparative_count_rerank"
    assert typed_payload["accepted"] is True
    selected_index = typed_payload["selected_index"]
    assert typed_payload["candidate_checks"][selected_index]["passed"] is True
    assert len(typed_payload["candidate_checks"]) == 3
    assert len(generator.calls) == 2
    assert len(generator.calls[1]["prompts"]) == 3
    assert result.status == "completed"


def test_orchestrator_stops_for_unverifiable_rare_word_preflight(tmp_path: Path) -> None:
    vlm = MockVLMClient(
        responses=[
            json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 0.4}]}),
            json.dumps(
                {
                    "score": 0.2,
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The image is unrelated to Acersecomicke.",
                            "prompt_span": "Acersecomicke",
                        }
                    ],
                    "revision_hint": "The requested rare word is not visually defined.",
                    "user_grounded": True,
                }
            ),
            json.dumps(
                {
                    "passed": False,
                    "score": 0.2,
                    "checks": [],
                    "errors": [
                        {
                            "type": "missing_object",
                            "evidence": "The requested rare word is not visually defined.",
                            "prompt_span": "Acersecomicke",
                        }
                    ],
                }
            ),
        ]
    )
    generator = MockImageGenerator()
    orchestrator = OrchestratorAgent(
        llm=MockLLMClient(),
        vlm=vlm,
        image_generator=generator,
        config=AgentConfig(n_images=1, max_rounds=2, creativity_level="high"),
        runs_dir=tmp_path,
        score_threshold=0.85,
        prompt_candidates_per_round=1,
        enable_clarifier=False,
        enable_constraint_check=True,
        enable_specialist_reports=False,
        enable_typed_action_backend=True,
    )

    result = orchestrator.run("Acersecomicke.", run_id="rare-word-clarify")
    run_dir = Path(result.run_dir)
    repair_plan = json.loads(
        (run_dir / "repair_plan_round_0.json").read_text(encoding="utf-8")
    )

    assert result.status == "needs_clarification"
    assert len(generator.calls) == 1
    assert repair_plan["typed_route"] == "unverifiable_rare_word_or_clarify"
    assert repair_plan["primary_action"] == "none"
    stop_event = next(event for event in result.events if event["type"] == "stop")
    assert stop_event["reason"] == "unverifiable_or_clarify"
