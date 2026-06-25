from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import MockVLMClient
from src.run_specialist_observation import build_parser, main
from src.specialist_agents import (
    analyze_specialist_observation,
    parse_specialist_observation_response,
    run_specialist_observation,
)


PROMPT = (
    "A cyan cat holds a red umbrella handle while sitting beside a purple "
    "teapot, no color leakage between the objects."
)


def test_specialist_observation_uses_one_vlm_call_and_blocks_relation_drift() -> None:
    vlm = MockVLMClient(responses=[json.dumps(_confused_handle_response())])

    result = run_specialist_observation(
        vlm=vlm,
        user_prompt=PROMPT,
        image_path="mock://image/0",
        generated_prompt="the subject visibly holds the teapot",
    )

    assert len(vlm.calls) == 1
    assert result["api_call_count"] == 1
    assert [report["agent"] for report in result["reports"]] == [
        "SubjectExistenceAgent",
        "AttributeBindingAgent",
        "SpatialLayoutAgent",
        "InteractionRelationAgent",
        "SymbolTextVisibilityAgent",
        "StyleBackgroundAgent",
    ]

    arbitration = result["arbitration"]
    assert arbitration["global_passed"] is False
    assert arbitration["dominant_failure"] == "interaction_relation"
    assert arbitration["selected_action"] == "relation_repair_or_object_insertion"
    assert "teapot handle" in arbitration["forbidden_phrases"]
    assert "separate red umbrella handle" in arbitration["prompt_patch"]
    assert "physically separate from the purple teapot" in arbitration["prompt_patch"]


def test_specialist_parser_accepts_fenced_json() -> None:
    response = "```json\n" + json.dumps(_confused_handle_response()) + "\n```"

    parsed = parse_specialist_observation_response(response)

    assert parsed["summary"]["dominant_failure"] == "interaction"
    assert parsed["interaction_relations"][0]["confused_with"] == "teapot handle"


def test_generated_prompt_drift_overrides_single_vlm_relation_pass() -> None:
    result = analyze_specialist_observation(
        user_prompt=PROMPT,
        image_path="mock://image/0",
        generated_prompt="the subject visibly holds the teapot, red umbrella handle",
        observation=_relation_pass_attribute_fail_response(),
    )

    arbitration = result["arbitration"]
    assert result["api_call_count"] == 0
    assert arbitration["dominant_failure"] == "interaction_relation"
    assert arbitration["selected_action"] == "relation_repair_or_object_insertion"
    assert "holds the teapot" in arbitration["forbidden_phrases"]
    assert "separate red umbrella handle" in arbitration["prompt_patch"]
    assert "holds the teapot" not in arbitration["prompt_patch"]


def test_specialist_relation_patch_is_generic_not_object_specific() -> None:
    user_prompt = (
        "A green wizard carries a silver lantern while standing beside an orange barrel."
    )
    result = analyze_specialist_observation(
        user_prompt=user_prompt,
        image_path="mock://image/0",
        generated_prompt="the subject visibly carries the barrel, green wizard, silver lantern",
        observation={
            "subjects": [
                {"name": "green wizard", "visible": True, "confidence": 0.9},
                {"name": "silver lantern", "visible": True, "confidence": 0.9},
                {"name": "orange barrel", "visible": True, "confidence": 0.9},
            ],
            "attributes": [
                {
                    "object": "wizard",
                    "attribute": "color",
                    "expected": "green",
                    "observed": "green",
                    "passed": True,
                    "confidence": 0.9,
                },
                {
                    "object": "lantern",
                    "attribute": "color",
                    "expected": "silver",
                    "observed": "silver",
                    "passed": True,
                    "confidence": 0.9,
                },
                {
                    "object": "barrel",
                    "attribute": "color",
                    "expected": "orange",
                    "observed": "orange",
                    "passed": True,
                    "confidence": 0.9,
                },
            ],
            "interaction_relations": [
                {
                    "subject": "wizard",
                    "action": "carries",
                    "object": "lantern",
                    "passed": True,
                    "confidence": 0.9,
                    "evidence": "Wizard and lantern are visible.",
                }
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "interaction_relation"
    assert "carries the barrel" in arbitration["forbidden_phrases"]
    assert "separate silver lantern" in arbitration["prompt_patch"]
    assert "physically separate from the orange barrel" in arbitration["prompt_patch"]
    assert "barrel" not in arbitration["prompt_patch"].split("separate silver lantern", 1)[0]


def test_specialist_spatial_report_accepts_phrase_field() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A blue notebook shows a yellow star symbol on its cover, next to a "
            "plain green notebook with no symbol."
        ),
        image_path="mock://image/notebooks",
        observation={
            "subjects": [
                {"name": "blue notebook", "visible": True},
                {"name": "green notebook", "visible": True},
                {"name": "star symbol", "visible": True},
            ],
            "attributes": [
                {"object": "blue notebook", "attribute": "color", "passed": True},
                {"object": "green notebook", "attribute": "color", "passed": True},
                {"object": "star symbol", "attribute": "color", "passed": True},
            ],
            "spatial_relations": [
                {
                    "subject": "blue notebook",
                    "phrase": "next to",
                    "object": "green notebook",
                    "passed": True,
                }
            ],
            "symbol_text_relations": [
                {
                    "subject": "blue notebook",
                    "action": "shows",
                    "object": "star symbol",
                    "passed": True,
                }
            ],
        },
    )

    assert result["arbitration"]["global_passed"] is True


def test_specialist_spatial_report_fails_one_bad_required_relation() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A gray dog stands behind a teal bench, while a pink ball rests under "
            "the bench."
        ),
        image_path="mock://image/spatial",
        observation={
            "subjects": [
                {"name": "dog", "visible": True, "count": 1},
                {"name": "bench", "visible": True, "count": 1},
                {"name": "ball", "visible": True, "count": 1},
            ],
            "attributes": [
                {"object": "dog", "attribute": "color", "passed": True},
                {"object": "bench", "attribute": "color", "passed": True},
                {"object": "ball", "attribute": "color", "passed": True},
            ],
            "spatial_relations": [
                {
                    "subject": "dog",
                    "phrase": "behind",
                    "object": "bench",
                    "passed": True,
                },
                {
                    "subject": "ball",
                    "phrase": "under",
                    "object": "bench",
                    "passed": False,
                    "confidence": 0.95,
                    "evidence": "The ball is on top of the bench, not under it.",
                },
            ],
        },
    )

    reports = {item["agent"]: item for item in result["reports"]}
    assert reports["SpatialLayoutAgent"]["passed"] is False
    assert result["arbitration"]["dominant_failure"] == "spatial_layout"
    assert result["arbitration"]["selected_action"] == "layout_guided_regeneration"
    assert "strict 2D layout" in result["arbitration"]["prompt_patch"]
    assert "pink ball below teal bench" in result["arbitration"]["prompt_patch"]


def test_specialist_count_failure_routes_to_count_repair() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "Two red cups and one blue bowl are on a wooden table; the cups are "
            "separate and both cups are fully visible."
        ),
        image_path="mock://image/0",
        observation={
            "subjects": [
                {"name": "red cups", "visible": True, "count": 2, "confidence": 0.95},
                {
                    "name": "blue bowl",
                    "visible": True,
                    "count": 0,
                    "confidence": 0.9,
                    "evidence": "No blue bowl is visible, only two cups.",
                },
            ],
            "attributes": [
                {"object": "cups", "attribute": "color", "observed": "red", "passed": True},
                {"object": "bowl", "attribute": "color", "observed": "blue", "passed": True},
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "count_quantity"
    assert arbitration["selected_action"] == "count_repair_or_regenerate"
    assert "exactly 1 bowl" in arbitration["prompt_patch"]


def test_specialist_missing_object_routes_to_subject_existence() -> None:
    result = analyze_specialist_observation(
        user_prompt="A magenta glass apple and a teal ceramic pear sit on a gray plate.",
        image_path="mock://image/0",
        observation={
            "subjects": [
                {"name": "ceramic pear", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "gray plate", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {"object": "ceramic pear", "attribute": "color", "observed": "teal", "passed": True},
                {"object": "plate", "attribute": "color", "observed": "gray", "passed": True},
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "subject_existence"
    assert arbitration["selected_action"] == "object_insertion_or_regenerate"
    assert "missing magenta glass apple" in arbitration["prompt_patch"]


def test_specialist_material_failure_is_not_color_patch() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A turquoise wooden chair, a crimson glass lamp, and a silver paper "
            "fan sit on a black rug."
        ),
        image_path="mock://image/material",
        observation={
            "subjects": [
                {"name": "chair", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "glass lamp", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "paper fan", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "rug", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {"object": "chair", "attribute": "color", "passed": True},
                {"object": "glass lamp", "attribute": "color", "passed": True},
                {"object": "paper fan", "attribute": "color", "passed": True},
                {"object": "rug", "attribute": "color", "passed": True},
                {
                    "object": "paper fan",
                    "attribute": "material",
                    "expected": "paper",
                    "observed": "metal electric fan",
                    "passed": False,
                    "confidence": 0.9,
                    "evidence": "The fan looks metallic/electric, not a paper fan.",
                },
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "material_binding"
    assert arbitration["selected_action"] == "material_repair_or_regenerate"
    assert "made of paper" in arbitration["prompt_patch"]
    assert "must visibly remain silver" not in arbitration["prompt_patch"]


def test_specialist_missing_object_beats_material_or_color_patch() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "An indigo ceramic turtle, a gold fabric pouch, and a pink metal "
            "whistle rest on a white tray."
        ),
        image_path="mock://image/missing-pouch",
        observation={
            "subjects": [
                {"name": "ceramic turtle", "visible": True, "count": 1, "confidence": 0.9},
                {
                    "name": "fabric pouch",
                    "visible": False,
                    "count": 0,
                    "confidence": 0.95,
                    "evidence": "No fabric pouch is visible.",
                },
                {"name": "metal whistle", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "tray", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {"object": "ceramic turtle", "attribute": "color", "passed": True},
                {"object": "fabric pouch", "attribute": "color", "passed": False},
                {"object": "metal whistle", "attribute": "color", "passed": True},
                {"object": "tray", "attribute": "color", "passed": True},
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "subject_existence"
    assert arbitration["selected_action"] == "object_insertion_or_regenerate"
    assert "missing gold fabric pouch" in arbitration["prompt_patch"]
    assert "do not replace it with another object" in arbitration["prompt_patch"]


def test_specialist_missing_colored_material_object_beats_wrong_attribute_patch() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A turquoise wooden chair, a crimson glass lamp, and a silver paper "
            "fan sit on a black rug."
        ),
        image_path="mock://image/missing-fan",
        observation={
            "subjects": [
                {"name": "chair", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "glass lamp", "visible": True, "count": 1, "confidence": 0.9},
                {
                    "name": "paper fan",
                    "visible": False,
                    "count": 0,
                    "confidence": 0.95,
                    "evidence": "The silver paper fan is missing from the image.",
                },
                {"name": "rug", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {
                    "object": "chair",
                    "attribute": "material",
                    "expected": "wooden",
                    "observed": "leather",
                    "passed": False,
                    "confidence": 0.9,
                    "evidence": "The chair looks like leather rather than wood.",
                },
                {"object": "glass lamp", "attribute": "color", "passed": True},
                {"object": "rug", "attribute": "color", "passed": True},
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "subject_existence"
    assert arbitration["selected_action"] == "object_insertion_or_regenerate"
    assert "missing silver paper fan" in arbitration["prompt_patch"]
    assert "must visibly remain turquoise" not in arbitration["prompt_patch"]


def test_specialist_spatial_patch_uses_explicit_2d_right_of_layout() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A yellow pyramid is clearly right of a red cylinder; only these "
            "two objects are visible."
        ),
        image_path="mock://image/spatial",
        observation={
            "subjects": [
                {"name": "pyramid", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "cylinder", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {"object": "pyramid", "attribute": "color", "observed": "yellow", "passed": True},
                {"object": "cylinder", "attribute": "color", "observed": "red", "passed": True},
            ],
            "spatial_relations": [
                {
                    "subject": "pyramid",
                    "relation": "right of",
                    "object": "cylinder",
                    "passed": False,
                    "confidence": 0.9,
                    "evidence": "The pyramid is left of the cylinder.",
                }
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "spatial_layout"
    assert arbitration["selected_action"] == "layout_guided_regeneration"
    assert "strict 2D layout" in arbitration["prompt_patch"]
    assert "right side of red cylinder" in arbitration["prompt_patch"]


def test_specialist_negative_attached_failure_routes_as_relation() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A green wizard carries a silver lantern while standing beside an "
            "orange barrel; the lantern is not attached to the barrel."
        ),
        image_path="mock://image/0",
        observation={
            "subjects": [
                {"name": "wizard", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "lantern", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "barrel", "visible": True, "count": 1, "confidence": 0.9},
            ],
            "attributes": [
                {"object": "wizard", "attribute": "color", "observed": "green", "passed": True},
                {"object": "lantern", "attribute": "color", "observed": "silver", "passed": True},
                {"object": "barrel", "attribute": "color", "observed": "orange", "passed": True},
            ],
            "interaction_relations": [
                {
                    "subject": "wizard",
                    "action": "carries",
                    "object": "lantern",
                    "passed": True,
                    "confidence": 0.9,
                }
            ],
            "negative_constraints": [
                {
                    "constraint": "the lantern is not attached to the barrel",
                    "passed": False,
                    "confidence": 0.9,
                    "evidence": "The lantern appears attached to the barrel.",
                }
            ],
        },
    )

    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "interaction_relation"
    assert arbitration["selected_action"] == "relation_repair_or_object_insertion"


def test_symbol_text_display_is_not_physical_interaction_or_forbidden_object() -> None:
    result = analyze_specialist_observation(
        user_prompt=(
            "A blue notebook shows a yellow star symbol on its cover, next to a "
            "plain green notebook with no symbol."
        ),
        image_path="mock://image/0",
        observation={
            "subjects": [
                {"name": "blue notebook", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "green notebook", "visible": True, "count": 1, "confidence": 0.9},
                {"name": "star symbol", "visible": True, "count": 1, "confidence": 0.8},
            ],
            "attributes": [
                {"object": "blue notebook", "attribute": "color", "observed": "blue", "passed": True},
                {"object": "green notebook", "attribute": "color", "observed": "green", "passed": True},
                {"object": "star symbol", "attribute": "color", "observed": "yellow", "passed": True},
            ],
            "symbol_text_relations": [
                {
                    "subject": "blue notebook",
                    "action": "shows",
                    "object": "star symbol",
                    "passed": False,
                    "confidence": 0.9,
                    "evidence": "The star symbol appears on the green notebook instead of the blue notebook.",
                    "confused_with": "green notebook",
                }
            ],
        },
    )

    report_by_agent = {item["agent"]: item for item in result["reports"]}
    assert report_by_agent["InteractionRelationAgent"]["passed"] is True
    assert report_by_agent["SymbolTextVisibilityAgent"]["passed"] is False
    arbitration = result["arbitration"]
    assert arbitration["dominant_failure"] == "symbol_text_visibility"
    assert arbitration["selected_action"] == "symbol_text_repair_or_regenerate"
    assert "green notebook" not in arbitration["forbidden_phrases"]
    assert "showsing" not in arbitration["prompt_patch"]
    assert "blue notebook clearly shows" in arbitration["prompt_patch"]


def test_run_specialist_observation_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "specialist_report.json"

    exit_code = main(
        [
            "--prompt",
            PROMPT,
            "--image-path",
            "mock://image/0",
            "--generated-prompt",
            "the subject visibly holds the teapot",
            "--vlm",
            "mock",
            "--mock-response",
            json.dumps(_confused_handle_response()),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["api_call_count"] == 1
    assert data["arbitration"]["dominant_failure"] == "interaction_relation"


def test_run_specialist_observation_cli_reuses_existing_report(tmp_path: Path) -> None:
    input_report = tmp_path / "input_report.json"
    input_report.write_text(
        json.dumps({"observation": _relation_pass_attribute_fail_response()}),
        encoding="utf-8",
    )
    output = tmp_path / "reanalyzed_report.json"

    exit_code = main(
        [
            "--prompt",
            PROMPT,
            "--image-path",
            "mock://image/0",
            "--generated-prompt",
            "the subject visibly holds the teapot",
            "--input-report",
            str(input_report),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["api_call_count"] == 0
    assert data["arbitration"]["dominant_failure"] == "interaction_relation"


def test_run_specialist_observation_cli_accepts_api_flags() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            PROMPT,
            "--image-path",
            "/tmp/image.jpg",
            "--vlm",
            "api",
            "--api-key-env",
            "DASHSCOPE_API_KEY",
            "--api-base-url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "--vlm-model",
            "qwen-vl-plus",
        ]
    )

    assert args.vlm == "api"
    assert args.vlm_model == "qwen-vl-plus"


def _confused_handle_response() -> dict[str, object]:
    return {
        "subjects": [
            {
                "name": "cyan cat",
                "visible": True,
                "count": 1,
                "confidence": 0.95,
                "evidence": "A cyan cat is visible.",
            },
            {
                "name": "purple teapot",
                "visible": True,
                "count": 1,
                "confidence": 0.92,
                "evidence": "A purple teapot is beside the cat.",
            },
        ],
        "attributes": [
            {
                "object": "cyan cat",
                "attribute": "color",
                "expected": "cyan",
                "observed": "cyan",
                "passed": True,
                "confidence": 0.94,
                "evidence": "The cat is cyan.",
            },
            {
                "object": "purple teapot",
                "attribute": "color",
                "expected": "purple",
                "observed": "purple",
                "passed": True,
                "confidence": 0.9,
                "evidence": "The teapot body is purple.",
            },
            {
                "object": "red umbrella handle",
                "attribute": "color",
                "expected": "red",
                "observed": "red loop on the teapot",
                "passed": False,
                "confidence": 0.84,
                "evidence": "The red handle appears attached to the teapot, not a separate umbrella handle.",
            },
        ],
        "spatial_relations": [],
        "interaction_relations": [
            {
                "subject": "cat",
                "action": "holds",
                "object": "teapot handle",
                "passed": False,
                "confidence": 0.86,
                "evidence": "The cat grips a teapot handle rather than a separate umbrella handle.",
                "confused_with": "teapot handle",
            }
        ],
        "negative_constraints": [
            {
                "constraint": "no color leakage between the objects",
                "passed": False,
                "confidence": 0.78,
                "evidence": "The red handle visually merges with the purple teapot.",
            }
        ],
        "summary": {
            "global_passed": False,
            "dominant_failure": "interaction",
            "repair_hint": "Make the cat hold a separate red umbrella handle.",
        },
    }


def _relation_pass_attribute_fail_response() -> dict[str, object]:
    return {
        "subjects": [
            {"name": "cat", "visible": True, "count": 1, "confidence": 0.95},
            {"name": "umbrella handle", "visible": True, "count": 1, "confidence": 0.9},
            {"name": "teapot", "visible": True, "count": 1, "confidence": 0.95},
        ],
        "attributes": [
            {
                "object": "cat",
                "attribute": "color",
                "expected": "cyan",
                "observed": "blue",
                "passed": False,
                "confidence": 0.9,
                "evidence": "The cat appears blue rather than cyan.",
            },
            {
                "object": "teapot",
                "attribute": "color",
                "expected": "purple",
                "observed": "purple",
                "passed": True,
                "confidence": 0.95,
            },
            {
                "object": "umbrella handle",
                "attribute": "color",
                "expected": "red",
                "observed": "red",
                "passed": True,
                "confidence": 0.95,
            },
        ],
        "spatial_relations": [],
        "interaction_relations": [
            {
                "subject": "cat",
                "action": "holds",
                "object": "umbrella handle",
                "passed": True,
                "confidence": 0.96,
                "evidence": "The cat holds the red handle.",
            }
        ],
        "negative_constraints": [],
        "summary": {
            "global_passed": False,
            "dominant_failure": "attribute",
            "repair_hint": "Change the cat's color from blue to cyan.",
        },
    }
