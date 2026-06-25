from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binding_variants import build_binding_variants, should_use_binding_variants
from src.prompt_constraints import approx_clip_token_count, extract_constraints


def test_binding_variants_build_short_color_separation_prompts() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    prompt = (
        "cinematic rainy street photo, a small red robot clearly gripping the handle "
        "of a blue umbrella, small red robot positioned center: compact anthropomorphic "
        "robot with glossy crimson metallic body, blue umbrella positioned center: "
        "vibrant cobalt blue umbrella with curved metal handle"
    )

    variants = build_binding_variants(
        prompt,
        constraints,
        max_variants=3,
        token_budget=70,
    )

    assert should_use_binding_variants(constraints)
    assert [variant["strategy"] for variant in variants] == [
        "base",
        "color_first",
        "object_separation",
    ]
    assert all("blue umbrella" in variant["prompt"].lower() for variant in variants)
    assert any("not the same color" in variant["prompt"].lower() for variant in variants)
    assert all(approx_clip_token_count(variant["prompt"]) <= 70 for variant in variants)


def test_binding_variants_support_generic_carry_interactions() -> None:
    constraints = extract_constraints(
        "A green wizard carries a silver lantern while standing beside an orange barrel."
    )

    variants = build_binding_variants(
        "A green wizard standing beside an orange barrel with a silver lantern.",
        constraints,
        max_variants=3,
        token_budget=70,
    )

    assert should_use_binding_variants(constraints)
    prompts = [variant["prompt"].lower() for variant in variants]
    assert any("wizard visibly carrying the lantern" in prompt for prompt in prompts)
    assert all("silver lantern" in prompt for prompt in prompts)


def test_binding_variants_support_spatial_relation_prompts() -> None:
    constraints = extract_constraints(
        (
            "A red cube is left of a blue sphere, and the blue sphere is under "
            "a green cone; all three objects are visible."
        )
    )

    variants = build_binding_variants(
        constraints.original_prompt,
        constraints,
        max_variants=3,
        token_budget=80,
    )

    assert should_use_binding_variants(constraints)
    assert [variant["strategy"] for variant in variants] == [
        "base",
        "spatial_literal",
        "spatial_axis_order",
    ]
    prompts = [variant["prompt"].lower() for variant in variants]
    assert any("cube left of sphere" in prompt for prompt in prompts)
    assert any("sphere under cone" in prompt for prompt in prompts)
    assert any("vertical order" in prompt for prompt in prompts)
