from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_constraints import (
    approx_clip_token_count,
    constraint_violations,
    extract_constraints,
    extract_intent_spec,
    lock_prompt_to_user_constraints,
)


def test_extract_constraints_captures_color_action_and_subject() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )

    assert constraints.colors["robot"] == "red"
    assert constraints.colors["umbrella"] == "blue"
    assert "handle" not in constraints.subjects
    assert "clearly gripping" in constraints.actions
    assert constraints.intent_spec is not None
    assert constraints.intent_spec.interaction_relations[0]["object"] == "umbrella handle"
    assert "red robot" in constraints.protected_phrases
    assert "blue umbrella" in constraints.protected_phrases


def test_extract_constraints_does_not_turn_relations_or_adjectives_into_subjects() -> None:
    constraints = extract_constraints(
        "a blue cup on the left of three red apples on a wooden table, clean studio photo"
    )

    assert constraints.colors == {"cup": "blue", "apples": "red"}
    assert "cup" in constraints.subjects
    assert "apples" in constraints.subjects
    assert "table" in constraints.subjects
    assert "left" not in constraints.subjects
    assert "wooden" not in constraints.subjects
    assert "three" not in constraints.subjects


def test_extract_constraints_strips_trailing_relation_prepositions() -> None:
    constraints = extract_constraints(
        "a white cup to the left of three red apples on a wooden table, realistic photo"
    )

    assert constraints.colors == {"cup": "white", "apples": "red"}
    assert "cup" in constraints.subjects
    assert "cup to" not in constraints.subjects
    assert "apples" in constraints.subjects
    assert "table" in constraints.subjects


def test_extract_constraints_strips_common_relation_connectors_generically() -> None:
    prompts = {
        "a red chair next to a blue table, realistic photo": {"chair", "table"},
        "a yellow bird in front of a black bicycle near a white dog": {
            "bird",
            "bicycle",
            "dog",
        },
        "a green bottle between two red cups on a wooden table": {
            "bottle",
            "cups",
            "table",
        },
        "a black cat under a white chair with a red cushion": {
            "cat",
            "chair",
            "cushion",
        },
    }

    for prompt, expected_subjects in prompts.items():
        constraints = extract_constraints(prompt)
        subjects = set(constraints.subjects)
        assert expected_subjects <= subjects
        assert all(
            not any(subject.endswith(f" {word}") for word in ("to", "next", "in", "between", "under", "with"))
            for subject in subjects
        )


def test_extract_constraints_protects_short_user_scene_style_phrase() -> None:
    constraints = extract_constraints(
        "a small red robot gripping a blue umbrella, cinematic rainy street photo"
    )

    assert "cinematic rainy street photo" in constraints.protected_phrases


def test_lock_prompt_restores_user_color_and_truncates_low_priority_tail() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )
    prompt = (
        "a small red robot holds a red umbrella on a rainy street, cinematic shallow depth of field, "
        "moody indigo twilight, volumetric rain mist, hyperrealistic detail, 35mm film grain, "
        "extra ornate background signage and complex reflections"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=35)

    assert "blue umbrella" in result["prompt"].lower()
    assert "red umbrella" not in result["prompt"].lower()
    assert result["token_count"] <= 35
    assert any("dropped" in item for item in result["applied"])


def test_lock_prompt_does_not_let_generic_color_binding_overwrite_specific_object() -> None:
    constraints = extract_constraints(
        "A black sign displays the exact yellow text 'NO' above a plain blue sign "
        "with no text, and one pink ball sits to the right of the blue sign."
    )
    prompt = (
        "A blue sign displays the exact yellow text 'NO' above a plain blue sign "
        "with no text, and one pink ball sits to the right of the blue sign., "
        "black sign, yellow text, black blue sign, blue blue sign, pink ball"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=200)
    locked = result["prompt"].lower()

    assert "black sign" in locked
    assert "blue sign" in locked
    assert "black blue sign" not in locked
    assert "blue blue sign" not in locked
    assert not any(
        item["type"] == "conflicting_user_color" and item["prompt_span"] == "sign"
        for item in result["violations"]
    )


def test_lock_prompt_keeps_user_scene_style_under_budget() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    prompt = (
        "cinematic composition, wet asphalt street with reflective puddles, "
        "blurred storefronts and streetlights in the distance, overcast grey sky, "
        "red robot positioned lower: small anthropomorphic robot in glossy red plastic, "
        "blue umbrella positioned center: fully open dome-shaped umbrella, "
        "a small red robot clearly gripping the handle of a blue umbrella"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=55)

    assert "cinematic rainy street photo" in result["prompt"].lower()
    assert "red robot" in result["prompt"].lower()
    assert "blue umbrella" in result["prompt"].lower()
    assert result["token_count"] <= 55


def test_lock_prompt_prioritizes_user_scene_style_before_layout_details() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo"
    )
    prompt = (
        "robot positioned lower: small red robot with visible hand, below umbrella, "
        "hand reaches umbrella handle, umbrella positioned upper: blue umbrella canopy "
        "and visible handle, above the robot, handle aligned with robot hand, "
        "a small red robot clearly gripping the handle of a blue umbrella, "
        "cinematic rainy street photo, cinematic composition, cinematic rainy street "
        "with wet pavement and distant lights, front view, medium shot"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=70)
    segments = [segment.strip() for segment in result["prompt"].split(",")]

    assert "cinematic rainy street photo" in result["prompt"].lower()
    assert segments[0].lower() == "cinematic rainy street photo"
    assert result["token_count"] <= 70


def test_lock_prompt_protects_singular_instance_layout_for_plural_subject() -> None:
    constraints = extract_constraints(
        "two yellow birds sitting on a black bicycle near a white dog"
    )
    prompt = (
        "yellow bird positioned left: small yellow songbird on the left handlebar, "
        "yellow bird positioned center: small yellow songbird on the rear bicycle seat, "
        "black bicycle positioned lower: black bicycle with visible seat, "
        "white dog positioned lower-left: white dog near the front wheel, "
        "sunlit suburban backyard, shallow depth of field, photorealistic detail"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=70)

    assert "yellow bird positioned left" in result["prompt"].lower()
    assert "yellow bird positioned center" in result["prompt"].lower()
    assert result["token_count"] <= 70


def test_constraint_violations_detects_missing_user_action() -> None:
    constraints = extract_constraints(
        "a small red robot clearly gripping the handle of a blue umbrella"
    )

    violations = constraint_violations(
        "a small red robot beside a blue umbrella in rain",
        constraints,
    )

    assert any(item["type"] == "missing_user_action_or_relation" for item in violations)
    assert approx_clip_token_count("a red robot") >= 3


def test_lock_prompt_removes_meta_constraint_segments() -> None:
    constraints = extract_constraints(
        "a woman wearing a yellow hat holding a black cat, realistic portrait photo"
    )
    prompt = (
        "a woman wearing a yellow hat holding a black cat, "
        "no color specification for clothing beyond the yellow hat, "
        "realistic portrait photo"
    )

    result = lock_prompt_to_user_constraints(prompt, constraints, token_budget=50)

    assert "no color specification" not in result["prompt"].lower()
    assert "beyond the yellow hat" not in result["prompt"].lower()
    assert "yellow hat" in result["prompt"].lower()
    assert "black cat" in result["prompt"].lower()
    assert any("removed meta constraint segment" in item for item in result["applied"])


def test_intent_spec_splits_chained_interactions_generically() -> None:
    intent = extract_intent_spec(
        "a woman wearing a yellow hat holding a black cat"
    )

    interactions = {
        (item["subject"], item["action"], item["object"])
        for item in intent.interaction_relations
    }
    assert ("woman", "wearing", "hat") in interactions
    assert ("woman", "holding", "cat") in interactions


def test_intent_spec_avoids_pseudo_subjects_and_keeps_interactions() -> None:
    intent = extract_intent_spec(
        "A studio still life with a cyan cat. The cat is holding the red umbrella "
        "handle. The purple teapot stays left of the cat, no color leakage."
    )

    assert "cat" in intent.subjects
    assert "teapot" in intent.subjects
    assert "cat is" not in intent.subjects
    assert "teapot stays" not in intent.subjects
    assert intent.colors["cat"] == "cyan"
    assert intent.colors["umbrella handle"] == "red"
    assert intent.colors["teapot"] == "purple"
    assert any(item["action"] == "holding" for item in intent.interaction_relations)
    assert any(item["phrase"] == "left of" for item in intent.relations)
    assert intent.negative_constraints == ["no color leakage"]


def test_extract_constraints_exposes_intent_counts_and_relations() -> None:
    constraints = extract_constraints(
        "three magenta glass apples above a teal bowl, without extra fruit"
    )

    assert constraints.intent_spec is not None
    assert constraints.intent_spec.counts["glass apples"] == 3
    assert constraints.intent_spec.colors["glass apples"] == "magenta"
    assert constraints.intent_spec.colors["bowl"] == "teal"
    assert constraints.intent_spec.negative_constraints == ["without extra fruit"]
    assert "above" in constraints.relations


def test_intent_spec_extracts_material_attributes_for_binding() -> None:
    intent = extract_intent_spec(
        "A turquoise wooden chair, a crimson glass lamp, and a silver paper fan "
        "sit on a black rug."
    )

    assert intent.colors["chair"] == "turquoise"
    assert intent.colors["glass lamp"] == "crimson"
    assert intent.colors["paper fan"] == "silver"
    assert intent.attributes["chair"] == ["wooden"]
    assert intent.attributes["glass lamp"] == ["glass"]
    assert intent.attributes["paper fan"] == ["paper"]


def test_intent_spec_extracts_rare_color_material_bindings() -> None:
    intent = extract_intent_spec(
        "A lavender metal watering can, an amber glass bowl, and a navy cloth "
        "ribbon sit on a gray shelf."
    )

    assert intent.colors["metal watering can"] == "lavender"
    assert intent.colors["glass bowl"] == "amber"
    assert intent.colors["cloth ribbon"] == "navy"
    assert intent.attributes["metal watering can"] == ["metal"]
    assert intent.attributes["glass bowl"] == ["glass"]
    assert intent.attributes["cloth ribbon"] == ["cloth"]


def test_intent_spec_extracts_occlusion_without_pseudo_subjects() -> None:
    intent = extract_intent_spec(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )

    assert intent.colors["screen"] == "red"
    assert intent.colors["suitcase"] == "green"
    assert "screen hides lower" not in intent.subjects
    assert "lower half" not in intent.subjects
    assert "screen" in intent.subjects
    assert "suitcase" in intent.subjects
    assert "suitcase handle" in intent.subjects
    relation = next(
        item for item in intent.interaction_relations if item["type"] == "occlusion"
    )
    assert relation["subject"] == "screen"
    assert relation["object"] == "suitcase"
    assert relation["hidden_part"] == "lower half"
    assert relation["visible_part"] == "suitcase handle"


def test_intent_spec_keeps_attribute_rich_subjects() -> None:
    intent = extract_intent_spec(
        "A tiny shiny metal robot stands beside a huge matte wooden crate; "
        "keep the size, material, and surface differences clear."
    )

    assert "metal robot" in intent.subjects or "robot" in intent.subjects
    assert "wooden crate" in intent.subjects or "crate" in intent.subjects
    assert any("metal" in values for values in intent.attributes.values())
    assert any("wooden" in values for values in intent.attributes.values())


def test_intent_spec_cleans_benchmark_action_tails_and_counts() -> None:
    intent = extract_intent_spec(
        "A white cat sits in front of a black bicycle, while a yellow bird "
        "perches above the bicycle seat."
    )

    assert intent.colors["cat"] == "white"
    assert intent.colors["bird"] == "yellow"
    assert "cat sits" not in intent.subjects
    assert "bird perches" not in intent.subjects
    assert any(
        item["subject"] == "cat"
        and item["phrase"] == "in front of"
        and item["object"] == "bicycle"
        for item in intent.relations
    )
    assert any(
        item["subject"] == "bird"
        and item["phrase"] == "above"
        and item["object"] == "bicycle seat"
        for item in intent.relations
    )

    counted = extract_intent_spec(
        "A transparent box contains one silver key and one black feather, "
        "without any coins or jewelry."
    )
    assert counted.subjects == ["key", "feather", "box"]
    assert counted.counts == {"key": 1, "feather": 1}
    assert "transparent box contains" not in counted.subjects


def test_intent_spec_preserves_same_class_color_bindings_and_symbol_relation() -> None:
    intent = extract_intent_spec(
        "A blue notebook shows a yellow star symbol on its cover, next to a "
        "plain green notebook with no symbol."
    )

    assert intent.colors["blue notebook"] == "blue"
    assert intent.colors["green notebook"] == "green"
    assert intent.colors["star symbol"] == "yellow"
    assert "notebook shows" not in intent.subjects
    assert "notebook" not in intent.subjects
    assert any(
        item["subject"] == "blue notebook"
        and item["action"] == "shows"
        and item["object"] == "star symbol"
        for item in intent.interaction_relations
    )
    assert any(
        item["subject"] == "blue notebook"
        and item["phrase"] == "next to"
        and item["object"] == "green notebook"
        for item in intent.relations
    )
    assert not any(
        item["subject"] == "star symbol"
        and item["phrase"] == "next to"
        and item["object"] == "green notebook"
        for item in intent.relations
    )


def test_intent_spec_handles_exact_text_without_text_layout_drift() -> None:
    intent = extract_intent_spec(
        "A black sign displays the exact yellow text 'GO' above a plain green "
        "sign with no text."
    )

    assert "exact text" not in intent.subjects
    assert intent.colors["text"] == "yellow"
    assert any(
        item["subject"] == "black sign"
        and item["action"] == "displays"
        and item["object"] == "text"
        for item in intent.interaction_relations
    )
    assert any(
        item["subject"] == "black sign"
        and item["phrase"] == "above"
        and item["object"] == "green sign"
        for item in intent.relations
    )


def test_intent_spec_removes_negative_clause_pseudo_subjects() -> None:
    intent = extract_intent_spec(
        "A striped ceramic mug is next to a polka-dot metal lunchbox; "
        "the patterns must not swap between the objects."
    )

    assert "striped ceramic mug" in intent.subjects
    assert "polka-dot metal lunchbox" in intent.subjects
    assert "patterns must not" not in intent.subjects
    assert "patterns" not in intent.subjects
    assert "the patterns must not swap between the objects" in intent.negative_constraints


def test_intent_spec_does_not_treat_negative_attached_as_positive_action() -> None:
    intent = extract_intent_spec(
        "A green wizard carries a silver lantern while standing beside an orange "
        "barrel; the lantern is not attached to the barrel."
    )

    assert "attached to" not in intent.actions
    assert "the lantern is not attached to the barrel" in intent.negative_constraints
    assert {
        (item["subject"], item["action"], item["object"])
        for item in intent.interaction_relations
    } == {("wizard", "carries", "lantern")}


def test_holdout_parser_removes_verb_phrase_subject_tails() -> None:
    fish = extract_intent_spec(
        "Exactly four blue fish swim through one orange hoop, with no fifth fish "
        "and no extra hoop."
    )
    dog = extract_intent_spec(
        "A gray dog stands behind a teal bench, while a pink ball rests under the bench."
    )

    assert "fish" in fish.subjects
    assert "fish swim" not in fish.subjects
    assert fish.colors["fish"] == "blue"
    assert fish.counts["fish"] == 4
    assert "dog" in dog.subjects
    assert "ball" in dog.subjects
    assert "dog stands" not in dog.subjects
    assert "ball rests" not in dog.subjects
    assert dog.colors["dog"] == "gray"
    assert dog.colors["ball"] == "pink"
    assert any(
        item["subject"] == "dog"
        and item["phrase"] == "behind"
        and item["object"] == "bench"
        for item in dog.relations
    )
    assert any(
        item["subject"] == "ball"
        and item["phrase"] == "under"
        and item["object"] == "bench"
        for item in dog.relations
    )


def test_intent_spec_filters_edit_run_pseudo_subject_fragments() -> None:
    cabinet = extract_intent_spec(
        "A red screen hides the lower half of a white cabinet, a blue label on "
        "the cabinet reads 'A1', and no handle is visible."
    )
    suitcase = extract_intent_spec(
        "A red screen hides the lower half of a green suitcase, while the "
        "suitcase handle remains clearly visible."
    )
    door = extract_intent_spec(
        "A blue robot holds a silver key while standing left of a red door; "
        "the door has no window."
    )

    assert "screen" in cabinet.subjects
    assert "cabinet" in cabinet.subjects
    assert "label" in cabinet.subjects
    assert "cabinet reads" not in cabinet.subjects
    assert "lower" not in cabinet.subjects
    assert "lower half" not in cabinet.subjects
    assert "screen" in suitcase.subjects
    assert "suitcase" in suitcase.subjects
    assert "lower" not in suitcase.subjects
    assert "has no window" not in door.subjects
