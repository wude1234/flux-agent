from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"


CASES_BY_CATEGORY = {
    "count_quantity": [
        ("ext90_count_001", "Exactly four teal buttons lie inside one shallow wooden box; no button is outside the box.", ["exact_count", "container", "negative_extra"]),
        ("ext90_count_002", "Exactly two red pencils and exactly three blue pencils rest on a white desk; no extra pencil is present.", ["multi_object_count", "color_count", "negative_extra"]),
        ("ext90_count_003", "Five yellow stars are printed in a single row on a black card, and only the middle star is outlined in white.", ["exact_count", "state_count", "arrangement"]),
        ("ext90_count_004", "Exactly one green bottle stands between exactly two purple cups on a gray shelf.", ["multi_object_count", "between_relation"]),
        ("ext90_count_005", "A silver plate holds exactly six black grapes arranged in a circle.", ["exact_count", "arrangement"]),
        ("ext90_count_006", "Exactly three orange balloons float above one small chair; no fourth balloon appears.", ["exact_count", "negative_extra", "above_relation"]),
        ("ext90_count_007", "Two cyan books are stacked on top of one magenta book; only these three books are visible.", ["exact_count", "stacking", "negative_extra"]),
        ("ext90_count_008", "Exactly one striped sock and exactly one plain sock hang from a rope, with no third sock.", ["exact_count", "attribute_count", "negative_extra"]),
        ("ext90_count_009", "Four white dice sit on a blue mat, and exactly one die shows a red top face.", ["exact_count", "state_count", "color_binding"]),
        ("ext90_count_010", "Exactly two gold keys are inside a transparent jar, and exactly one silver key lies outside the jar.", ["multi_object_count", "inside_outside"]),
    ],
    "spatial_layout": [
        ("ext90_spatial_001", "A red cube is left of a blue sphere, and the blue sphere is below a yellow cone; all three objects are separated.", ["left_right", "above_below", "chain"]),
        ("ext90_spatial_002", "A green triangle is above a purple square, while an orange circle is right of the square.", ["above_below", "left_right", "chain"]),
        ("ext90_spatial_003", "A black mug is in front of a white plate, and a small red spoon is behind the plate.", ["front_behind", "three_object_layout"]),
        ("ext90_spatial_004", "A yellow cylinder sits in the top-left corner, and a blue cube sits in the bottom-right corner.", ["corner_layout", "diagonal"]),
        ("ext90_spatial_005", "A pink pyramid is between a gray sphere on the left and a brown box on the right.", ["between_relation", "left_right"]),
        ("ext90_spatial_006", "A cyan vase is below a red frame, and a green candle is above the frame.", ["above_below", "chain"]),
        ("ext90_spatial_007", "A blue bowl is to the right of a yellow fork, and both are below a red napkin.", ["left_right", "above_below", "group_relation"]),
        ("ext90_spatial_008", "A white cube is directly under a black ring, and a small orange star is directly over the ring.", ["direct_alignment", "above_below"]),
        ("ext90_spatial_009", "A purple ball is left of a green block, and the green block is left of a red block in one horizontal line.", ["ordered_sequence", "left_right"]),
        ("ext90_spatial_010", "A silver coin is centered between a blue card above it and a red card below it.", ["center_between", "vertical_layout"]),
    ],
    "attribute_binding": [
        ("ext90_attribute_001", "A cracked ceramic bowl sits beside a smooth metal bowl; both bowls are unpainted white.", ["material_binding", "surface_state"]),
        ("ext90_attribute_002", "A wet leather glove and a dry wool glove lie on a wooden bench.", ["state_binding", "material_binding"]),
        ("ext90_attribute_003", "A striped fabric kite and a plain paper kite float in a clear sky.", ["pattern_binding", "material_binding"]),
        ("ext90_attribute_004", "A transparent glass apple and an opaque plastic apple sit on a black tray.", ["transparency_binding", "material_binding"]),
        ("ext90_attribute_005", "A folded silk scarf lies next to an unfolded cotton scarf.", ["state_binding", "material_binding"]),
        ("ext90_attribute_006", "A rusty iron lock is attached to a polished brass lock on a gray door.", ["surface_state", "material_binding"]),
        ("ext90_attribute_007", "A square wooden plate and a round stone plate rest on a white table.", ["shape_binding", "material_binding"]),
        ("ext90_attribute_008", "A torn paper flag is beside an intact plastic flag.", ["state_binding", "material_binding"]),
        ("ext90_attribute_009", "A fluffy wool pillow sits above a flat rubber pillow.", ["texture_binding", "material_binding", "spatial_relation"]),
        ("ext90_attribute_010", "A glowing glass cube is left of a matte clay cube.", ["state_binding", "material_binding", "left_right"]),
    ],
    "color_binding": [
        ("ext90_color_001", "A turquoise velvet hat, a crimson wooden comb, and a silver paper lantern sit on a black cloth.", ["rare_color", "material_binding", "three_objects"]),
        ("ext90_color_002", "An indigo metal spoon, a gold ceramic cup, and a pink fabric ribbon rest on a white tray.", ["rare_color", "material_binding", "three_objects"]),
        ("ext90_color_003", "A lime glass bottle, a maroon rubber ball, and a cyan stone cube stand on a gray shelf.", ["color_material_binding", "three_objects"]),
        ("ext90_color_004", "A navy paper fan is beside a coral leather wallet and a bronze plastic whistle.", ["color_material_binding", "three_objects"]),
        ("ext90_color_005", "A violet wooden chair, an amber glass lamp, and a white metal key are arranged on a red rug.", ["color_material_binding", "three_objects"]),
        ("ext90_color_006", "A beige ceramic turtle, a teal fabric pouch, and a black rubber stamp lie on a yellow board.", ["color_material_binding", "three_objects"]),
        ("ext90_color_007", "A magenta stone vase stands next to an olive paper box and a silver cloth flower.", ["color_material_binding", "three_objects"]),
        ("ext90_color_008", "A red glass pear, a blue wooden pear, and a green metal pear sit in separate bowls.", ["same_object_type", "color_material_binding"]),
        ("ext90_color_009", "A purple plastic spoon lies under an orange cloth napkin, beside a turquoise ceramic bowl.", ["color_material_binding", "under_relation"]),
        ("ext90_color_010", "A black velvet mask, a yellow metal bell, and a pink paper envelope are on a white table.", ["color_material_binding", "three_objects"]),
    ],
    "interaction_relation": [
        ("ext90_interaction_001", "A small wooden hand grips the handle of a red umbrella, with the umbrella clearly not touching the ground.", ["gripping", "handle_contact"]),
        ("ext90_interaction_002", "A green clamp is attached to the rim of a blue bucket.", ["attached_to", "rim_contact"]),
        ("ext90_interaction_003", "A silver hook holds a yellow ring, and the ring hangs below the hook.", ["holding", "hanging_contact"]),
        ("ext90_interaction_004", "A pink ribbon is tied around the neck of a white bottle.", ["tied_around", "contact_relation"]),
        ("ext90_interaction_005", "A black magnet touches the left side of a gray metal box.", ["touching", "side_contact"]),
        ("ext90_interaction_006", "A red clothespin clips the top edge of a blue card.", ["clipping", "edge_contact"]),
        ("ext90_interaction_007", "A yellow handle is inserted into the slot of a purple drawer.", ["inserted_into", "slot_contact"]),
        ("ext90_interaction_008", "A brown rope loops through the hole of a silver key.", ["through_relation", "hole_contact"]),
        ("ext90_interaction_009", "A clear suction cup sticks to the front of a green tile.", ["attached_to", "surface_contact"]),
        ("ext90_interaction_010", "A blue glove holds the stem of a red flower without covering the flower head.", ["holding", "occlusion_avoidance"]),
    ],
    "negation_absence": [
        ("ext90_negation_001", "A yellow bowl sits on a table with no spoon, fork, or chopsticks visible.", ["forbidden_objects", "single_subject"]),
        ("ext90_negation_002", "A plain blue notebook lies closed with no logo, text, or symbol on its cover.", ["forbidden_symbol", "plain_object"]),
        ("ext90_negation_003", "A red bicycle stands alone with no basket attached to the front.", ["forbidden_part", "object_state"]),
        ("ext90_negation_004", "A white mug has no handle and no printed design.", ["forbidden_part", "forbidden_symbol"]),
        ("ext90_negation_005", "A green plate holds one sandwich and no chips or salad.", ["forbidden_objects", "food_scene"]),
        ("ext90_negation_006", "A black suitcase is closed with no visible zipper pull.", ["forbidden_part", "closed_state"]),
        ("ext90_negation_007", "A gray wall clock shows no numbers and no hands.", ["forbidden_parts", "plain_object"]),
        ("ext90_negation_008", "A transparent jar contains only marbles and no coins.", ["forbidden_object", "container"]),
        ("ext90_negation_009", "A purple hat sits on a stool with no feather attached.", ["forbidden_part", "object_relation"]),
        ("ext90_negation_010", "A tan envelope is sealed and has no stamp or address text.", ["forbidden_symbol", "forbidden_text"]),
    ],
    "occlusion_visibility": [
        ("ext90_occlusion_001", "A red screen hides the lower half of a green backpack, while the top handle remains clearly visible.", ["partial_occlusion", "visible_part", "occluder_insertion"]),
        ("ext90_occlusion_002", "A blue cloth covers the right half of a yellow drum, while the left rim remains visible.", ["partial_occlusion", "side_occlusion"]),
        ("ext90_occlusion_003", "A black board blocks the center of a white poster, while all four poster corners remain visible.", ["center_occlusion", "visible_corners"]),
        ("ext90_occlusion_004", "A purple curtain hides the left side of a silver chair, while the chair legs remain visible.", ["side_occlusion", "visible_part"]),
        ("ext90_occlusion_005", "An orange box sits in front of a cyan vase, hiding only the vase bottom.", ["front_occlusion", "visible_top"]),
        ("ext90_occlusion_006", "A gray card covers the top half of a pink book, while the bottom title area is visible.", ["top_occlusion", "visible_part"]),
        ("ext90_occlusion_007", "A green leaf covers one corner of a red square, with the other three corners visible.", ["corner_occlusion", "visible_corners"]),
        ("ext90_occlusion_008", "A white towel covers the middle of a brown guitar, while the headstock and body bottom are visible.", ["middle_occlusion", "visible_parts"]),
        ("ext90_occlusion_009", "A yellow sticky note hides the number area of a black calculator, while the calculator edges remain visible.", ["localized_occlusion", "visible_edges"]),
        ("ext90_occlusion_010", "A blue panel covers the lower left part of a white cabinet, while the upper right door remains visible.", ["corner_occlusion", "visible_part"]),
    ],
    "text_symbol": [
        ("ext90_text_001", "A black sign displays the exact white text 'UP' above a plain red sign with no text.", ["exact_text", "plain_companion", "spatial_relation"]),
        ("ext90_text_002", "A green folder shows a white circle symbol on its front, next to a plain yellow folder with no symbol.", ["symbol_binding", "plain_companion"]),
        ("ext90_text_003", "A blue badge displays the exact yellow text 'OK', while a nearby orange badge has no text.", ["exact_text", "plain_companion"]),
        ("ext90_text_004", "A purple card shows a silver star symbol in its center, beside a plain white card with no symbol.", ["symbol_binding", "plain_companion"]),
        ("ext90_text_005", "A red label reads exactly 'A7' in black letters, placed below a plain gray label.", ["exact_text", "spatial_relation", "plain_companion"]),
        ("ext90_text_006", "A cyan sticker has a black triangle symbol, and a pink sticker beside it is completely blank.", ["symbol_binding", "plain_companion"]),
        ("ext90_text_007", "A white flag displays the exact red text 'NO', and a black flag beside it has no text.", ["exact_text", "plain_companion"]),
        ("ext90_text_008", "A yellow tile shows one blue plus symbol, next to a plain green tile with no symbol.", ["symbol_binding", "plain_companion"]),
        ("ext90_text_009", "A brown book cover displays the exact gold text 'MAP', while a blue book cover beside it is blank.", ["exact_text", "plain_companion"]),
        ("ext90_text_010", "A silver button has a black moon symbol, and a red button below it has no symbol.", ["symbol_binding", "spatial_relation", "plain_companion"]),
    ],
    "multi_compositional": [
        ("ext90_multi_001", "Exactly two teal cups are left of one orange plate, and a purple spoon lies under the plate; no fork is present.", ["count", "spatial", "negative_object"]),
        ("ext90_multi_002", "A red screen hides the lower half of a white cabinet, a blue label on the cabinet reads 'B2', and no handle is visible.", ["occlusion", "exact_text", "negative_part"]),
        ("ext90_multi_003", "A yellow cube is above a green sphere, both are right of a red cone, and no extra shape appears.", ["spatial_chain", "negative_extra"]),
        ("ext90_multi_004", "A black sign displays the exact white text 'IN' above a plain blue sign with no text, and one pink ball sits to the left of the blue sign.", ["exact_text", "plain_companion", "spatial"]),
        ("ext90_multi_005", "A purple cloth covers the right half of a green suitcase, while exactly two silver keys lie below it and no third key is present.", ["occlusion", "count", "negative_extra"]),
        ("ext90_multi_006", "A turquoise wooden chair is left of a crimson glass lamp, and a silver paper fan rests under the lamp on a black rug.", ["color_material", "spatial", "under_relation"]),
        ("ext90_multi_007", "A blue clamp is attached to a yellow box, and exactly one red sticker with a white star is on the box.", ["interaction", "count", "symbol"]),
        ("ext90_multi_008", "A green curtain hides the left half of a brown drum, a white label on the drum reads 'C3', and no stick is visible.", ["occlusion", "exact_text", "negative_object"]),
        ("ext90_multi_009", "Exactly three black coins are inside a clear jar, the jar is below a red shelf, and no coins are outside.", ["count", "container", "spatial", "negative_extra"]),
        ("ext90_multi_010", "A pink ribbon loops through the hole of one silver key, and the key lies right of a plain blue tag with no text.", ["interaction", "spatial", "negative_text"]),
    ],
}


def _case(case_id: str, category: str, prompt: str, focus: list[str]) -> dict[str, object]:
    return {
        "id": case_id,
        "category": category,
        "prompt": prompt,
        "focus": focus,
    }


def _build_cases(limit_per_category: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for category, entries in CASES_BY_CATEGORY.items():
        for case_id, prompt, focus in entries[:limit_per_category]:
            cases.append(_case(case_id, category, prompt, focus))
    return cases


def _payload(name: str, cases: list[dict[str, object]]) -> dict[str, object]:
    categories = list(CASES_BY_CATEGORY)
    return {
        "version": name,
        "description": (
            "Fresh anti-overfit extension prompts for the FLUX agent nightly "
            "22-45-90 experiment. These prompts are category-level variants, "
            "not prompt-specific patches."
        ),
        "categories": categories,
        "anti_overfit_policy": {
            "source": "new prompts, same failure categories as the 90-case baseline",
            "rule": "do not use these prompts for prompt-specific code paths",
            "new90_relation": "new45 is the first five cases per category from new90",
        },
        "cases": cases,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    new45 = _payload("extension_weak_categories_45_2026_06_24", _build_cases(5))
    new90 = _payload("extension_new90_2026_06_24", _build_cases(10))
    path45 = BENCHMARK_DIR / "hard_prompts_extension_weak_categories_45.json"
    path90 = BENCHMARK_DIR / "hard_prompts_extension_new90.json"
    _write(path45, new45)
    _write(path90, new90)
    print(json.dumps({"new45": str(path45), "new45_cases": 45, "new90": str(path90), "new90_cases": 90}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
