from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEV_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "compact_dev_single_color_001",
        "category": "color_binding",
        "prompt": (
            "A turquoise wooden chair sits alone on a plain black floor; the chair "
            "must be turquoise and visibly wooden."
        ),
        "focus": ["color_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["turquoise wooden chair", "black floor"],
            "counts": {"chair": 1},
            "colors": {"chair": "turquoise", "floor": "black"},
            "attributes": {"chair": ["wooden"]},
        },
    },
    {
        "id": "compact_dev_single_color_002",
        "category": "color_binding",
        "prompt": (
            "A magenta metal whistle lies beside a teal paper envelope; each object "
            "keeps its own color and material."
        ),
        "focus": ["color_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["magenta metal whistle", "teal paper envelope"],
            "counts": {"whistle": 1, "envelope": 1},
            "colors": {"whistle": "magenta", "envelope": "teal"},
            "attributes": {"whistle": ["metal"], "envelope": ["paper"]},
            "relations": ["beside"],
        },
    },
    {
        "id": "compact_dev_single_spatial_001",
        "category": "spatial_layout",
        "prompt": (
            "A yellow pyramid is clearly right of a red cylinder; only these two "
            "objects are visible."
        ),
        "focus": ["right-of", "object_separation", "single_axis"],
        "expected": {
            "subjects": ["yellow pyramid", "red cylinder"],
            "counts": {"pyramid": 1, "cylinder": 1},
            "colors": {"pyramid": "yellow", "cylinder": "red"},
            "spatial_relations": [
                {"subject": "pyramid", "relation": "right_of", "object": "cylinder"}
            ],
        },
    },
    {
        "id": "compact_dev_single_spatial_002",
        "category": "spatial_layout",
        "prompt": (
            "A silver spoon is clearly under a green arch; only the spoon and arch "
            "are visible."
        ),
        "focus": ["under", "object_separation", "single_axis"],
        "expected": {
            "subjects": ["silver spoon", "green arch"],
            "counts": {"spoon": 1, "arch": 1},
            "colors": {"spoon": "silver", "arch": "green"},
            "spatial_relations": [
                {"subject": "spoon", "relation": "under", "object": "arch"}
            ],
        },
    },
    {
        "id": "compact_dev_single_interaction_001",
        "category": "interaction_relation",
        "prompt": (
            "A green robot grips a silver handle with one hand; the handle is a "
            "separate object."
        ),
        "focus": ["interaction_relation", "part_relation", "single_axis"],
        "expected": {
            "subjects": ["green robot", "silver handle"],
            "counts": {"robot": 1, "handle": 1},
            "colors": {"robot": "green", "handle": "silver"},
            "interaction_relations": [
                {"subject": "robot", "action": "grips", "object": "handle"}
            ],
        },
    },
    {
        "id": "compact_dev_single_interaction_002",
        "category": "interaction_relation",
        "prompt": (
            "A red bear touches the side of a yellow drum with one paw; the drum is "
            "a separate object."
        ),
        "focus": ["interaction_relation", "touch_relation", "single_axis"],
        "expected": {
            "subjects": ["red bear", "yellow drum"],
            "counts": {"bear": 1, "drum": 1},
            "colors": {"bear": "red", "drum": "yellow"},
            "interaction_relations": [
                {"subject": "bear", "action": "touches", "object": "drum side"}
            ],
        },
    },
    {
        "id": "compact_dev_single_count_001",
        "category": "count_quantity",
        "prompt": (
            "Exactly three blue blocks are on a white table, with no fourth block."
        ),
        "focus": ["count", "negative_count", "single_axis"],
        "expected": {
            "subjects": ["blue blocks", "white table"],
            "counts": {"blue blocks": 3},
            "colors": {"blocks": "blue", "table": "white"},
            "negative_constraints": ["no fourth block"],
        },
    },
    {
        "id": "compact_dev_single_count_002",
        "category": "count_quantity",
        "prompt": "Exactly five green buttons are in a straight row, with no sixth button.",
        "focus": ["count", "negative_count", "single_axis"],
        "expected": {
            "subjects": ["green buttons"],
            "counts": {"green buttons": 5},
            "colors": {"buttons": "green"},
            "negative_constraints": ["no sixth button"],
        },
    },
    {
        "id": "compact_dev_single_negation_001",
        "category": "negation_absence",
        "prompt": "One pink scarf lies on a wooden chair, and no hat is present.",
        "focus": ["negation_absence", "count", "single_axis"],
        "expected": {
            "subjects": ["pink scarf", "wooden chair"],
            "counts": {"scarf": 1},
            "colors": {"scarf": "pink"},
            "attributes": {"chair": ["wooden"]},
            "negative_constraints": ["no hat"],
        },
    },
    {
        "id": "compact_dev_single_negation_002",
        "category": "negation_absence",
        "prompt": "One blue cup sits on a gray table, and no spoon is present.",
        "focus": ["negation_absence", "count", "single_axis"],
        "expected": {
            "subjects": ["blue cup", "gray table"],
            "counts": {"cup": 1},
            "colors": {"cup": "blue", "table": "gray"},
            "negative_constraints": ["no spoon"],
        },
    },
    {
        "id": "compact_dev_single_text_symbol_001",
        "category": "text_symbol",
        "prompt": "A black sign displays the exact yellow text 'GO'.",
        "focus": ["exact_text", "text_color", "single_axis"],
        "expected": {
            "subjects": ["black sign", "yellow text"],
            "colors": {"sign": "black", "text": "yellow"},
            "text": [{"object": "sign", "value": "GO", "color": "yellow"}],
        },
    },
    {
        "id": "compact_dev_single_text_symbol_002",
        "category": "text_symbol",
        "prompt": "A purple folder shows one white triangle symbol on its front.",
        "focus": ["symbol_binding", "symbol_color", "single_axis"],
        "expected": {
            "subjects": ["purple folder", "white triangle symbol"],
            "counts": {"triangle symbol": 1},
            "colors": {"folder": "purple", "triangle symbol": "white"},
            "symbols": [{"object": "folder front", "value": "triangle", "color": "white"}],
        },
    },
    {
        "id": "compact_dev_single_attribute_001",
        "category": "attribute_binding",
        "prompt": (
            "A small rough ceramic turtle stands beside a large smooth plastic box; "
            "keep the size, texture, and material differences clear."
        ),
        "focus": ["size_binding", "texture_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["ceramic turtle", "plastic box"],
            "attributes": {
                "turtle": ["small", "rough", "ceramic"],
                "box": ["large", "smooth", "plastic"],
            },
            "relations": ["beside"],
        },
    },
    {
        "id": "compact_dev_single_attribute_002",
        "category": "attribute_binding",
        "prompt": (
            "A striped fabric pouch is next to a checkered rubber ball; the pattern "
            "and material must not swap."
        ),
        "focus": ["pattern_binding", "material_binding", "negative_attribute_leakage", "single_axis"],
        "expected": {
            "subjects": ["fabric pouch", "rubber ball"],
            "attributes": {
                "pouch": ["striped", "fabric"],
                "ball": ["checkered", "rubber"],
            },
            "negative_constraints": ["pattern and material must not swap"],
        },
    },
    {
        "id": "compact_dev_single_occlusion_001",
        "category": "occlusion_visibility",
        "prompt": (
            "A yellow cloth partially covers a blue vase, but the vase base remains visible."
        ),
        "focus": ["occlusion_visibility", "part_visibility", "single_axis"],
        "expected": {
            "subjects": ["yellow cloth", "blue vase", "vase base"],
            "colors": {"cloth": "yellow", "vase": "blue"},
            "occlusion": [{"occluder": "cloth", "target": "vase", "degree": "partial"}],
            "attributes": {"vase base": ["visible"]},
        },
    },
    {
        "id": "compact_dev_single_occlusion_002",
        "category": "occlusion_visibility",
        "prompt": (
            "A red screen hides the lower half of a green suitcase, while the "
            "suitcase handle remains clearly visible."
        ),
        "focus": ["occlusion_visibility", "part_visibility", "single_axis"],
        "expected": {
            "subjects": ["red screen", "green suitcase", "suitcase handle"],
            "colors": {"screen": "red", "suitcase": "green"},
            "occlusion": [{"occluder": "screen", "target": "suitcase lower half", "degree": "hidden"}],
            "attributes": {"suitcase handle": ["clearly visible"]},
        },
    },
    {
        "id": "compact_dev_scene_001",
        "category": "multi_compositional",
        "prompt": (
            "Exactly two cyan ceramic mugs are left of one orange wooden tray, "
            "and a purple spoon lies under the tray; no extra mug or fork is present."
        ),
        "focus": [
            "count",
            "color_binding",
            "material_binding",
            "left-right",
            "under",
            "negation_absence",
        ],
        "expected": {
            "subjects": ["cyan ceramic mugs", "orange wooden tray", "purple spoon"],
            "counts": {"cyan ceramic mugs": 2, "orange wooden tray": 1, "purple spoon": 1},
            "colors": {"mugs": "cyan", "tray": "orange", "spoon": "purple"},
            "attributes": {"mugs": ["ceramic"], "tray": ["wooden"]},
            "spatial_relations": [
                {"subject": "mugs", "relation": "left_of", "object": "tray"},
                {"subject": "spoon", "relation": "under", "object": "tray"},
            ],
            "negative_constraints": ["no extra mug", "no fork"],
        },
    },
    {
        "id": "compact_dev_scene_002",
        "category": "multi_compositional",
        "prompt": (
            "A green robot grips a silver paper fan handle while standing in front "
            "of a red glass lamp; the fan is not attached to the lamp."
        ),
        "focus": [
            "interaction_relation",
            "part_relation",
            "material_binding",
            "front-back",
            "negative_relation",
            "color_binding",
        ],
        "expected": {
            "subjects": ["green robot", "silver paper fan handle", "red glass lamp"],
            "colors": {"robot": "green", "fan handle": "silver", "lamp": "red"},
            "attributes": {"fan": ["paper"], "lamp": ["glass"]},
            "interaction_relations": [
                {"subject": "robot", "action": "grips", "object": "fan handle"}
            ],
            "spatial_relations": [
                {"subject": "robot", "relation": "in_front_of", "object": "lamp"}
            ],
            "negative_constraints": ["fan is not attached to the lamp"],
        },
    },
    {
        "id": "compact_dev_scene_003",
        "category": "multi_compositional",
        "prompt": (
            "A black sign displays the exact yellow text 'NO' above a plain blue "
            "sign with no text, and one pink ball sits to the right of the blue sign."
        ),
        "focus": [
            "exact_text",
            "negative_text",
            "same-class_objects",
            "above",
            "right-of",
            "count",
        ],
        "expected": {
            "subjects": ["black sign", "blue sign", "pink ball"],
            "counts": {"pink ball": 1},
            "colors": {"sign with text": "black", "plain sign": "blue", "text": "yellow", "ball": "pink"},
            "text": [{"object": "black sign", "value": "NO", "color": "yellow"}],
            "spatial_relations": [
                {"subject": "black sign", "relation": "above", "object": "blue sign"},
                {"subject": "ball", "relation": "right_of", "object": "blue sign"},
            ],
            "negative_constraints": ["blue sign has no text"],
        },
    },
    {
        "id": "compact_dev_scene_004",
        "category": "multi_compositional",
        "prompt": (
            "A transparent box contains exactly one gold key and three white shells; "
            "a teal fabric pouch is behind the box, with no coins or rings."
        ),
        "focus": [
            "container_count",
            "material_binding",
            "behind",
            "negation_absence",
            "color_binding",
        ],
        "expected": {
            "subjects": ["transparent box", "gold key", "white shells", "teal fabric pouch"],
            "counts": {"key": 1, "white shells": 3, "fabric pouch": 1},
            "colors": {"key": "gold", "shells": "white", "pouch": "teal"},
            "attributes": {"pouch": ["fabric"]},
            "spatial_relations": [
                {"subject": "key", "relation": "inside", "object": "box"},
                {"subject": "shells", "relation": "inside", "object": "box"},
                {"subject": "pouch", "relation": "behind", "object": "box"},
            ],
            "negative_constraints": ["no coins", "no rings"],
        },
    },
    {
        "id": "compact_dev_scene_005",
        "category": "multi_compositional",
        "prompt": (
            "A yellow blanket partially covers a blue metal suitcase, but the "
            "suitcase handle remains visible; one red sticker on the suitcase shows "
            "a white star symbol."
        ),
        "focus": [
            "occlusion_visibility",
            "part_visibility",
            "material_binding",
            "symbol_binding",
            "color_binding",
            "count",
        ],
        "expected": {
            "subjects": ["yellow blanket", "blue metal suitcase", "suitcase handle", "red sticker", "white star symbol"],
            "counts": {"red sticker": 1},
            "colors": {"blanket": "yellow", "suitcase": "blue", "sticker": "red", "star symbol": "white"},
            "attributes": {"suitcase": ["metal"], "suitcase handle": ["visible"]},
            "occlusion": [{"occluder": "blanket", "target": "suitcase", "degree": "partial"}],
            "symbols": [{"object": "red sticker", "value": "star", "color": "white"}],
        },
    },
    {
        "id": "compact_dev_scene_006",
        "category": "multi_compositional",
        "prompt": (
            "A white cat touches a brown wooden drum while holding a blue plastic "
            "brush, and a green cube is below the drum; no second brush is visible."
        ),
        "focus": [
            "two_actions",
            "material_binding",
            "below",
            "count_negation",
            "color_binding",
        ],
        "expected": {
            "subjects": ["white cat", "brown wooden drum", "blue plastic brush", "green cube"],
            "counts": {"brush": 1},
            "colors": {"cat": "white", "drum": "brown", "brush": "blue", "cube": "green"},
            "attributes": {"drum": ["wooden"], "brush": ["plastic"]},
            "interaction_relations": [
                {"subject": "cat", "action": "touches", "object": "drum"},
                {"subject": "cat", "action": "holding", "object": "brush"},
            ],
            "spatial_relations": [
                {"subject": "cube", "relation": "under", "object": "drum"}
            ],
            "negative_constraints": ["no second brush"],
        },
    },
)


HOLDOUT_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "compact_holdout_single_color_001",
        "category": "color_binding",
        "prompt": (
            "A crimson glass lamp stands alone on a plain gray floor; the lamp "
            "must be crimson and visibly glass."
        ),
        "focus": ["color_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["crimson glass lamp", "gray floor"],
            "counts": {"lamp": 1},
            "colors": {"lamp": "crimson", "floor": "gray"},
            "attributes": {"lamp": ["glass"]},
        },
    },
    {
        "id": "compact_holdout_single_color_002",
        "category": "color_binding",
        "prompt": (
            "An indigo ceramic turtle rests beside a gold fabric pouch; each object "
            "keeps its own color and material."
        ),
        "focus": ["color_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["indigo ceramic turtle", "gold fabric pouch"],
            "counts": {"turtle": 1, "pouch": 1},
            "colors": {"turtle": "indigo", "pouch": "gold"},
            "attributes": {"turtle": ["ceramic"], "pouch": ["fabric"]},
            "relations": ["beside"],
        },
    },
    {
        "id": "compact_holdout_single_spatial_001",
        "category": "spatial_layout",
        "prompt": (
            "A blue cone is clearly left of a green arch; only these two objects "
            "are visible."
        ),
        "focus": ["left-right", "object_separation", "single_axis"],
        "expected": {
            "subjects": ["blue cone", "green arch"],
            "counts": {"cone": 1, "arch": 1},
            "colors": {"cone": "blue", "arch": "green"},
            "spatial_relations": [
                {"subject": "cone", "relation": "left_of", "object": "arch"}
            ],
        },
    },
    {
        "id": "compact_holdout_single_spatial_002",
        "category": "spatial_layout",
        "prompt": (
            "A pink ball is clearly under a teal bench; only the ball and bench are visible."
        ),
        "focus": ["under", "object_separation", "single_axis"],
        "expected": {
            "subjects": ["pink ball", "teal bench"],
            "counts": {"ball": 1, "bench": 1},
            "colors": {"ball": "pink", "bench": "teal"},
            "spatial_relations": [
                {"subject": "ball", "relation": "under", "object": "bench"}
            ],
        },
    },
    {
        "id": "compact_holdout_single_interaction_001",
        "category": "interaction_relation",
        "prompt": (
            "A purple fox holds a gold rope in its front paws; the rope is a "
            "separate object."
        ),
        "focus": ["interaction_relation", "hand-object_binding", "single_axis"],
        "expected": {
            "subjects": ["purple fox", "gold rope"],
            "counts": {"fox": 1, "rope": 1},
            "colors": {"fox": "purple", "rope": "gold"},
            "interaction_relations": [
                {"subject": "fox", "action": "holds", "object": "rope"}
            ],
        },
    },
    {
        "id": "compact_holdout_single_interaction_002",
        "category": "interaction_relation",
        "prompt": (
            "A gray dog touches the top of a turquoise bell with one paw; the bell "
            "is a separate object."
        ),
        "focus": ["interaction_relation", "touch_relation", "single_axis"],
        "expected": {
            "subjects": ["gray dog", "turquoise bell"],
            "counts": {"dog": 1, "bell": 1},
            "colors": {"dog": "gray", "bell": "turquoise"},
            "interaction_relations": [
                {"subject": "dog", "action": "touches", "object": "bell top"}
            ],
        },
    },
    {
        "id": "compact_holdout_single_count_001",
        "category": "count_quantity",
        "prompt": (
            "Exactly four orange marbles are inside a clear bowl, with no fifth marble."
        ),
        "focus": ["count", "negative_count", "container", "single_axis"],
        "expected": {
            "subjects": ["orange marbles", "clear bowl"],
            "counts": {"orange marbles": 4},
            "colors": {"marbles": "orange"},
            "spatial_relations": [
                {"subject": "marbles", "relation": "inside", "object": "bowl"}
            ],
            "negative_constraints": ["no fifth marble"],
        },
    },
    {
        "id": "compact_holdout_single_count_002",
        "category": "count_quantity",
        "prompt": "Exactly two red toy boats float side by side, with no third boat.",
        "focus": ["count", "negative_count", "single_axis"],
        "expected": {
            "subjects": ["red toy boats"],
            "counts": {"red toy boats": 2},
            "colors": {"boats": "red"},
            "negative_constraints": ["no third boat"],
        },
    },
    {
        "id": "compact_holdout_single_negation_001",
        "category": "negation_absence",
        "prompt": "One yellow lemon rests on a blue plate, and no knife is present.",
        "focus": ["negation_absence", "count", "single_axis"],
        "expected": {
            "subjects": ["yellow lemon", "blue plate"],
            "counts": {"lemon": 1},
            "colors": {"lemon": "yellow", "plate": "blue"},
            "negative_constraints": ["no knife"],
        },
    },
    {
        "id": "compact_holdout_single_negation_002",
        "category": "negation_absence",
        "prompt": "One white shell sits inside a clear jar, and no bead is present.",
        "focus": ["negation_absence", "count", "single_axis"],
        "expected": {
            "subjects": ["white shell", "clear jar"],
            "counts": {"shell": 1},
            "colors": {"shell": "white"},
            "spatial_relations": [
                {"subject": "shell", "relation": "inside", "object": "jar"}
            ],
            "negative_constraints": ["no bead"],
        },
    },
    {
        "id": "compact_holdout_single_text_symbol_001",
        "category": "text_symbol",
        "prompt": "A green poster displays the exact white text 'SALE'.",
        "focus": ["exact_text", "text_color", "single_axis"],
        "expected": {
            "subjects": ["green poster", "white text"],
            "colors": {"poster": "green", "text": "white"},
            "text": [{"object": "poster", "value": "SALE", "color": "white"}],
        },
    },
    {
        "id": "compact_holdout_single_text_symbol_002",
        "category": "text_symbol",
        "prompt": "A red lunchbox shows one white moon symbol on its lid.",
        "focus": ["symbol_binding", "symbol_color", "single_axis"],
        "expected": {
            "subjects": ["red lunchbox", "white moon symbol"],
            "counts": {"moon symbol": 1},
            "colors": {"lunchbox": "red", "moon symbol": "white"},
            "symbols": [{"object": "lunchbox lid", "value": "moon", "color": "white"}],
        },
    },
    {
        "id": "compact_holdout_single_attribute_001",
        "category": "attribute_binding",
        "prompt": (
            "A tiny shiny metal robot stands beside a huge matte wooden crate; "
            "keep the size, material, and surface differences clear."
        ),
        "focus": ["size_binding", "surface_binding", "material_binding", "single_axis"],
        "expected": {
            "subjects": ["metal robot", "wooden crate"],
            "attributes": {
                "robot": ["tiny", "shiny", "metal"],
                "crate": ["huge", "matte", "wooden"],
            },
            "relations": ["beside"],
        },
    },
    {
        "id": "compact_holdout_single_attribute_002",
        "category": "attribute_binding",
        "prompt": (
            "A plaid fabric backpack is next to a checkered plastic bottle; the "
            "pattern and material must not swap."
        ),
        "focus": ["pattern_binding", "material_binding", "negative_attribute_leakage", "single_axis"],
        "expected": {
            "subjects": ["fabric backpack", "plastic bottle"],
            "attributes": {
                "backpack": ["plaid", "fabric"],
                "bottle": ["checkered", "plastic"],
            },
            "negative_constraints": ["pattern and material must not swap"],
        },
    },
    {
        "id": "compact_holdout_single_occlusion_001",
        "category": "occlusion_visibility",
        "prompt": (
            "A purple screen partially covers a white vase, but the vase base remains visible."
        ),
        "focus": ["occlusion_visibility", "part_visibility", "single_axis"],
        "expected": {
            "subjects": ["purple screen", "white vase", "vase base"],
            "colors": {"screen": "purple", "vase": "white"},
            "occlusion": [{"occluder": "screen", "target": "vase", "degree": "partial"}],
            "attributes": {"vase base": ["visible"]},
        },
    },
    {
        "id": "compact_holdout_single_occlusion_002",
        "category": "occlusion_visibility",
        "prompt": (
            "A yellow blanket hides the upper half of a green suitcase, while the "
            "suitcase handle remains clearly visible."
        ),
        "focus": ["occlusion_visibility", "part_visibility", "single_axis"],
        "expected": {
            "subjects": ["yellow blanket", "green suitcase", "suitcase handle"],
            "colors": {"blanket": "yellow", "suitcase": "green"},
            "occlusion": [{"occluder": "blanket", "target": "suitcase upper half", "degree": "hidden"}],
            "attributes": {"suitcase handle": ["clearly visible"]},
        },
    },
    {
        "id": "compact_holdout_scene_001",
        "category": "multi_compositional",
        "prompt": (
            "Exactly two magenta glass bottles are right of one green stone block, "
            "and a silver coin lies under the block; no extra bottle or key is present."
        ),
        "focus": ["count", "color_binding", "material_binding", "right-of", "under", "negation_absence"],
        "expected": {
            "subjects": ["magenta glass bottles", "green stone block", "silver coin"],
            "counts": {"magenta glass bottles": 2, "green stone block": 1, "silver coin": 1},
            "colors": {"bottles": "magenta", "block": "green", "coin": "silver"},
            "attributes": {"bottles": ["glass"], "block": ["stone"]},
            "spatial_relations": [
                {"subject": "bottles", "relation": "right_of", "object": "block"},
                {"subject": "coin", "relation": "under", "object": "block"},
            ],
            "negative_constraints": ["no extra bottle", "no key"],
        },
    },
    {
        "id": "compact_holdout_scene_002",
        "category": "multi_compositional",
        "prompt": (
            "A purple fox grips a gold fabric ribbon while standing behind a teal "
            "ceramic vase; the ribbon is not attached to the vase."
        ),
        "focus": ["interaction_relation", "material_binding", "behind", "negative_relation", "color_binding"],
        "expected": {
            "subjects": ["purple fox", "gold fabric ribbon", "teal ceramic vase"],
            "colors": {"fox": "purple", "ribbon": "gold", "vase": "teal"},
            "attributes": {"ribbon": ["fabric"], "vase": ["ceramic"]},
            "interaction_relations": [
                {"subject": "fox", "action": "grips", "object": "ribbon"}
            ],
            "spatial_relations": [
                {"subject": "fox", "relation": "behind", "object": "vase"}
            ],
            "negative_constraints": ["ribbon is not attached to the vase"],
        },
    },
    {
        "id": "compact_holdout_scene_003",
        "category": "multi_compositional",
        "prompt": (
            "A red folder displays a white moon symbol above a plain orange folder "
            "with no symbol, and one black cube sits left of the orange folder."
        ),
        "focus": ["symbol_binding", "negative_symbol", "same-class_objects", "above", "left-right", "count"],
        "expected": {
            "subjects": ["red folder", "orange folder", "black cube", "white moon symbol"],
            "counts": {"black cube": 1},
            "colors": {"folder with moon": "red", "plain folder": "orange", "cube": "black", "moon symbol": "white"},
            "symbols": [{"object": "red folder", "value": "moon", "color": "white"}],
            "spatial_relations": [
                {"subject": "red folder", "relation": "above", "object": "orange folder"},
                {"subject": "cube", "relation": "left_of", "object": "orange folder"},
            ],
            "negative_constraints": ["orange folder has no symbol"],
        },
    },
    {
        "id": "compact_holdout_scene_004",
        "category": "multi_compositional",
        "prompt": (
            "A clear bowl contains exactly one crimson marble and three yellow leaves; "
            "a blue paper fan is in front of the bowl, with no spoon or fork."
        ),
        "focus": ["container_count", "material_binding", "front-back", "negation_absence", "color_binding"],
        "expected": {
            "subjects": ["clear bowl", "crimson marble", "yellow leaves", "blue paper fan"],
            "counts": {"marble": 1, "yellow leaves": 3, "paper fan": 1},
            "colors": {"marble": "crimson", "leaves": "yellow", "fan": "blue"},
            "attributes": {"fan": ["paper"]},
            "spatial_relations": [
                {"subject": "marble", "relation": "inside", "object": "bowl"},
                {"subject": "leaves", "relation": "inside", "object": "bowl"},
                {"subject": "fan", "relation": "in_front_of", "object": "bowl"},
            ],
            "negative_constraints": ["no spoon", "no fork"],
        },
    },
    {
        "id": "compact_holdout_scene_005",
        "category": "multi_compositional",
        "prompt": (
            "A pink screen hides the upper half of a black wooden chair, but the "
            "chair legs remain visible; one yellow label on the chair shows the "
            "exact blue text 'OK'."
        ),
        "focus": ["occlusion_visibility", "part_visibility", "material_binding", "exact_text", "color_binding", "count"],
        "expected": {
            "subjects": ["pink screen", "black wooden chair", "chair legs", "yellow label", "blue text"],
            "counts": {"yellow label": 1},
            "colors": {"screen": "pink", "chair": "black", "label": "yellow", "text": "blue"},
            "attributes": {"chair": ["wooden"], "chair legs": ["visible"]},
            "occlusion": [{"occluder": "screen", "target": "chair upper half", "degree": "hidden"}],
            "text": [{"object": "yellow label", "value": "OK", "color": "blue"}],
        },
    },
    {
        "id": "compact_holdout_scene_006",
        "category": "multi_compositional",
        "prompt": (
            "A gray dog touches a turquoise metal bell while holding an orange rubber "
            "ring, and a white pyramid is above the bell; no second ring is visible."
        ),
        "focus": ["two_actions", "material_binding", "above", "count_negation", "color_binding"],
        "expected": {
            "subjects": ["gray dog", "turquoise metal bell", "orange rubber ring", "white pyramid"],
            "counts": {"ring": 1},
            "colors": {"dog": "gray", "bell": "turquoise", "ring": "orange", "pyramid": "white"},
            "attributes": {"bell": ["metal"], "ring": ["rubber"]},
            "interaction_relations": [
                {"subject": "dog", "action": "touches", "object": "bell"},
                {"subject": "dog", "action": "holding", "object": "ring"},
            ],
            "spatial_relations": [
                {"subject": "pyramid", "relation": "above", "object": "bell"}
            ],
            "negative_constraints": ["no second ring"],
        },
    },
)


def build_benchmark_payload(split: str) -> dict[str, Any]:
    if split == "dev":
        cases = _default_compact_cases(DEV_CASES)
    elif split == "holdout":
        cases = _default_compact_cases(HOLDOUT_CASES)
    else:
        raise ValueError(f"unknown split: {split}")
    return {
        "version": f"flux_agent_compact_compositional_{split}_v1",
        "description": (
            "Compact anti-overfit benchmark. Each image carries multiple hard "
            "constraints where useful, but the split also includes single-axis "
            "probes for attribution. Dev and holdout use matched templates with "
            "different concrete objects, colors, materials, relations, text, and "
            "symbols."
        ),
        "split": split,
        "anti_overfit_policy": {
            "source_benchmarks_checked": [
                "hard_prompts_mini.json",
                "hard_prompts_mini_holdout.json",
            ],
            "dev_holdout_templates_matched": True,
            "concrete_prompt_overlap_allowed": False,
            "single_axis_cases_per_main_category": 2,
            "multi_axis_stress_cases": 4,
            "use_for_iteration": "Run dev during coding; run holdout only after a general change.",
            "primary_metric": "hard-constraint pass, then typed failure accuracy",
        },
        "cases": [dict(case) for case in cases],
    }


def _default_compact_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    single_axis = [
        dict(case)
        for case in cases
        if "single_axis" in {str(item) for item in case.get("focus", []) or []}
    ]
    multi_axis = [
        dict(case)
        for case in cases
        if str(case.get("category")) == "multi_compositional"
    ][:4]
    return [*single_axis, *multi_axis]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build compact benchmark JSON files.")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "benchmarks")
    return parser


def write_benchmarks(*, split: str, out_dir: Path) -> list[Path]:
    splits = ["dev", "holdout"] if split == "all" else [split]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in splits:
        payload = build_benchmark_payload(item)
        path = out_dir / f"hard_prompts_compact_{item}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_benchmarks(split=args.split, out_dir=args.out_dir)
    print(json.dumps({"written": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
