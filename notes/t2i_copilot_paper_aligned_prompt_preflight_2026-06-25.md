# T2I-Copilot Paper-Aligned Prompt Preflight - 2026-06-25

Source:

```text
project_flux/benchmarks/t2i_copilot_paper_aligned_5each.json
```

The file contains 105 prompts from 21 categories, 5 prompts per category. It is
not a normal homogeneous hard-prompt set. It mixes easy single-object prompts,
text rendering, exact count, spatial composition, role reversal, misspellings,
rare words, and GenAI-Bench compositional reasoning. GPU generation should be
preceded by prompt preflight; otherwise misspelling/rare-word prompts can be
misrouted as ordinary missing-object failures and waste full FLUX rounds.

## Current Fail-8 Preflight

| Case | Prompt | Prompt Type | Correct Route | Local Edit? | Run Policy |
|---|---|---|---|---|---|
| `drawbench_conflicting_001` | `A horse riding an astronaut.` | role reversal / unusual relation | `role_action_binding_regeneration` or conflict-aware candidate rerank | no | keep as stress case, not first smoke |
| `drawbench_misspellings_001` | `Tcennis rpacket.` | typo / lexical grounding | `lexical_grounding_regeneration` or clarify | no | normalize before GPU; do not treat as generic missing object |
| `drawbench_rare_words_001` | `Acersecomicke.` | rare or unverifiable word | `unverifiable_rare_word_or_clarify` | no | skip or clarify unless definition is known |
| `drawbench_reddit_001` | church windows depicting hamburger and fries | multi-object symbol-in-scene | `multi_constraint_decompose` + rerank | maybe only small localized symbol fixes | stress case |
| `genai_attribute_001` | baker pulling freshly baked bread out of oven | scene/action/attribute | existence + action + scene check | no by default | do not infer exact bread count |
| `genai_comparison_001` | more pens than pencils | comparative count | `comparative_count_rerank` | no | generate candidates and VLM rerank |
| `genai_differentiation_001` | larger yellow person and smaller different-color person | comparative attribute binding | `comparative_attribute_binding` | no by default | candidate rerank |
| `genai_negation_001` | girl with glasses drawing, girl without glasses singing | role/action binding with negation | `role_action_binding_regeneration` | no | do not route to forbidden-object removal |

The first three fail-8 prompts are bad choices for validating typed-action
image generation speed. They are useful for testing lexical/clarification
preflight, but they should not consume repeated FLUX rounds before the route is
known.

## Category Taxonomy

| Category | Main Prompt Type | Expected Backend | Local Edit Policy |
|---|---|---|---|
| `drawbench_colors` | single-object color | attribute check, rerank or small patch | possible if object visible |
| `drawbench_conflicting` | unusual / reversed roles | role-action candidate rerank | no generic edit |
| `drawbench_counting` | exact object count | count-aware candidate rerank | no, except localized extra-object removal with evidence |
| `drawbench_dall_e` | shape/material binding | attribute/material rerank, possible patch | only if target object is visible |
| `drawbench_descriptions` | definition-to-object grounding | semantic/lexical interpreter before GPU | no local edit by default |
| `drawbench_gary_marcus_et_al.` | complex reasoning, occlusion, spatial, negation | decompose then rerank/layout | edit only for localized occluder/removal |
| `drawbench_misspellings` | typo recovery | lexical normalization or clarify before GPU | no |
| `drawbench_positional` | object-on-object spatial relation | layout/spatial candidate rerank | no generic edit |
| `drawbench_rare_words` | rare/unverifiable terms | definition lookup or clarify | no |
| `drawbench_reddit` | multi-object symbolic scenes | multi-constraint decompose and rerank | only localized symbol/text fixes |
| `drawbench_text` | exact text rendering | OCR verify, text overlay, text rerank | text-specific overlay, not PowerPaint |
| `genai_attribute` | scene/action/attribute | existence/action/attribute check | no by default |
| `genai_scene` | scene relation and atmosphere | scene/action candidate rerank | no by default |
| `genai_spatial_relation` | spatial/object relation | layout-guided regeneration + rerank | no generic edit |
| `genai_action_relation` | action/functional relation | action VQA + candidate rerank | local only if contact target is clear |
| `genai_part_relation` | accessory/part binding | object-part VQA and rerank | possible for localized accessory |
| `genai_counting` | count + role/attribute | count-focused rerank | no generic edit |
| `genai_comparison` | comparative count/size | comparative VQA rerank | no |
| `genai_differentiation` | comparative attributes / left-right differences | typed candidate rerank | no generic edit |
| `genai_negation` | absence plus role/action binding | role/action/absence VQA rerank | removal only if forbidden object is localized |
| `genai_universal` | universal quantifier across many objects | broad constraint rerank | no generic edit |

## Preflight Rules To Add Before More GPU Runs

1. Lexical prompts first:

```text
misspellings -> normalized variant + literal-preserving variant
rare/nonsense word -> mark clarify/unverifiable unless a definition is known
```

These prompts should not enter the ordinary `missing_object -> regenerate`
loop.

2. Do not create hidden exact-count constraints from mass or scene nouns.

Example:

```text
freshly baked bread
```

means bread should exist and be freshly baked; it does not require exactly one
bread object.

3. Comparative prompts need typed routes:

```text
more pens than pencils -> comparative_count_rerank
larger person / smaller person -> comparative_attribute_binding
```

They should not become generic `wrong_count`, `missing_object`, or `relation`
failures.

4. Role/action negation is not forbidden-object removal.

Example:

```text
girl with glasses drawing / girl without glasses singing
```

The glasses are required for one subject and absent for another. The repair
route should preserve both roles and actions, not remove glasses from the image
globally.

5. Local editing remains narrow.

Use PowerPaint/SAM only for localized visible-object edits, localized forbidden
object/symbol removal, or occlusion-object insertion. Do not use it as the
default backend for count, spatial chains, comparative reasoning, broad
multi-constraint scenes, misspellings, or rare words.

## Better Experiment Split

For P5.5 typed-action validation, split the 105 prompts into three subsets:

```text
lexical_preflight_only:
  drawbench_misspellings, drawbench_rare_words

typed_action_fast_gpu:
  count, comparison, differentiation, negation, attribute, spatial, text

stress_gpu:
  conflicting, gary_marcus, reddit, universal, broad scene prompts
```

The current fail-8 begins with `conflicting`, `misspellings`, and `rare_words`,
which is useful for revealing preflight bugs but inefficient for measuring
typed-action image backend quality.

## Implemented Guardrail

Implemented on 2026-06-25:

```text
project_flux/src/repair_planner.py
project_flux/src/orchestrator.py
project_flux/scripts/run_mini_benchmark.py
```

Behavior:

- short misspelling prompts are preflight-routed before ordinary missing-object
  repair;
- normalizable DrawBench misspellings get `lexical_grounding_regeneration` and
  a `normalized_prompt`;
- rare or visually undefined one-word prompts get
  `unverifiable_rare_word_or_clarify`;
- lexical/rare prompts are not protected by the empty-question hard-pass guard;
- `unverifiable_rare_word_or_clarify` stops with `needs_clarification` after a
  full generate-observe-plan cycle instead of spending another FLUX round;
- benchmark summaries now include `needs_clarification_count`.

Validation:

```text
project_flux tests: 395 passed
```

For the full 105-case benchmark, judge the agent by these metrics together:

```text
completion_passed
route_none_count
typed_action_attempts / typed_action_accepted
typed_route_counts by category
needs_clarification_count for rare/unverifiable prompts
non_edit_route_skipped_powerpaint_count
false_pass_blocked_count
infrastructure_failures
```

`needs_clarification` is not an image success. It is a valid agent decision for
rare/unverifiable prompts when no visual definition is available.
