# All-Categories 10-Each Analysis And Fixes

Date: 2026-06-23

## Run

Benchmark:

`benchmarks/hard_prompts_all_categories_10each.json`

Run directory:

`runs_mini_benchmark/real-flux-all-categories-10each-512-30-gpu0`

Command family:

FLUX + M-GRAG, 512x512, 30 steps, max_rounds=2, n_images=1, VLM/API evaluator.

Important caveat:

The original run did not enable the efficient editing agent, object insertion repair, or PowerPaint/SAM2 editing flags. The run measures generation plus prompt rewrite, not local editing effectiveness.

## Results

Overall:

- 90 total cases
- 47 completed
- 43 max_rounds_reached
- 0 infrastructure failures

By category:

- attribute_binding: 8/10
- negation_absence: 8/10, but several were false passes before the hard-pass fix
- interaction_relation: 7/10
- occlusion_visibility: 6/10
- spatial_layout: 6/10
- text_symbol: 6/10
- color_binding: 2/10
- count_quantity: 2/10
- multi_compositional: 2/10

## Main Findings

1. Count remains weak.

Typical errors include extra/missing instances and failure to preserve exact counts after prompt rewrite.

2. Complex color/material binding remains weak.

Three-object rare color/material prompts often fail the narrow VQA check even when broad evaluator scores are high.

3. Multi-compositional prompts expose compounded failures.

Count + layout + text/negative/occlusion constraints fail much more often than single-axis prompts.

4. Editing was not exercised in the 90-case run.

No `efficient_repair_round_*.json`, `object_repair_round_*.json`, or `relation_repair_round_*.json` files were produced.

5. Hard-pass guard still had a false-pass gap for negation.

Example:

`all10_negation_006`: prompt requires no bowl and no spoon, evaluator found bowl/spoon, but question-level hard VQA passed and the hard-pass guard completed the run.

## Fixes Implemented

1. Added semantic hard error type:

`forbidden_object_present`

This catches evaluator evidence such as:

- contains forbidden object
- visible object violates "no ..."
- present object violates "without ..."

2. Updated hard-pass blocking.

The hard-pass guard no longer overrides evaluator failures of type:

- `missing_object`
- `wrong_count`
- `wrong_symbol_text`
- `forbidden_object_present`
- repair-requiring `wrong_relation`

3. Updated completion hard-user errors.

`forbidden_object_present` now blocks completion like other user-grounded hard failures.

4. Added benchmark-level auto editing flags.

New `scripts/run_mini_benchmark.py` flag:

`--auto-efficient-repair-for-categories`

Default editable categories:

- `occlusion_visibility`
- `occlusion`
- `text_symbol`
- `text_symbol_layout`
- `negation_absence`
- `multi_compositional`

The flag automatically appends the efficient repair / PowerPaint / Grounded-SAM2 parameters for these categories only. It intentionally does not auto-enable editing for `count_quantity`, because exact count errors are usually better handled by regeneration/reranking/layout, not local inpaint.

## Validation

Test command:

`python -m pytest tests/test_benchmarks.py tests/test_orchestrator.py tests/test_repair_planner.py tests/test_run_m4_cli.py tests/test_editing_agent.py tests/test_local_editor.py`

Result:

- 174 passed

Full related suite after hard-pass fix:

- 194 passed

Dry-run checks:

- `occlusion_visibility` now automatically receives efficient repair, editing mask, PowerPaint, and Grounded-SAM2 flags.
- `count_quantity` does not receive editing flags.

Offline recheck on previous hard-pass/evaluator disagreements:

The new guard blocks 4 prior false-pass negation cases:

- `all10_negation_004`
- `all10_negation_006`
- `all10_negation_009`
- `all10_negation_010`

## Next Experiment

Rerun an editable subset with:

`--auto-efficient-repair-for-categories`

Recommended categories:

- `occlusion_visibility`
- `text_symbol`
- `negation_absence`
- `multi_compositional`

This checks whether the repair agent actually triggers and improves the cases that can plausibly be fixed by local editing.

## Follow-up Other-Category Probe

The 12-case auto-edit check only covers:

- `negation_absence`
- `occlusion_visibility`
- `text_symbol`

Before making broad planner changes, run a second probe over the remaining categories:

- `count_quantity`
- `color_binding`
- `spatial_layout`
- `attribute_binding`
- `interaction_relation`
- `multi_compositional`

Recommended 24-case subset:

- Count: `all10_count_001`, `all10_count_004`, `all10_count_006`, `all10_count_009`
- Color/material: `all10_color_001`, `all10_color_002`, `all10_color_005`, `all10_color_009`
- Spatial: `all10_spatial_001`, `all10_spatial_005`, `all10_spatial_006`, `all10_spatial_010`
- Attribute: `all10_attribute_003`, `all10_attribute_005`, `all10_attribute_007`, `all10_attribute_010`
- Interaction: `all10_interaction_004`, `all10_interaction_005`, `all10_interaction_009`, `all10_interaction_010`
- Multi: `all10_multi_002`, `all10_multi_003`, `all10_multi_005`, `all10_multi_008`

Purpose:

- Confirm count failures should remain regeneration/rerank/layout-guided rather than local editing.
- Confirm spatial failures need layout-guided regeneration or object-state checks.
- Confirm color/material failures should become per-object typed patch or multi-candidate rerank, not generic full-prompt rewrite.
- Confirm interaction failures need relation repair only when the planner can localize a contact point.
- Confirm multi-compositional failures should be decomposed into typed subfailures before choosing edit vs regenerate.

Unified planner changes should be failure-type based, not prompt-specific:

- `wrong_count` -> count-aware regeneration/rerank; avoid local edit unless only one removable extra object is localized.
- `wrong_spatial_relation` -> layout-guided regeneration for global layout; local edit only for simple overlay/removal.
- `wrong_color_material_binding` -> per-object attribute patch + rerank; local edit only when one existing object has a localized region.
- `wrong_interaction_relation` -> relation/contact repair when bbox/contact target exists; otherwise regenerate with relation emphasis.
- `forbidden_object_present` -> object/symbol/text removal route when localized; otherwise regenerate.
- `wrong_exact_text` -> text overlay or OCR-aware regeneration.

## Combined 36-Case Probe Result

Runs:

- `runs_mini_benchmark/real-flux-autoedit-check12-512-30-gpu0`
- `runs_mini_benchmark/real-flux-othercats-probe24-512-30-gpu0`

Combined result:

- 36 total cases
- 12 completed
- 24 failed
- 0 infrastructure failures

By category:

- `count_quantity`: 3/4
- `color_binding`: 2/4
- `occlusion_visibility`: 2/4
- `attribute_binding`: 1/4
- `multi_compositional`: 1/4
- `negation_absence`: 1/4
- `spatial_layout`: 1/4
- `text_symbol`: 1/4
- `interaction_relation`: 0/4

Key signal:

The auto-edit path is wired, but it is rarely reached:

- Repair plans: 46 `regenerate`, 3 `object_insertion`, 1 `none`
- Typed routes: 46 `none`, 2 `missing_required_object`, 1 `occlusion_object_insertion`, 1 `layout_guided_regeneration`
- Efficient edit attempts: 2 total
  - `shape_overlay`, accepted=false
  - `text_overlay`, accepted=true

This means the next bottleneck is not GPU generation or PowerPaint availability. The bottleneck is repair planner routing: many editable failures are still returned as generic `regenerate`.

Representative planner misses:

- `all10_negation_006`: forbidden bowl/spoon present -> `regenerate`, should route to localized forbidden-object removal/erase if region can be located.
- `all10_text_002`: forbidden star on green box + duplicate star on red box -> `regenerate`, should route to symbol removal plus count-aware regeneration if multiple duplicates are present.
- `all10_color_001`: silver paper fan/color binding failure -> `regenerate`, should route to per-object attribute patch or candidate rerank, not whole prompt rewrite only.
- `all10_interaction_009`: missing silver hook/contact relation -> `regenerate`, should route to relation/contact repair if hook/loop region is localizable; otherwise relation-focused regeneration.
- `all10_spatial_001`: wrong count plus wrong relations -> `regenerate` is reasonable, but should be layout-guided rather than plain regeneration.

Updated implementation priority:

1. Add typed planner routes for forbidden-object/symbol/text presence.
2. Add wrong-exact-text and wrong-symbol routes to text/symbol overlay when the object count is otherwise acceptable.
3. Add per-object color/material attribute patch route; keep broad color-binding failures on rerank/regenerate when multiple objects are wrong.
4. Add interaction/contact route only when a specific contact target can be localized.
5. Add layout-guided regeneration for spatial/count compound failures instead of plain regenerate.

## Typed Router Implementation Update

Date: 2026-06-23

Implemented after the 90-case and 36-case evidence above:

- `repair_planner.py`
  - Added a unified failure-to-route normalization layer.
  - Local editable routes:
    - `forbidden_object_removal`
    - `forbidden_symbol_removal`
    - `exact_text_overlay`
    - `single_attribute_patch`
    - `relation_contact_repair`
  - Non-local routes:
    - `count_aware_regeneration`
    - `layout_guided_regeneration`
    - `relation_focused_regeneration`
    - `multi_constraint_decompose`
  - Fixed two routing bugs found by tests/probes:
    - `No red screen is visible` no longer becomes forbidden-object removal;
      it stays typed occlusion insertion.
    - `red instead of blue` no longer gets misclassified as a count failure.
- `editing_agent.py`
  - Maps the new typed routes to efficient repair kinds:
    - exact text -> `text_overlay`
    - forbidden symbol -> `symbol_overlay`
    - forbidden object / single attribute / contact repair -> `existing_object_inpaint`
    - count -> `count_rerank`
    - spatial / relation-focused / broad multi -> `layout_regenerate`
- `scripts/run_mini_benchmark.py`
  - Default seed policy is now stable by `case_id`, so the same case keeps the
    same seed across 90-case, 36-case, and subset reruns.
  - Summary now reports:
    - `typed_route_counts`
    - `route_none_count`
    - `efficient_edit_attempts`
    - `accepted_edit_count`
    - `false_pass_blocked_count`

Validation:

```text
64 passed:
tests/test_repair_planner.py
tests/test_editing_agent.py
tests/test_benchmarks.py

156 passed:
tests/test_orchestrator.py
tests/test_run_m4_cli.py
tests/test_local_editor.py
tests/test_prompt_constraints.py
tests/test_visual_reflector.py
```

Next experiment:

Rerun the same 36-case probe with fixed case-id seeds and `max_rounds=2`.
Do not judge success only by strict pass rate. The route metrics must move:

- `typed_route=None` should drop sharply from the old 46 occurrences.
- Editable local categories should show more efficient repair attempts, but
  count/spatial/multi should not spam PowerPaint.
- Accepted edits must pass post-repair original-constraint checks.
- False-pass blocked count must not regress.

If route accuracy improves but strict pass does not, the next bottleneck is
backend quality for the selected routes, not planner classification.

## Typed Router36 Partial R2 Finding And Fix

Date: 2026-06-24

Run:

```text
runs_mini_benchmark/real-flux-typed-router36-512-30-gpu0-r2
```

Important caveat:

```text
Only 24/36 cases completed. The run stopped after generating the first image
for all10_occlusion_001, so the later occlusion/text/multi cases were not
evaluated in this directory.
```

Partial result on the 24 completed cases:

```text
completion_passed: 6/24
hard_passed:       12/24
eval_passed:       8/24
route_none_count:  5
efficient edits:   0
```

Diagnosis:

- The typed router is doing its job: `route_none_count` dropped substantially
  and most failures now have explicit typed routes such as
  `count_aware_regeneration`, `layout_guided_regeneration`,
  `material_guided_regeneration`, `forbidden_object_removal`, and
  `relation_contact_repair`.
- The local edit tool path was still not connected:
  - `relation_contact_repair` was selected, but the benchmark auto-edit flags
    did not add `--enable-relation-repair`, so the route fell back to
    regeneration.
  - `forbidden_object_removal` was selected, but efficient inpaint skipped
    because the repair plan had no localized bbox.
  - Some forbidden targets were dirty strings such as
    `no visible zipper pull`; localization should target `zipper pull`.

Implemented fix after this partial run:

- `scripts/run_mini_benchmark.py`
  - `--auto-efficient-repair-for-categories` now also enables:
    - `--enable-relation-repair`
    - `--enable-vlm-target-locator`
- `orchestrator.py`
  - Efficient local inpaint/shape routes now try VLM target localization before
    the bbox gate when a typed editable route lacks `bbox`.
  - Forbidden localization target cleaning strips negative words, e.g.
    `no visible zipper pull` -> `zipper pull`.
  - Writes `efficient_repair_target_locator_round_<n>.json` for bbox evidence.

Validation:

```text
141 passed:
tests/test_benchmarks.py
tests/test_orchestrator.py
tests/test_editing_agent.py
tests/test_repair_planner.py
```

Next retest:

Run only the 12 cases missing from r2 plus the known tool-connection cases:

```text
all10_occlusion_001
all10_occlusion_002
all10_occlusion_003
all10_occlusion_008
all10_text_002
all10_text_005
all10_text_006
all10_text_010
all10_multi_002
all10_multi_003
all10_multi_005
all10_multi_008
all10_negation_004
all10_interaction_005
```

Success signal is not just pass rate. The retest must show nonzero
`efficient_edit_attempts` for editable local routes and explicit target-locator
artifacts when forbidden/removal routes lack an initial bbox.
