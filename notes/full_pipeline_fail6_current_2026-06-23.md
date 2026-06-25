# Full Pipeline Fail6 Run - 2026-06-23

## Goal
Run the six recent hard failures from prompt generation onward:

prompt -> FLUX/M-GRAG generation -> VLM evaluation -> repair planner -> optional repair/edit route -> final VLM summary.

This is not the edit-only experiment. It validates whether the integrated agent actually routes failures into the right repair tools.

## Command
Run directory:

`runs_mini_benchmark/real-flux-full-pipeline-fail6-current-512-30-gpu0-gpu1-net`

Benchmark:

`benchmarks/hard_prompts_fail6_full_pipeline.json`

Main settings:

- FLUX/M-GRAG on GPU0.
- Grounded-SAM2 and PowerPaint repair subprocesses configured for GPU1.
- 512x512, 30 steps.
- M-GRAG intervention: first 20 steps, bias 1.0, delta 1.3.
- VLM: DashScope OpenAI-compatible `qwen-vl-plus`.
- `max_rounds=2`, `n_images=1`.
- Efficient repair agent, editing mask agent, object insertion repair, and PowerPaint relation editor enabled.

## Outputs
- Summary: `runs_mini_benchmark/real-flux-full-pipeline-fail6-current-512-30-gpu0-gpu1-net/summary.md`
- Visual grid: `reports/full_pipeline_fail6_current_grid.jpg`

## Aggregate Result
- Total: 6
- Evaluable: 6
- Infrastructure failures: 0
- Completion passed: 1 / 6

The one passed case was:

- `compact_dev_single_spatial_001`: yellow pyramid right of red cylinder.

The five failed cases were:

- `holdout_spatial_001`: three-object spatial chain failed.
- `holdout_color_002`: multi-object color/material binding failed.
- `compact_dev_single_occlusion_002`: red occluding screen missing.
- `compact_dev_scene_001`: count + layout + under relation failed.
- `compact_dev_scene_003`: black sign missing; prompt repair drifted into blue sign.

## Repair Routing Observations

No `efficient_repair_round_N.json` was produced in this integrated run.

The occlusion case produced `object_repair_round_1.json`, but it was rejected:

`object insertion requires layout_context unless using typed occlusion repair`

This is the key integration gap. The editing agent can run in isolation, but the full pipeline does not yet provide a reliable typed region/layout context for a missing occluder.

## Case Notes

### `holdout_spatial_001`
Final score: 0.377.

The image still stacks pyramid/cylinder/cubes instead of satisfying:

- yellow pyramid right of red cylinder
- red cylinder above blue cube
- only one cube

Repair planner selected `regenerate` twice. This is reasonable for generic PowerPaint avoidance, but it needs layout-guided regeneration rather than plain prompt rewriting.

### `holdout_color_002`
Final score: 0.65.

The scene is close visually but still fails:

- chair duplicated
- silver fan not on rug / sometimes treated as missing
- material/color binding remains brittle

Repair planner selected `regenerate` twice. This category should use either multi-candidate rerank or per-object attribute repair when the target object is visible and isolated.

### `compact_dev_single_spatial_001`
Final score: 1.0.

Passed on the first generated image. No repair needed.

This confirms FLUX/M-GRAG can solve simple two-object left/right layout without editing.

### `compact_dev_single_occlusion_002`
Final score: 0.436.

The generated image has a green suitcase on a red background, but no distinct red screen hiding the lower half.

Round 1 repair plan selected `object_insertion` for target `screen`, but object insertion was rejected because there was no layout context. This should become a typed occlusion repair:

- infer suitcase bbox from VLM/object state
- create lower-half foreground bbox/shape mask
- insert a flat red occluding screen while preserving handle area

### `compact_dev_scene_001`
Final score: 0.578.

The base image is visually clean but fails compositional constraints:

- two spoons instead of one
- spoon not under tray
- mugs not clearly left of tray
- cyan vs teal issue

Plain prompt regeneration did not fix the mixed count/layout relation. This category should route to layout-guided regeneration first, with count-aware rerank over multiple candidates.

### `compact_dev_scene_003`
Final score: 0.2.

The selected image has correct `NO` text, blue signs, and pink ball, but the required black sign is missing.

The revised prompt drifted into:

`A blue sign displays the exact yellow text 'NO' ... black blue sign, blue blue sign ...`

This is a general prompt-optimizer failure: VLM-observed wrong colors leaked back into the rewritten prompt. The arbiter must preserve original hard constraints and forbid substituting observed wrong attributes for required attributes.

## Required Next Fixes

1. Add typed occlusion repair route.
   - Convert `screen hides lower half of suitcase` into a bbox/shape-mask insertion task.
   - Use existing object bbox/part bbox from VLM object state.
   - Do not require generic layout context for this special case.

2. Harden prompt optimizer against observation drift.
   - Original hard constraints must override image observations.
   - Observed wrong attributes should be stored as forbidden confusions, not inserted into the revised prompt.
   - Example: if prompt says `black sign` and VLM observes only blue signs, revised prompt must emphasize `black upper sign`, not `blue sign displays NO`.

3. Add layout-guided regeneration for multi-object spatial failures.
   - Use structured layout hints for left/right/above/under chains.
   - Route spatial chains and mixed count/layout scenes away from local PowerPaint.

4. Use multiple candidates for color/material binding and mixed scenes.
   - `n_images=1` is too brittle for these hard prompts.
   - Efficient setting: keep single image for simple categories, use 2-4 candidates for known unstable categories.

5. Text/symbol repair should stay deterministic/OCR-gated.
   - Generic PowerPaint is not suitable for exact text.
   - Text overlay only helps if the base sign geometry/color is already correct enough.

## Rerun After Completion-Gate Fix

Run directory:

`runs_mini_benchmark/real-flux-fail6-fullpipeline-after-gatefix-512-30-gpu0`

Visual grid:

`reports/contact_sheet_fail6_after_gatefix_fullpipeline.jpg`

Code fix validated before rerun:

- Tightened the hard-pass guard so narrow question-level VQA cannot erase evaluator hard failures.
- Added semantic evaluator-error typing so evidence like `image contains two vases` is treated as `wrong_count` even when the VLM judge labels it `wrong_attribute`.
- Test suite passed: `191 passed`.

Aggregate after the fix:

- Total: 6
- Evaluable: 6
- Infrastructure failures: 0
- Completion passed: 1 / 6
- Passed: `compact_dev_single_spatial_001`
- Failed: `holdout_spatial_001`, `holdout_color_002`, `compact_dev_single_occlusion_002`, `compact_dev_scene_001`, `compact_dev_scene_003`

Important difference from earlier runs:

- `holdout_color_002` is no longer incorrectly completed when hard VQA/evaluator disagree. The summary now reports `hard VQA=False` and the run ends with `max_rounds_reached`, which is the correct conservative behavior.
- No `efficient_repair_round_N.json` or `object_repair_round_N.json` was produced in this rerun. The integrated planner routed all failed cases to `regenerate`, including the occlusion case.

Case-level rerun notes:

- `holdout_spatial_001`: still fails spatial chain; prompt repair produces stacked objects, not the requested right/above relations.
- `holdout_color_002`: visually close, but hard VQA rejects it. The agent still needs candidate rerank or per-object visible-target repair, not one more generic prompt rewrite.
- `compact_dev_single_spatial_001`: passes first round.
- `compact_dev_single_occlusion_002`: still fails because the red screen is missing. This should route to typed `shape_overlay` or typed occluder insertion, but currently stays on `regenerate`.
- `compact_dev_scene_001`: still fails mixed count/layout/under relation. Needs layout-guided regeneration and count-aware candidate selection.
- `compact_dev_scene_003`: visually close and exact text is good, but the hard checker/evaluator still rejects sign/count/spatial details. This category needs OCR/text-specific verification plus better sign-object parsing.

Next implementation priority:

1. Make missing primitive occluders route to typed `shape_overlay` before generic regenerate.
2. For complex spatial/mixed scenes, enable layout-guided regeneration instead of plain prompt rewrite.
3. For color/material binding, use 2-4 candidates with hard VQA rerank before expensive editing.
4. For text/sign cases, add OCR/text-aware verification and avoid treating required two-sign prompts as one-sign count failures.
