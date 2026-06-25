# Fail6 Editing Agent Experiment - 2026-06-23

## Goal
Validate the automatic editing-agent path on the six recent failed FLUX/M-GRAG cases:

Grounded-SAM2 text mask -> dilated mask -> PowerPaint subprocess edit.

The edit commands were intentionally written as local repair instructions an editing agent would receive, not as full prompt regeneration.

## Implementation Updates
- `scripts/run_grounded_sam2_mask.py`
  - Added lightweight Florence-2 + SAM2 backend that does not import T2I-Copilot's visualization-heavy `test_REF.py`.
  - Uses local Florence cache via `HF_HOME=/mnt/ssd1/powerpaint_envs/hf-cache`.
  - Uses `attn_implementation="eager"` and `use_cache=False` to avoid Florence-2 remote-code incompatibilities with newer Transformers.
  - Supports nested/multi-object Florence polygons and unions them before SAM2.
- `src/editing_agent.py`
  - Default Grounded-SAM2 python changed to `/mnt/ssd1/conda/envs/tweediemix/bin/python`.
  - PowerPaint remains isolated in `/mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python`.
- `scripts/run_fail6_editing_agent_batch.py`
  - Batch runner for the six failed prompts, with per-case `editing_instruction.json`.
- `scripts/build_fail6_editing_final_grid.py`
  - Builds the final visual grid.

## Environment Finding
- PowerPaint env works for PowerPaint but has `transformers==4.28.0`, too old for Florence-2.
- `sam2` env has hydra/omegaconf but lacks transformers/cv2.
- `tweediemix` env has CUDA torch, cv2, supervision, hydra/omegaconf, and a new enough Transformers stack, so it is currently the best Grounded-SAM2 subprocess env.

## Run Outputs
- Main 6-case batch:
  - `runs_edit_smoke/fail6_editing_agent_real_sam_g0_6cases`
- Re-run for fixed multi-polygon color case:
  - `runs_edit_smoke/fail6_editing_agent_real_sam_g0_color002_polyfix2`
- Final grid:
  - `reports/fail6_editing_agent_final_grid.jpg`

## Case-Level Result
| Case | Mask | Visual edit result | Takeaway |
|---|---|---|---|
| `holdout_spatial_001` | Grounded-SAM2 | Edited only the red cylinder/base area, did not solve full pyramid/cylinder/cube relation. | Layout relation failures should route to layout-guided regeneration, not local edit. |
| `holdout_color_002` | Grounded-SAM2 after multi-polygon fix | Local region edited, but still does not reliably enforce turquoise chair + crimson lamp + silver fan + black rug. | Multi-object color/material binding needs per-object sub-edits or regeneration, not one broad edit. |
| `compact_dev_single_spatial_001` | Grounded-SAM2 | Replaced cylinder with a white pyramid-like object; relation still wrong. | Direction swaps are poor PowerPaint targets unless using object-level cut/move/composite. |
| `compact_dev_single_occlusion_002` | Grounded-SAM2 | Added a label/paper-like object on suitcase instead of a red screen occluder. | Additive occluder edit needs bbox/shape-guided mask for the new screen, not mask of the existing suitcase. |
| `compact_dev_scene_001` | Grounded-SAM2 | Targeted spoon area and changed it, but did not fix mugs-left-of-tray relation. | Mixed count/spatial edits need typed sub-repairs or regeneration. |
| `compact_dev_scene_003` | Grounded-SAM2 | Made upper sign black/yellow-ish but corrupted exact text (`NO` became wrong/extra text). | Sign color is editable; exact text should use text-specific regeneration/OCR edit, not generic PowerPaint. |

## Agent Policy Update
Do not trigger PowerPaint simply because a VLM check failed.

Preferred routing:
- Existing object attribute/color: use Grounded-SAM2 on the exact object, then edit/recolor.
- Add new occluder/object: use bbox or shape mask for the new object, not SAM mask of the existing object.
- Multi-object binding: split into per-object edit plans or regenerate.
- Spatial direction/layout: layout-guided regeneration first; local edit only for tiny object replacement.
- Exact text/symbol: use text-specific generation/edit path with OCR verification; generic PowerPaint is not reliable.

## Tests
- `25 passed`:
  - `tests/test_grounded_sam2_mask_script.py`
  - `tests/test_editing_agent.py`
  - `tests/test_run_m4_cli.py`
