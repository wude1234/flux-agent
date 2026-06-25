# Decision-Only Batch Architecture

Date: 2026-06-25

## Why This Exists

The project had two coupled problems:

- full benchmark commands were too slow because every case started a new
  `run_m4` process and therefore a new FLUX subprocess/model load;
- the agent evaluation path could silently execute typed-action candidate
  generation or editing, so "test whether the agent chose the right repair
  direction" became expensive image generation.

## Decision

Use a cheap decision-first stage before local editing or typed action backend:

```text
micro-batched FLUX/M-GRAG round-0 images
  -> per-case OrchestratorAgent with pregenerated image
  -> VLM constraint/evaluator
  -> specialist reports
  -> repair_plan_round_0.json
  -> summary route/action counts
```

This tests whether the agent can choose the correct repair direction without
running PowerPaint/SAM2 or generating extra typed-action candidates.

## 2026-06-25 Reality Check

`--batch-decision` is currently experimental, not the default recommended
runner.

Real run:

```text
runs_mini_benchmark/batch-decision-paper5each-512-30-gpu0-r2
```

Observed behavior:

```text
first FLUX micro-batch stayed at 0 generated images
GPU0 memory was allocated but GPU utilization stayed 0%
no chunk stdout/stderr was written because subprocess.run never returned
process state was observed as D / I/O wait during FLUX loading
```

Conclusion:

```text
the batch diffusers FLUX path can hang during model/cache loading
do not use --batch-decision for overnight/full benchmark yet
use normal --decision-only until the batch backend is replaced or proven safe
```

## Important Runtime Policy

Do not make CPU-offloaded FLUX a default long-lived service.

Reason:

- FLUX-dev with CPU offload can keep large model state in CPU memory/swap;
- a long-lived worker may survive the first image and then get killed around
  later images under memory pressure;
- a full 90-prompt single-process batch has the same risk.

The intended safer compromise was micro-batching:

```text
--batch-decision
--batch-decision-chunk-size 5
```

This would reduce FLUX cold starts from one per case to one per small chunk,
while letting each FLUX process exit and release GPU/CPU memory.  However, the
current implementation uses the diffusers `infer_mgrag_flux.py` loading path,
which hung in the real r2 run above.  Treat it as a backend prototype only.

## Implemented Changes

- `src/run_m4.py`
  - added `--decision-only`;
  - decision-only forces `max_rounds=1`;
  - decision-only disables typed action backend, local repair, relation repair,
    object insertion, efficient repair, mask refiner, SAM, and editors;
  - typed action backend no longer auto-enables just because evaluator and
    `max_rounds > 1` are set. It must be enabled explicitly.

- `scripts/run_mini_benchmark.py`
  - added `--decision-only`;
  - added `--batch-decision`;
  - added `--batch-decision-chunk-size`;
  - added `--batch-generation-timeout-seconds`;
  - batch-decision generates micro-batched FLUX/M-GRAG images once per chunk,
    then runs the existing orchestrator with `MockImageGenerator(existing_paths=...)`;
  - decision-only command construction strips edit/repair arguments so command
    logs do not misleadingly show PowerPaint/SAM2 flags.

- `infer_mgrag_flux.py`
  - added repeated `--prompt_seed` so batch generation can preserve the same
    fixed seed policy as single-case benchmark runs.

- `src/typed_action_backend.py`
  - extracted typed-action route policy and prompt variant generation out of
    `orchestrator.py`.

## How To Use

Decision-only normal path, one FLUX load per case:

```bash
python scripts/run_mini_benchmark.py \
  --benchmark benchmarks/your_cases.json \
  --runs-dir runs_mini_benchmark/decision_only \
  --run-prefix decision-only \
  --generator flux \
  --llm mock \
  --vlm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --vlm-model qwen-vl-plus \
  --cuda-visible-devices 0 \
  --width 512 \
  --height 512 \
  --steps 30 \
  --decision-only \
  --max-rounds 2
```

Experimental faster decision-only path, micro-batched FLUX:

```bash
python scripts/run_mini_benchmark.py \
  --benchmark benchmarks/your_cases.json \
  --runs-dir runs_mini_benchmark/batch_decision \
  --run-prefix batch-decision \
  --generator flux \
  --llm mock \
  --vlm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --vlm-model qwen-vl-plus \
  --cuda-visible-devices 0 \
  --width 512 \
  --height 512 \
  --steps 30 \
  --batch-decision \
  --batch-decision-chunk-size 5
```

Read the result by category and route:

```text
summary.json
summary.md
repair_plan_round_0.json in each case run directory
```

## Validation

```text
python -m pytest -q
```

Result:

```text
401 passed, 152 warnings
```

The tests validate command construction and mock/dry-run behavior.  They do not
prove that the diffusers FLUX batch backend is safe on this machine.

## Next Work

After decision-only confirms route quality, enable expensive action backends
explicitly on selected failure routes:

```text
--enable-typed-action-backend
--enable-efficient-repair-agent
--auto-efficient-repair-for-categories
```

Do not use editing as the default benchmark mode until the repair-plan route
distribution is correct.
