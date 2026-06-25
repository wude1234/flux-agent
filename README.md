# FLUX Multimodal T2I Agent Project

This directory is the FLUX-backed implementation area for the fused agent project.

Parent directory roles:

- `../code/`: copied upstream repositories; treat as read-only references.
- `../papers/`: verified PDFs for selected fusion papers.
- `../notes/`: planning, milestones, agentization, and iteration guidance.
- `./`: new fused implementation.

Expected layout:

```text
project/
  src/
  tests/
  runs/
```

Implementation rules:

- Put project code under `project/src/`.
- Put tests under `project/tests/`.
- Put run artifacts under `project/runs/`.
- Keep upstream repos read-only unless explicitly experimenting with them.
- Keep generation, LLM, VLM, reward, and ComfyUI backends behind adapters.

## M0 Environment And Smoke Test

M0 is mock-only. It does not need SDXL, GPU, diffusers, an LLM API, or a VLM API.

Create and activate the project M0 environment:

```bash
conda env create -f environment-m0.yml
conda activate mult-t2i-agent-m0
```

Run the M0 unit tests from the repository root:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent
python -m pytest project/tests/test_state.py project/tests/test_memory.py
```

Run the mock agent smoke test from `project/`:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
python -m src.run_agent --prompt "a red car on a rainy street" --mock
```

The smoke test writes a timestamped JSON log under `project/runs/`.

## M4 Real Smoke Tests

The project now has a separate real-run environment:

```bash
conda activate mult-t2i-agent-real
```

This environment is cloned from the existing `omnigen2` stack and contains
CUDA-enabled PyTorch, diffusers, transformers, accelerate, safetensors, Pillow,
and pytest. On this server, GPU access may require running outside restricted
sandboxed shells; verify with:

```bash
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Run all project tests in the real environment:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m pytest project/tests
```

Run M4 with real local SDXL image generation and mock LLM/VLM feedback:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m4 \
  --prompt "a small red robot holding a blue umbrella, cinematic photo, rainy street" \
  --generator sdxl \
  --llm mock \
  --vlm mock \
  --disable-clarifier \
  --max-rounds 1 \
  --n-images 1 \
  --steps 12 \
  --width 768 \
  --height 768 \
  --seed 123
```

The local SDXL model path defaults to:

```text
/mnt/ssd3/xc/pretrained_models/stable-diffusion-xl-base-1.0
```

### FLUX.1-dev generator

The M4 runner can also use the local Black Forest Labs FLUX checkout and
checkpoints, matching the T2I-Copilot paper's FLUX backbone choice:

```bash
cd /home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project
/home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project/.conda-m0/bin/python -m src.run_m4 \
  --prompt "a red cube on a white table" \
  --generator flux \
  --llm mock \
  --vlm mock \
  --disable-clarifier \
  --max-rounds 1 \
  --n-images 1 \
  --steps 1 \
  --width 256 \
  --height 256 \
  --seed 123
```

Defaults on this machine:

```text
FLUX repo: /home/zrr/flux
FLUX python: /mnt/ssd1/conda/envs/flux-dev/bin/python
FLUX model: /home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/flux1-dev.safetensors
FLUX AE: /home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/ae.safetensors
HF cache: /mnt/ssd3/zrr/hf_cache
```

The adapter forces Hugging Face offline mode by default and reuses the local
T5/CLIP/NSFW cache. Use `--flux-online` only if you intentionally want missing
assets to be downloaded.

The run writes `config.json`, `state_round_0.json`, `run.json`,
`final_report.md`, and PNG images under `project/runs/<run-id>/`.

M4.2 enables a lightweight binding mitigation layer by default in `run_m4`:
the VLM performs a dedicated user-constraint check, failed color/action/relation
constraints are merged into the round feedback, the next prompt is rewritten
with stronger binding language, and an automatic negative prompt is passed to
SDXL. This does not guarantee SDXL will obey difficult bindings, but it makes
the failure visible in logs and gives the retry round a better prompt-space
signal. For attribute-binding cases, prefer at least two candidates:

```bash
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m4 \
  --prompt "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo" \
  --generator sdxl \
  --llm api \
  --vlm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --vlm-model qwen-vl-plus \
  --disable-clarifier \
  --max-rounds 2 \
  --n-images 2 \
  --steps 20 \
  --width 768 \
  --height 768 \
  --seed 124
```

Optional controls:

```text
--disable-constraint-check
--disable-auto-negative-prompt
--negative-prompt "extra negatives here"
```

To use DashScope/Bailian OpenAI-compatible APIs for LLM/VLM feedback, set the
key only in the shell environment:

```bash
export DASHSCOPE_API_KEY="YOUR_KEY_HERE"
```

Recommended first API models:

```text
LLM: qwen-plus
VLM: qwen-vl-plus
VLM higher-quality option: qwen-vl-max
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
```

Then run:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m4 \
  --prompt "a small red robot holding a blue umbrella, cinematic photo, rainy street" \
  --generator sdxl \
  --llm api \
  --vlm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --vlm-model qwen-vl-plus \
  --disable-clarifier \
  --max-rounds 1 \
  --n-images 1 \
  --steps 12 \
  --width 768 \
  --height 768 \
  --seed 123
```

## M5 Layout Planner Smoke Test

M5 ports the useful ChainArchitect idea from LayerCraft behind project adapters.
It produces a validated layout package only: background, ordered foreground
objects, bboxes, and relations. It does not enable LayerCraft OIN/inpainting.

Mock layout run:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m5 \
  --prompt "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo" \
  --llm mock \
  --canvas-width 1024 \
  --canvas-height 1024
```

API layout run:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m5 \
  --prompt "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo" \
  --llm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --canvas-width 1024 \
  --canvas-height 1024
```

The run writes `layout.json` under `project/runs/<run-id>/`.

## M5.1 Layout-Guided M4 Prompting

M5.1 can feed the layout planner into the M4 loop as compact prompt guidance.
This still uses the normal image generator; it does not enforce boxes with
regional diffusion or inpainting yet.

Mock smoke:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m4 \
  --prompt "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo" \
  --generator mock \
  --llm mock \
  --vlm mock \
  --disable-clarifier \
  --disable-constraint-check \
  --use-layout-planner \
  --layout-llm mock \
  --max-rounds 1 \
  --n-images 1
```

Real SDXL/API run:

```bash
cd /home/zrr/study/t2i_agent_papers_2024_2025/mult-t2i-agent/project
DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
/home/zrr/anaconda3/envs/mult-t2i-agent-real/bin/python -m src.run_m4 \
  --prompt "a small red robot clearly gripping the handle of a blue umbrella, cinematic rainy street photo" \
  --generator sdxl \
  --llm api \
  --vlm api \
  --layout-llm api \
  --api-key-env DASHSCOPE_API_KEY \
  --llm-model qwen-plus \
  --vlm-model qwen-vl-plus \
  --disable-clarifier \
  --use-layout-planner \
  --max-rounds 2 \
  --n-images 2 \
  --steps 20 \
  --width 768 \
  --height 768 \
  --seed 124
```

The M4 run writes both `run.json` and `layout.json`; `images_generated` events
show the layout-guided prompt actually sent to the generator.
