#!/usr/bin/env bash
set -euo pipefail

cd /home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project_flux

PY=/home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project/.conda-m0/bin/python
OUT=/mnt/ssd1/t2i_agent_outputs/mult-t2i-agent/project_flux/runs_mini_benchmark
LOGDIR=/mnt/ssd1/t2i_agent_outputs/mult-t2i-agent/project_flux/nightly_logs
LOG="$LOGDIR/nightly_22_45_90_gpu0_tmux_$(date +%Y%m%d_%H%M%S).log"

export HF_HOME=/mnt/ssd3/zrr/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/ssd3/zrr/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT" "$LOGDIR"
exec > >(tee -a "$LOG") 2>&1

echo "[start] nightly 22 -> 45 -> 90 gpu0: $(date)"
echo "[log] $LOG"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "[error] DASHSCOPE_API_KEY is not set"
  exit 2
fi

"$PY" scripts/build_nightly_extension_benchmarks.py

check_stage() {
  local summary_path="$1"
  local stage_name="$2"
  "$PY" - "$summary_path" "$stage_name" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
stage_name = sys.argv[2]
if not summary_path.exists():
    raise SystemExit(f"[error] {stage_name}: missing summary {summary_path}")

data = json.loads(summary_path.read_text())
agg = data.get("aggregate", {})
total = int(agg.get("total") or len(data.get("results", [])) or 0)
status_counts = agg.get("status_counts", {})
subprocess_failed = int(status_counts.get("subprocess_failed", 0))
evaluable = int(agg.get("evaluable_cases", 0))
passed = int(agg.get("completion_passed", 0))
typed = agg.get("typed_route_counts", {})
edits = int(agg.get("efficient_edit_attempts", 0))
accepted = int(agg.get("accepted_edit_count", 0))

print(
    f"[check] {stage_name}: total={total}, evaluable={evaluable}, "
    f"passed={passed}, subprocess_failed={subprocess_failed}, "
    f"typed_routes={typed}, edit_attempts={edits}, accepted_edits={accepted}"
)

if total > 0 and subprocess_failed == total:
    raise SystemExit(
        f"[error] {stage_name}: all {total} cases are subprocess_failed; "
        "stop before wasting the remaining stages"
    )
PY
}

COMMON_ARGS=(
  --generator flux
  --llm mock
  --vlm api
  --api-key-env DASHSCOPE_API_KEY
  --llm-model qwen-plus
  --vlm-model qwen-vl-plus
  --cuda-visible-devices 0
  --width 512
  --height 512
  --steps 30
  --max-rounds 2
  --n-images 1
  --score-threshold 0.85
  --flux-timeout-seconds 900
  --subprocess-timeout-seconds 1200
  --retry-on-infra-failure 1
  --seed-policy case-id
  --seed-base 7100
  --flux-hf-home /mnt/ssd3/zrr/hf_cache
  --flux-model-path /home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/flux1-dev.safetensors
  --flux-ae-path /home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/ae.safetensors
  --flux-attn-mode mgrag
  --mgrag-delta-scale 1.3
  --mgrag-bias-scale 1.0
  --mgrag-intervene-steps 20
  --mgrag-dtype bfloat16
  --mgrag-image-format jpg
)

EDIT_ARGS=(
  --enable-vlm-target-locator
  --enable-relation-repair
  --enable-object-insertion-repair
  --enable-efficient-repair-agent
  --enable-editing-mask-agent
  --editing-mask-mode auto
  --editing-mask-dilation-kernel-size 31
  --allow-editing-bbox-fallback
  --grounded-sam2-python /mnt/ssd1/conda/envs/tweediemix/bin/python
  --grounded-sam2-cuda-visible-devices 0
  --grounded-sam2-hf-home /mnt/ssd1/powerpaint_envs/hf-cache
  --relation-editor powerpaint-subprocess
  --relation-candidates 1
  --relation-inpaint-cuda-visible-devices 0
  --powerpaint-python /mnt/ssd1/powerpaint_envs/.conda-powerpaint/bin/python
  --powerpaint-timeout-seconds 1800
  --relation-steps 20
  --relation-guidance-scale 7.5
  --relation-strength 1.0
)

echo "[stage] nightly22: $(date)"
"$PY" scripts/run_mini_benchmark.py \
  --benchmark benchmarks/hard_prompts_all_categories_10each.json \
  --runs-dir "$OUT/real-flux-nightly22-512-30-gpu0-r3" \
  --run-prefix real-flux-nightly22-512-30-gpu0-r3 \
  "${COMMON_ARGS[@]}" "${EDIT_ARGS[@]}" \
  --case-id all10_count_001 \
  --case-id all10_count_004 \
  --case-id all10_count_006 \
  --case-id all10_count_009 \
  --case-id all10_spatial_001 \
  --case-id all10_spatial_005 \
  --case-id all10_spatial_006 \
  --case-id all10_spatial_010 \
  --case-id all10_attribute_003 \
  --case-id all10_attribute_005 \
  --case-id all10_attribute_007 \
  --case-id all10_attribute_010 \
  --case-id all10_color_001 \
  --case-id all10_color_002 \
  --case-id all10_color_005 \
  --case-id all10_color_009 \
  --case-id all10_interaction_004 \
  --case-id all10_interaction_009 \
  --case-id all10_interaction_010 \
  --case-id all10_negation_006 \
  --case-id all10_negation_009 \
  --case-id all10_negation_010
check_stage "$OUT/real-flux-nightly22-512-30-gpu0-r3/summary.json" "nightly22"

echo "[stage] extension45: $(date)"
"$PY" scripts/run_mini_benchmark.py \
  --benchmark benchmarks/hard_prompts_extension_weak_categories_45.json \
  --runs-dir "$OUT/real-flux-extension45-512-30-gpu0-r3" \
  --run-prefix real-flux-extension45-512-30-gpu0-r3 \
  "${COMMON_ARGS[@]}" "${EDIT_ARGS[@]}"
check_stage "$OUT/real-flux-extension45-512-30-gpu0-r3/summary.json" "extension45"

echo "[stage] extension90: $(date)"
"$PY" scripts/run_mini_benchmark.py \
  --benchmark benchmarks/hard_prompts_extension_new90.json \
  --runs-dir "$OUT/real-flux-extension90-512-30-gpu0-r3" \
  --run-prefix real-flux-extension90-512-30-gpu0-r3 \
  "${COMMON_ARGS[@]}" "${EDIT_ARGS[@]}"
check_stage "$OUT/real-flux-extension90-512-30-gpu0-r3/summary.json" "extension90"

echo "[done] nightly 22 -> 45 -> 90 finished: $(date)"
