"""
Fixed-window M-GRAG on FLUX.1-Dev
==================================
  - 使用同一个 AdaptiveSinglePassProcessor（内部通过 is_double_block 分支
    处理 Double / Single Block，Double+Single 挂载同一实例）
  - 标准 processor 只使用 FluxAttnProcessor2_0（与自适应版一致）

"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0

prev_attn_outputs = {}  # layer_idx -> tensor (Q_img, H*d), CPU fp16
curr_layer_deltas = []  # 当前step内各layer的delta标量
_layer_counter    = [0]


def resolve_hf_cache_dir() -> str | None:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return hub_cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    return None

def reset_for_new_step():
    _layer_counter[0] = 0
    curr_layer_deltas.clear()


class AdaptiveSinglePassProcessor:
    def __init__(self, delta_scale: float = 1.0, bias_scale: float = 1.0):
        self.delta_scale = delta_scale
        self.bias_scale  = bias_scale
        self.txt_len_standard = 512

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.FloatTensor:

        batch_size      = hidden_states.shape[0]
        is_double_block = encoder_hidden_states is not None

        # --- 1. Q K V Projections ---
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key   = attn.norm_k(key)

        # --- 2. 特征拼接 ---
        if is_double_block:
            enc_query = attn.add_q_proj(encoder_hidden_states)
            enc_key   = attn.add_k_proj(encoder_hidden_states)
            enc_value = attn.add_v_proj(encoder_hidden_states)

            enc_query = enc_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_key   = enc_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_value = enc_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None: enc_query = attn.norm_added_q(enc_query)
            if attn.norm_added_k is not None: enc_key   = attn.norm_added_k(enc_key)

            joint_query    = torch.cat([enc_query, query], dim=2)
            joint_key_orig = torch.cat([enc_key,   key],   dim=2)
            joint_value    = torch.cat([enc_value, value],  dim=2)
            txt_seq_len    = enc_query.shape[2]
        else:
            joint_query, joint_key_orig, joint_value = query, key, value
            txt_seq_len = self.txt_len_standard

        # --- 3. RoPE ---
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            if isinstance(image_rotary_emb, tuple):
                joint_query    = apply_rotary_emb(joint_query,    image_rotary_emb)
                joint_key_orig = apply_rotary_emb(joint_key_orig, image_rotary_emb)
            else:
                joint_query    = apply_rotary_emb(joint_query,    image_rotary_emb, use_real=False)
                joint_key_orig = apply_rotary_emb(joint_key_orig, image_rotary_emb, use_real=False)

        # --- 4. M-GRAG 方差放大（K_txt） ---
        k_txt_rot   = joint_key_orig[:, :, :txt_seq_len, :]
        k_img_rot   = joint_key_orig[:, :, txt_seq_len:, :]
        txt_bias    = k_txt_rot.mean(dim=2, keepdim=True)
        k_txt_mgrag = self.bias_scale * txt_bias + self.delta_scale * (k_txt_rot - txt_bias)
        joint_key_mgrag = torch.cat([k_txt_mgrag, k_img_rot], dim=2)

        # --- 5. Single-Pass Attention ---
        attn_output = F.scaled_dot_product_attention(
            joint_query, joint_key_mgrag, joint_value, is_causal=False
        )

        # --- 6. 主路自监测 ---
        if is_double_block:
            self._rolling_monitor(attn_output, txt_seq_len)

        # --- 7. Format & Projection ---
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_output = attn_output.to(query.dtype)

        if is_double_block:
            enc_out = attn_output[:, :txt_seq_len]
            img_out = attn_output[:, txt_seq_len:]
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            if not attn.context_pre_only:
                enc_out = attn.to_add_out(enc_out)
            return img_out, enc_out
        else:
            return attn_output

    def _rolling_monitor(self, attn_out, txt_seq_len):
        layer_idx = _layer_counter[0]
        _layer_counter[0] += 1

        img_out = attn_out[0, :, txt_seq_len:, :]            # (H, Q_img, d)
        H, Q_img, d = img_out.shape
        curr_feat = img_out.detach().permute(1, 0, 2).reshape(Q_img, H * d).half().cpu()

        if layer_idx in prev_attn_outputs:
            prev_feat = prev_attn_outputs[layer_idx].float()
            curr_f    = curr_feat.float()
            diff_norm = torch.norm(curr_f - prev_feat, p='fro').item()
            base_norm = torch.norm(prev_feat,           p='fro').item() + 1e-8
            curr_layer_deltas.append(diff_norm / base_norm)
            del prev_feat, curr_f

        prev_attn_outputs[layer_idx] = curr_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="/data/xst/TACA/T2I-CompBench/examples/dataset") # 评估集路径
    parser.add_argument("--prompt_file", type=str,   default=None) # 评估集文件
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Single prompt to generate. Can be repeated. When set, --prompt_file is not required.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="img")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--image_format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--gpu_id",      type=int,   default=0)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--delta_list",  type=float, nargs='+', default=[1.3]) # 差值放大参数
    parser.add_argument("--bias_list",   type=float, nargs='+', default=[1.0]) # 平均值放大参数
    parser.add_argument("--resolution",  type=int,   default=1024)
    parser.add_argument("--width",       type=int,   default=None)
    parser.add_argument("--height",      type=int,   default=None)
    parser.add_argument("--steps",       type=int,   default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--prompt_seed",
        action="append",
        type=int,
        default=None,
        help=(
            "Optional per-prompt seed. Can be repeated and must match the "
            "number of --prompt values in direct prompt mode."
        ),
    )
    parser.add_argument("--num_samples", type=int,   default=300)
    parser.add_argument(
        "--intervene_steps",
        type=int,
        default=20,
        help="Fixed number of initial denoising steps using M-GRAG. "
             "3 means steps 0,1,2. Set to 0 for baseline (no intervention).",
    ) # 方差放大方法的干预步数
    parser.add_argument(
        "--cpu-offload-mode",
        type=str,
        choices=["model", "sequential", "none"],
        default="model",
        help="CPU offload strategy. model=enable_model_cpu_offload (default, peak ~20G VRAM), "
             "sequential=enable_sequential_cpu_offload (slower, peak ~12G VRAM), "
             "none=no offload (fastest, needs ~24G VRAM).",
    )
    args = parser.parse_args()

    if args.intervene_steps < 0:
        raise ValueError("--intervene_steps must be non-negative")
    if args.intervene_steps > args.steps:
        raise ValueError("--intervene_steps cannot exceed --steps")
    if args.prompt is None and not args.prompt_file:
        raise ValueError("Either --prompt or --prompt_file must be provided")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.prompt_seed is not None and args.prompt is None:
        raise ValueError("--prompt_seed can only be used with direct --prompt mode")
    if args.prompt_seed is not None and len(args.prompt_seed) != len(args.prompt or []):
        raise ValueError("--prompt_seed count must match --prompt count")

    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else args.resolution
    if width < 64 or height < 64:
        raise ValueError("--width/--height must be at least 64")

    device = args.device or f"cuda:{args.gpu_id}"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    cache_dir = resolve_hf_cache_dir()
    if cache_dir:
        print(f"[cache] HUGGINGFACE cache_dir={cache_dir}", flush=True)

    # GPU memory pre-check
    if device.startswith("cuda"):
        try:
            import subprocess
            gpu_id = device.split(":")[-1] if ":" in device else "0"
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu_id],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                free_mb = int(result.stdout.strip())
                print(f"[GPU {gpu_id}] free memory: {free_mb} MiB", flush=True)
                min_required = {"model": 18000, "sequential": 10000, "none": 22000}.get(args.cpu_offload_mode, 18000)
                if free_mb < min_required:
                    print(f"[WARNING] Free memory {free_mb} MiB < recommended {min_required} MiB for --cpu-offload-mode {args.cpu_offload_mode}", flush=True)
                    print(f"[WARNING] May OOM or hang. Consider using --cpu-offload-mode sequential or wait for GPU to free up.", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not check GPU memory: {e}", flush=True)

    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        local_files_only=bool(args.local_files_only),
        cache_dir=cache_dir,
    )

    # Apply CPU offload strategy
    if args.cpu_offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
        print(f"[offload] enable_model_cpu_offload(device={device})", flush=True)
    elif args.cpu_offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        print(f"[offload] enable_sequential_cpu_offload(device={device}) - slower but lower peak VRAM", flush=True)
    elif args.cpu_offload_mode == "none":
        pipe.to(device)
        print(f"[offload] no offload, full model on {device} - fastest but needs ~24G VRAM", flush=True)

    pipe.set_progress_bar_config(disable=True)

    direct_prompt_mode = args.prompt is not None
    if direct_prompt_mode:
        all_prompts = [item.strip() for item in args.prompt or [] if item.strip()]
        prompts_to_test = list(enumerate(all_prompts))
    else:
        with open(os.path.join(args.data_dir, args.prompt_file), 'r') as f:
            all_prompts = [l.strip() for l in f.readlines()]

        random.seed(args.seed)
        indices = sorted(random.sample(range(len(all_prompts)), min(args.num_samples, len(all_prompts))))
        prompts_to_test = [(i, all_prompts[i]) for i in indices]
    if not prompts_to_test:
        raise ValueError("No prompts selected for generation")

    # 标准 processor
    standard_processor = FluxAttnProcessor2_0()

    for b_scale in args.bias_list:
        for d_scale in args.delta_list:
            print(f"\n   delta={d_scale}, bias={b_scale}, "
                  f"intervene_steps={args.intervene_steps}")

            mgrag_processor = AdaptiveSinglePassProcessor(
                delta_scale=d_scale, bias_scale=b_scale
            )

            save_dir = os.path.join(
                args.output_dir
                if args.output_dir
                else os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f"outputs/{args.prompt_file.replace('.txt', '')}"
                    f"/mgrag_fixed_b{b_scale}_d{d_scale}"
                    f"_s{args.intervene_steps}_r{args.resolution}"
                )
            )
            os.makedirs(save_dir, exist_ok=True)
            output_records = []

            for local_idx, (prompt_idx, prompt) in enumerate(
                tqdm(prompts_to_test, desc=f"B:{b_scale} D:{d_scale} S:{args.intervene_steps}")
            ):
                if direct_prompt_mode:
                    output_index = args.start_index + local_idx
                    save_name = os.path.join(
                        save_dir,
                        f"{args.output_prefix}_{output_index:04d}.{args.image_format}",
                    )
                else:
                    safe_prompt = prompt.replace('/', ' ').replace('\\', ' ')
                    save_name = os.path.join(save_dir, f"{safe_prompt}_{prompt_idx:06d}.png")
                if os.path.exists(save_name):
                    output_records.append(
                        {
                            "prompt_index": prompt_idx,
                            "prompt": prompt,
                            "path": save_name,
                            "skipped_existing": True,
                        }
                    )
                    continue

                # 每次生成前重置状态并重新挂载 processor
                prev_attn_outputs.clear()
                reset_for_new_step()

                if args.intervene_steps > 0:
                    for block in pipe.transformer.transformer_blocks:
                        block.attn.set_processor(mgrag_processor)
                    for block in pipe.transformer.single_transformer_blocks:
                        block.attn.set_processor(mgrag_processor)
                else:
                    for block in pipe.transformer.transformer_blocks:
                        block.attn.set_processor(standard_processor)
                    for block in pipe.transformer.single_transformer_blocks:
                        block.attn.set_processor(standard_processor)

                def step_end_callback(pipeline, step_index, timestep, callback_kwargs):
                    if args.intervene_steps > 0 and step_index == args.intervene_steps - 1:
                        # 固定窗口结束：切换回标准 processor
                        for block in pipeline.transformer.transformer_blocks:
                            block.attn.set_processor(standard_processor)
                        for block in pipeline.transformer.single_transformer_blocks:
                            block.attn.set_processor(standard_processor)
                        prev_attn_outputs.clear()
                    reset_for_new_step()
                    return callback_kwargs

                prompt_seed = (
                    int(args.prompt_seed[local_idx])
                    if args.prompt_seed is not None
                    else int(args.seed) + local_idx
                )
                generator = torch.Generator("cpu").manual_seed(prompt_seed)
                image = pipe(
                    prompt,
                    height=height, width=width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    callback_on_step_end=step_end_callback,
                ).images[0]
                image.save(save_name)
                output_records.append(
                        {
                            "prompt_index": prompt_idx,
                            "prompt": prompt,
                            "path": save_name,
                            "seed": prompt_seed,
                            "skipped_existing": False,
                        }
                )

            torch.cuda.empty_cache()

            if direct_prompt_mode:
                metadata_path = os.path.join(save_dir, f"{args.output_prefix}_mgrag_metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model_id": args.model_id,
                            "local_files_only": bool(args.local_files_only),
                            "width": width,
                            "height": height,
                            "steps": args.steps,
                            "guidance_scale": args.guidance_scale,
                            "seed": args.seed,
                            "prompt_seeds": list(args.prompt_seed or []),
                            "bias_scale": b_scale,
                            "delta_scale": d_scale,
                            "intervene_steps": args.intervene_steps,
                            "outputs": output_records,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

    print("\n All done.")


if __name__ == "__main__":
    main()
