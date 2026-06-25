"""Image generation adapter interfaces and mock/local implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
from typing import Any, Protocol, Sequence


class ImageGenerator(Protocol):
    """Image generation backend interface."""

    def generate(
        self,
        prompt: str | Sequence[str],
        n: int = 1,
        negative_prompt: str | None = None,
    ) -> list[str]:
        ...


@dataclass
class MockImageGenerator:
    """Return existing paths or local placeholder paths without real generation."""

    existing_paths: Sequence[str | Path] = ()
    placeholder_dir: str | Path | None = None
    create_placeholders: bool = False
    prefix: str = "mock_image"
    calls: list[dict[str, object]] = field(default_factory=list)
    _counter: int = 0

    def generate(
        self,
        prompt: str | Sequence[str],
        n: int = 1,
        negative_prompt: str | None = None,
    ) -> list[str]:
        prompts = _normalize_prompts(prompt, n)
        if n < 1:
            raise ValueError("n must be at least 1")

        outputs: list[str] = []
        for offset in range(n):
            existing = self._existing_path(offset)
            if existing is not None:
                outputs.append(existing)
                continue
            outputs.append(self._placeholder_path(prompts[offset], self._counter + offset))

        self.calls.append(
            {
                "prompt": prompts[0],
                "prompts": list(prompts),
                "n": n,
                "negative_prompt": negative_prompt,
                "outputs": list(outputs),
            }
        )
        self._counter += n
        return outputs

    def _existing_path(self, offset: int) -> str | None:
        if offset >= len(self.existing_paths):
            return None
        return str(self.existing_paths[offset])

    def _placeholder_path(self, prompt: str, index: int) -> str:
        if self.placeholder_dir is None:
            return f"mock://image/{index:04d}"

        directory = Path(self.placeholder_dir)
        path = directory / f"{self.prefix}_{index:04d}.txt"
        if self.create_placeholders:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"MOCK_IMAGE_PLACEHOLDER\nindex={index}\nprompt={prompt.strip()}\n",
                encoding="utf-8",
            )
        return str(path)


@dataclass
class DiffusersSDXLGenerator:
    """Local SDXL backend used as a secondary candidate/repair generator."""

    model_path: str | Path
    output_dir: str | Path
    device: str = "cuda"
    dtype: str = "float16"
    variant: str | None = "fp16"
    single_file: str | Path | None = None
    guidance_scale: float = 7.0
    num_inference_steps: int = 20
    width: int = 768
    height: int = 768
    seed: int | None = 42
    negative_prompt: str | None = None
    prefix: str = "sdxl_image"
    calls: list[dict[str, object]] = field(default_factory=list)
    _pipeline: Any = field(default=None, init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.output_dir = Path(self.output_dir).resolve()
        if self.single_file is not None:
            self.single_file = Path(self.single_file)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SDXL model path does not exist: {self.model_path}")
        if self.single_file is not None and not self.single_file.exists():
            raise FileNotFoundError(f"SDXL single file does not exist: {self.single_file}")
        if self.width < 64 or self.height < 64:
            raise ValueError("width and height must be at least 64")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")

    def generate(
        self,
        prompt: str | Sequence[str],
        n: int = 1,
        negative_prompt: str | None = None,
    ) -> list[str]:
        prompts = _normalize_prompts(prompt, n)
        negative_prompt = _clean_optional_prompt(
            negative_prompt if negative_prompt is not None else self.negative_prompt
        )
        pipe = self._load_pipeline()
        generator = self._torch_generator()
        kwargs: dict[str, Any] = {
            "prompt": prompts,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "generator": generator,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = [negative_prompt for _ in prompts]
        result = pipe(**kwargs)
        images = list(getattr(result, "images", []) or [])
        if len(images) < n:
            raise RuntimeError(
                f"SDXL generation finished but produced {len(images)} image(s), expected {n}."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        for image in images[:n]:
            path = self.output_dir / f"{self.prefix}_{self._counter:04d}.png"
            image.save(path)
            outputs.append(str(path))
            self._counter += 1
        self.calls.append(
            {
                "prompt": prompts[0],
                "prompts": list(prompts),
                "n": n,
                "negative_prompt": negative_prompt,
                "outputs": list(outputs),
                "model_path": str(self.model_path),
                "device": self.device,
            }
        )
        return outputs

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except Exception as exc:  # pragma: no cover - depends on local env.
            raise RuntimeError(
                "Diffusers SDXL backend requires torch and diffusers in this Python environment."
            ) from exc

        torch_dtype = _torch_dtype(torch, self.dtype)
        if self.single_file is not None:
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(self.single_file),
                torch_dtype=torch_dtype,
                variant=self.variant,
                local_files_only=True,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=torch_dtype,
                variant=self.variant,
                local_files_only=True,
            )
        self._pipeline = pipe.to(self.device)
        return self._pipeline

    def _torch_generator(self):
        if self.seed is None:
            return None
        try:
            import torch
        except Exception:  # pragma: no cover - handled by _load_pipeline first.
            return None
        device = self.device if str(self.device).startswith("cuda") else "cpu"
        return torch.Generator(device=device).manual_seed(int(self.seed) + self._counter)


@dataclass
class FluxCLIImageGenerator:
    """Local FLUX.1 generator backed by the Black Forest Labs flux CLI.

    This adapter deliberately shells out to the existing ``/home/zrr/flux``
    checkout so the project uses the same weight-loading path that has already
    been prepared on this machine.
    """

    flux_repo: str | Path
    output_dir: str | Path
    python: str | Path = "/mnt/ssd1/conda/envs/flux-dev/bin/python"
    project_dir: str | Path | None = None
    model_name: str = "flux-dev"
    attention_mode: str = "mgrag"
    mgrag_script: str | Path | None = None
    mgrag_model_id: str = "black-forest-labs/FLUX.1-dev"
    mgrag_delta_scale: float = 1.3
    mgrag_bias_scale: float = 1.0
    mgrag_intervene_steps: int = 20
    mgrag_local_files_only: bool = True
    mgrag_dtype: str = "bfloat16"
    mgrag_image_format: str = "jpg"
    mgrag_cpu_offload_mode: str = "model"
    model_path: str | Path | None = (
        "/home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/flux1-dev.safetensors"
    )
    ae_path: str | Path | None = (
        "/home/zrr/flux/checkpoints/black-forest-labs_FLUX.1-dev/ae.safetensors"
    )
    hf_home: str | Path | None = "/mnt/ssd3/zrr/hf_cache"
    device: str = "cuda"
    offload: bool = True
    offline: bool = True
    guidance_scale: float = 2.5
    num_inference_steps: int = 20
    width: int = 768
    height: int = 768
    seed: int | None = None
    negative_prompt: str | None = None
    timeout_seconds: int | None = None
    cuda_visible_devices: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, object]] = field(default_factory=list)
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.flux_repo = Path(self.flux_repo)
        self.output_dir = Path(self.output_dir).resolve()
        self.project_dir = (
            Path(self.project_dir).resolve()
            if self.project_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.mgrag_script = (
            Path(self.mgrag_script).resolve()
            if self.mgrag_script is not None
            else self.project_dir / "infer_mgrag_flux.py"
        )
        self.python = Path(self.python) if "/" in str(self.python) else self.python
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
        if self.ae_path is not None:
            self.ae_path = Path(self.ae_path)
        if self.hf_home is not None:
            self.hf_home = Path(self.hf_home)

        if not self.flux_repo.exists():
            raise FileNotFoundError(f"FLUX repo does not exist: {self.flux_repo}")
        if not (self.flux_repo / "src" / "flux").exists():
            raise FileNotFoundError(f"FLUX source package does not exist under: {self.flux_repo}")
        if isinstance(self.python, Path) and not self.python.exists():
            raise FileNotFoundError(f"FLUX Python executable does not exist: {self.python}")
        if self.attention_mode not in {"baseline", "mgrag"}:
            raise ValueError("attention_mode must be baseline or mgrag")
        if not self.mgrag_script.exists():
            raise FileNotFoundError(f"M-GRAG script does not exist: {self.mgrag_script}")
        if self.mgrag_intervene_steps < 0:
            raise ValueError("mgrag_intervene_steps must be non-negative")
        if self.mgrag_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("mgrag_dtype must be bfloat16, float16, or float32")
        if self.mgrag_image_format not in {"jpg", "png"}:
            raise ValueError("mgrag_image_format must be jpg or png")
        if self.mgrag_cpu_offload_mode not in {"model", "sequential", "none"}:
            raise ValueError("mgrag_cpu_offload_mode must be model, sequential, or none")
        if self.model_path is not None and not self.model_path.exists():
            raise FileNotFoundError(f"FLUX model checkpoint does not exist: {self.model_path}")
        if self.ae_path is not None and not self.ae_path.exists():
            raise FileNotFoundError(f"FLUX AE checkpoint does not exist: {self.ae_path}")
        if self.width < 64 or self.height < 64:
            raise ValueError("width and height must be at least 64")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")

    def generate(
        self,
        prompt: str | Sequence[str],
        n: int = 1,
        negative_prompt: str | None = None,
    ) -> list[str]:
        prompts = _normalize_prompts(prompt, n)
        negative_prompt = _clean_optional_prompt(
            negative_prompt if negative_prompt is not None else self.negative_prompt
        )
        flux_prompts = [_merge_flux_negative_prompt(item, negative_prompt) for item in prompts]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        before = set(self._output_files())
        if self.attention_mode == "mgrag":
            command = self._build_mgrag_command(flux_prompts)
            cwd = str(self.project_dir)
        else:
            command = self._build_baseline_command(flux_prompts)
            cwd = str(self.flux_repo)
        result = subprocess.run(
            command,
            cwd=cwd,
            env=self._build_env(),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0:
            details = _summarize_process_output(result.stdout, result.stderr)
            raise RuntimeError(f"FLUX generation failed with exit code {result.returncode}.{details}")

        new_files = [path for path in self._output_files() if path not in before]
        if len(new_files) < n:
            details = _summarize_process_output(result.stdout, result.stderr)
            raise RuntimeError(
                f"FLUX generation finished but produced {len(new_files)} image(s), expected {n}.{details}"
            )

        outputs = [str(path) for path in new_files[:n]]
        self._counter += n
        self.calls.append(
            {
                "prompt": prompts[0],
                "prompts": list(prompts),
                "flux_prompts": list(flux_prompts),
                "n": n,
                "negative_prompt": negative_prompt,
                "outputs": list(outputs),
                "command": list(command),
                "attention_mode": self.attention_mode,
                "mgrag_delta_scale": self.mgrag_delta_scale,
                "mgrag_bias_scale": self.mgrag_bias_scale,
                "mgrag_intervene_steps": self.mgrag_intervene_steps,
            }
        )
        return outputs

    def _build_baseline_command(self, prompts: Sequence[str]) -> list[str]:
        command = [
            str(self.python),
            "-m",
            "flux",
            "t2i",
            "--name",
            self.model_name,
            "--height",
            str(self.height),
            "--width",
            str(self.width),
            "--num_steps",
            str(self.num_inference_steps),
            "--prompt",
            "|".join(prompt.replace("|", ",") for prompt in prompts),
            "--device",
            self.device,
            "--guidance",
            str(self.guidance_scale),
            "--output_dir",
            str(self.output_dir),
        ]
        if self.seed is not None:
            command.extend(["--seed", str(int(self.seed))])
        if self.offload:
            command.append("--offload")
        return command

    def _build_mgrag_command(self, prompts: Sequence[str]) -> list[str]:
        effective_intervene_steps = min(
            int(self.mgrag_intervene_steps),
            int(self.num_inference_steps),
        )
        command = [
            str(self.python),
            str(self.mgrag_script),
            "--output_dir",
            str(self.output_dir),
            "--output_prefix",
            "img",
            "--start_index",
            str(self._counter),
            "--image_format",
            self.mgrag_image_format,
            "--model_id",
            self.mgrag_model_id,
            "--dtype",
            self.mgrag_dtype,
            "--delta_list",
            str(self.mgrag_delta_scale),
            "--bias_list",
            str(self.mgrag_bias_scale),
            "--intervene_steps",
            str(effective_intervene_steps),
            "--steps",
            str(self.num_inference_steps),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--guidance_scale",
            str(self.guidance_scale),
            "--seed",
            str(int(self.seed) if self.seed is not None else 42),
            "--device",
            self.device,
        ]
        if self.mgrag_local_files_only:
            command.append("--local_files_only")
        command.extend(["--cpu-offload-mode", self.mgrag_cpu_offload_mode])
        for prompt in prompts:
            command.extend(["--prompt", prompt])
        return command

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        src_path = str(self.flux_repo / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if self.model_path is not None:
            env["FLUX_MODEL"] = str(self.model_path)
        if self.ae_path is not None:
            env["FLUX_AE"] = str(self.ae_path)
        if self.hf_home is not None:
            env["HF_HOME"] = str(self.hf_home)
            env["HUGGINGFACE_HUB_CACHE"] = str(self.hf_home / "hub")
        if self.offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env.update(self.extra_env)
        return env

    def _output_files(self) -> list[Path]:
        return sorted(
            [
                *self.output_dir.glob("img_*.jpg"),
                *self.output_dir.glob("img_*.png"),
            ]
        )


@dataclass
class FusionImageGenerator:
    """FLUX-first candidate fusion with SDXL as auxiliary backend."""

    flux: ImageGenerator
    sdxl: ImageGenerator
    policy: str = "parallel"
    negative_prompt: str | None = None
    calls: list[dict[str, object]] = field(default_factory=list)
    last_metadata: list[dict[str, object]] = field(default_factory=list)
    children: dict[str, ImageGenerator] = field(init=False)

    def __post_init__(self) -> None:
        if self.policy not in {"parallel", "flux-first", "sdxl-repair"}:
            raise ValueError("policy must be parallel, flux-first, or sdxl-repair")
        self.children = {"flux": self.flux, "sdxl": self.sdxl}

    def generate(
        self,
        prompt: str | Sequence[str],
        n: int = 1,
        negative_prompt: str | None = None,
    ) -> list[str]:
        prompts = _normalize_prompts(prompt, n)
        negative_prompt = _clean_optional_prompt(
            negative_prompt if negative_prompt is not None else self.negative_prompt
        )
        outputs: list[str] = []
        metadata: list[dict[str, object]] = []

        if self.policy == "flux-first":
            flux_outputs, flux_meta = self._generate_backend(
                "flux", self.flux, prompts, negative_prompt
            )
            outputs.extend(flux_outputs)
            metadata.extend(flux_meta)
            if not outputs:
                sdxl_outputs, sdxl_meta = self._generate_backend(
                    "sdxl", self.sdxl, prompts, negative_prompt
                )
                outputs.extend(sdxl_outputs)
                metadata.extend(sdxl_meta)
        elif self.policy == "sdxl-repair":
            for label, backend in (("flux", self.flux), ("sdxl_repair", self.sdxl)):
                backend_outputs, backend_meta = self._generate_backend(
                    label, backend, prompts, negative_prompt
                )
                outputs.extend(backend_outputs)
                metadata.extend(backend_meta)
        else:
            for label, backend in (("flux", self.flux), ("sdxl", self.sdxl)):
                backend_outputs, backend_meta = self._generate_backend(
                    label, backend, prompts, negative_prompt
                )
                outputs.extend(backend_outputs)
                metadata.extend(backend_meta)

        if not outputs:
            raise RuntimeError("Fusion generation produced no images.")
        self.last_metadata = metadata
        self.calls.append(
            {
                "prompt": prompts[0],
                "prompts": list(prompts),
                "n": n,
                "negative_prompt": negative_prompt,
                "policy": self.policy,
                "outputs": list(outputs),
                "metadata": [dict(item) for item in metadata],
            }
        )
        return outputs

    def _generate_backend(
        self,
        label: str,
        backend: ImageGenerator,
        prompts: Sequence[str],
        negative_prompt: str | None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        outputs = backend.generate(
            list(prompts),
            n=len(prompts),
            negative_prompt=negative_prompt,
        )
        metadata = [
            {
                "backend": label,
                "prompt": prompts[min(index, len(prompts) - 1)],
                "path": path,
                "candidate_index": index,
            }
            for index, path in enumerate(outputs)
        ]
        return list(outputs), metadata


def _clean_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("prompt must be a non-empty string")
    return cleaned


def _normalize_prompts(prompt: str | Sequence[str], n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if isinstance(prompt, str):
        return [_clean_prompt(prompt)] * n
    prompts = [_clean_prompt(str(item)) for item in prompt]
    if len(prompts) != n:
        raise ValueError("prompt sequence length must match n")
    return prompts


def _clean_optional_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("negative_prompt must be a string or None")
    cleaned = value.strip()
    return cleaned or None


def _merge_flux_negative_prompt(prompt: str, negative_prompt: str | None) -> str:
    if not negative_prompt:
        return prompt
    return f"{prompt}\nAvoid: {negative_prompt}."


def _torch_dtype(torch_module: Any, dtype: str):
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float32":
        return torch_module.float32
    raise ValueError("dtype must be float16, bfloat16, or float32")


def _summarize_process_output(stdout: str | None, stderr: str | None, max_chars: int = 2000) -> str:
    chunks = []
    if stdout:
        chunks.append("stdout:\n" + stdout[-max_chars:])
    if stderr:
        chunks.append("stderr:\n" + stderr[-max_chars:])
    if not chunks:
        return ""
    return "\n" + "\n".join(chunks)
