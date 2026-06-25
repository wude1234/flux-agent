import json
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients import OpenAICompatibleLLMClient, OpenAICompatibleVLMClient
from src.image_generator import FluxCLIImageGenerator, FusionImageGenerator, MockImageGenerator


def test_openai_compatible_llm_uses_transport_without_network() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {"choices": [{"message": {"content": "hello"}}]}

    client = OpenAICompatibleLLMClient(
        model="qwen-plus",
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    assert client.text("write a prompt") == "hello"
    assert calls[0][0] == "https://example.test/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer test-key"
    assert calls[0][2]["model"] == "qwen-plus"
    assert calls[0][2]["messages"][0]["content"] == "write a prompt"


def test_openai_compatible_client_retries_transient_connection_error() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        if len(calls) == 1:
            raise RuntimeError("API connection error: SSL EOF")
        return {"choices": [{"message": {"content": "ok after retry"}}]}

    client = OpenAICompatibleLLMClient(
        model="qwen-plus",
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=transport,
        retry_backoff=0,
        max_retries=1,
    )

    assert client.text("retry please") == "ok after retry"
    assert len(calls) == 2


def test_openai_compatible_vlm_embeds_local_image_data_uri(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    )
    calls = []

    def transport(url, headers, payload, timeout):
        del url, headers, timeout
        calls.append(payload)
        return {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.8, "errors": []})}}
            ]
        }

    client = OpenAICompatibleVLMClient(
        model="qwen-vl-plus",
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    response = client.vision("score this", [str(image_path)])

    assert json.loads(response)["score"] == 0.8
    content = calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "score this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_compatible_vlm_compresses_large_local_images(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "large.png"
    image = Image.new("RGB", (512, 512))
    pixels = []
    for y in range(512):
        for x in range(512):
            pixels.append(((x * 37 + y * 13) % 256, (x * 11 + y * 29) % 256, (x * 5 + y * 7) % 256))
    image.putdata(pixels)
    image.save(image_path)
    calls = []

    def transport(url, headers, payload, timeout):
        del url, headers, timeout
        calls.append(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = OpenAICompatibleVLMClient(
        model="qwen-vl-plus",
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=transport,
        max_image_data_uri_bytes=20_000,
        image_preprocess_max_side=256,
    )

    assert client.vision("check", [str(image_path)]) == "ok"
    url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    payload = client.calls[0]["image_payloads"][0]

    assert url.startswith("data:image/jpeg;base64,")
    assert payload["compressed"] is True
    assert payload["bytes"] <= 20_000
    assert payload["original_data_uri_bytes"] > payload["bytes"]

def test_flux_cli_generator_invokes_local_flux_and_collects_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flux_repo = tmp_path / "flux"
    (flux_repo / "src" / "flux").mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "flux1-dev.safetensors"
    model.write_text("model", encoding="utf-8")
    ae = tmp_path / "ae.safetensors"
    ae.write_text("ae", encoding="utf-8")
    hf_home = tmp_path / "hf_cache"
    output_dir = tmp_path / "images"

    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        del capture_output, text, timeout
        calls.append((command, cwd, env))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "img_0.jpg").write_text("image 0", encoding="utf-8")
        (output_dir / "img_1.jpg").write_text("image 1", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.image_generator.subprocess.run", fake_run)

    generator = FluxCLIImageGenerator(
        flux_repo=flux_repo,
        python=python,
        model_path=model,
        ae_path=ae,
        hf_home=hf_home,
        output_dir=output_dir,
        width=256,
        height=256,
        num_inference_steps=1,
        seed=123,
        cuda_visible_devices="0",
        attention_mode="baseline",
    )

    outputs = generator.generate(["a red cube", "a blue sphere"], n=2, negative_prompt="blurry")

    assert outputs == [str(output_dir / "img_0.jpg"), str(output_dir / "img_1.jpg")]
    command, cwd, env = calls[0]
    assert cwd == str(flux_repo)
    assert command[:4] == [str(python), "-m", "flux", "t2i"]
    assert "--name" in command
    assert "flux-dev" in command
    prompt_arg = command[command.index("--prompt") + 1]
    assert "a red cube" in prompt_arg
    assert "a blue sphere" in prompt_arg
    assert "Avoid: blurry." in prompt_arg
    assert env["FLUX_MODEL"] == str(model)
    assert env["FLUX_AE"] == str(ae)
    assert env["HF_HOME"] == str(hf_home)
    assert env["HUGGINGFACE_HUB_CACHE"] == str(hf_home / "hub")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert generator.calls[0]["negative_prompt"] == "blurry"


def test_flux_cli_generator_defaults_to_mgrag_attention_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flux_repo = tmp_path / "flux"
    (flux_repo / "src" / "flux").mkdir(parents=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    mgrag_script = project_dir / "infer_mgrag_flux.py"
    mgrag_script.write_text("# mgrag", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "flux1-dev.safetensors"
    model.write_text("model", encoding="utf-8")
    ae = tmp_path / "ae.safetensors"
    ae.write_text("ae", encoding="utf-8")
    output_dir = tmp_path / "images"
    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        del env, capture_output, text, timeout
        calls.append((command, cwd))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "img_0000.jpg").write_text("image 0", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.image_generator.subprocess.run", fake_run)

    generator = FluxCLIImageGenerator(
        flux_repo=flux_repo,
        project_dir=project_dir,
        mgrag_script=mgrag_script,
        python=python,
        model_path=model,
        ae_path=ae,
        output_dir=output_dir,
        width=256,
        height=256,
        num_inference_steps=30,
        seed=123,
    )

    outputs = generator.generate("a red cube", n=1)

    assert outputs == [str(output_dir / "img_0000.jpg")]
    command, cwd = calls[0]
    assert cwd == str(project_dir)
    assert command[:2] == [str(python), str(mgrag_script)]
    assert command[command.index("--delta_list") + 1] == "1.3"
    assert command[command.index("--bias_list") + 1] == "1.0"
    assert command[command.index("--intervene_steps") + 1] == "20"
    assert "--local_files_only" in command
    assert generator.calls[0]["attention_mode"] == "mgrag"


def test_flux_cli_generator_clamps_mgrag_intervention_to_total_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flux_repo = tmp_path / "flux"
    (flux_repo / "src" / "flux").mkdir(parents=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    mgrag_script = project_dir / "infer_mgrag_flux.py"
    mgrag_script.write_text("# mgrag", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "flux1-dev.safetensors"
    model.write_text("model", encoding="utf-8")
    ae = tmp_path / "ae.safetensors"
    ae.write_text("ae", encoding="utf-8")
    output_dir = tmp_path / "images"
    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        del cwd, env, capture_output, text, timeout
        calls.append(command)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "img_0000.jpg").write_text("image 0", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.image_generator.subprocess.run", fake_run)

    generator = FluxCLIImageGenerator(
        flux_repo=flux_repo,
        project_dir=project_dir,
        mgrag_script=mgrag_script,
        python=python,
        model_path=model,
        ae_path=ae,
        output_dir=output_dir,
        width=256,
        height=256,
        num_inference_steps=12,
        mgrag_intervene_steps=20,
    )

    generator.generate("a red cube", n=1)

    command = calls[0]
    assert command[command.index("--steps") + 1] == "12"
    assert command[command.index("--intervene_steps") + 1] == "12"


def test_flux_cli_generator_baseline_attention_uses_flux_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flux_repo = tmp_path / "flux"
    (flux_repo / "src" / "flux").mkdir(parents=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    mgrag_script = project_dir / "infer_mgrag_flux.py"
    mgrag_script.write_text("# mgrag", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "flux1-dev.safetensors"
    model.write_text("model", encoding="utf-8")
    ae = tmp_path / "ae.safetensors"
    ae.write_text("ae", encoding="utf-8")
    output_dir = tmp_path / "images"
    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        del env, capture_output, text, timeout
        calls.append((command, cwd))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "img_0.jpg").write_text("image 0", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.image_generator.subprocess.run", fake_run)

    generator = FluxCLIImageGenerator(
        flux_repo=flux_repo,
        project_dir=project_dir,
        mgrag_script=mgrag_script,
        python=python,
        model_path=model,
        ae_path=ae,
        output_dir=output_dir,
        width=256,
        height=256,
        num_inference_steps=30,
        attention_mode="baseline",
    )

    outputs = generator.generate("a red cube", n=1)

    assert outputs == [str(output_dir / "img_0.jpg")]
    command, cwd = calls[0]
    assert cwd == str(flux_repo)
    assert command[:4] == [str(python), "-m", "flux", "t2i"]
    assert generator.calls[0]["attention_mode"] == "baseline"


def test_flux_cli_generator_resolves_relative_output_dir_for_flux_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flux_repo = tmp_path / "flux"
    project_dir = tmp_path / "project"
    (flux_repo / "src" / "flux").mkdir(parents=True)
    project_dir.mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "flux1-dev.safetensors"
    model.write_text("model", encoding="utf-8")
    ae = tmp_path / "ae.safetensors"
    ae.write_text("ae", encoding="utf-8")
    relative_output = Path("runs") / "case" / "images"
    expected_output = (project_dir / relative_output).resolve()
    calls = []

    def fake_run(command, cwd, env, capture_output, text, timeout):
        del env, capture_output, text, timeout
        calls.append((command, cwd))
        output_arg = Path(command[command.index("--output_dir") + 1])
        assert output_arg.is_absolute()
        assert output_arg == expected_output
        output_arg.mkdir(parents=True, exist_ok=True)
        (output_arg / "img_0.jpg").write_text("image 0", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.image_generator.subprocess.run", fake_run)
    monkeypatch.chdir(project_dir)

    generator = FluxCLIImageGenerator(
        flux_repo=flux_repo,
        python=python,
        model_path=model,
        ae_path=ae,
        output_dir=relative_output,
        width=256,
        height=256,
        num_inference_steps=1,
        attention_mode="baseline",
    )

    outputs = generator.generate("a red cube", n=1)

    assert outputs == [str(expected_output / "img_0.jpg")]
    assert calls[0][1] == str(flux_repo)


def test_fusion_generator_keeps_flux_first_metadata() -> None:
    flux = MockImageGenerator(existing_paths=["/tmp/flux_a.png"], prefix="flux_mock")
    sdxl = MockImageGenerator(existing_paths=["/tmp/sdxl_a.png"], prefix="sdxl_mock")
    generator = FusionImageGenerator(flux=flux, sdxl=sdxl, policy="parallel")

    outputs = generator.generate("a cyan cat holding a red umbrella handle", n=1)

    assert outputs == ["/tmp/flux_a.png", "/tmp/sdxl_a.png"]
    assert [item["backend"] for item in generator.last_metadata] == ["flux", "sdxl"]
    assert generator.last_metadata[0]["prompt"] == "a cyan cat holding a red umbrella handle"
    assert flux.calls[0]["n"] == 1
    assert sdxl.calls[0]["n"] == 1
