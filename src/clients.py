"""Mockable LLM and VLM client interfaces."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import request
from urllib.error import HTTPError, URLError


class LLMClient(Protocol):
    """Text-only language model adapter."""

    def text(self, prompt: str) -> str:
        ...


class VLMClient(Protocol):
    """Vision-language model adapter."""

    def vision(self, prompt: str, image_paths: list[str]) -> str:
        ...


JsonTransport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


@dataclass
class MockLLMClient:
    """Deterministic text client used by tests and M0 smoke runs."""

    responses: Sequence[str] = ()
    default_response: str = "mock text response"
    calls: list[str] = field(default_factory=list)
    _cursor: int = 0

    def text(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.calls.append(prompt)
        if self.responses:
            index = min(self._cursor, len(self.responses) - 1)
            self._cursor += 1
            return self.responses[index]
        return self.default_response


@dataclass
class MockVLMClient:
    """Deterministic vision client that never opens images or calls an API."""

    responses: Sequence[str] = ()
    default_response: str = "mock vision response"
    calls: list[dict[str, object]] = field(default_factory=list)
    _cursor: int = 0

    def vision(self, prompt: str, image_paths: list[str]) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list of strings")
        if any(not isinstance(path, str) or not path for path in image_paths):
            raise ValueError("image_paths must contain non-empty strings")
        self.calls.append({"prompt": prompt, "image_paths": deepcopy(image_paths)})
        if self.responses:
            index = min(self._cursor, len(self.responses) - 1)
            self._cursor += 1
            return self.responses[index]
        return self.default_response


@dataclass
class OpenAICompatibleLLMClient:
    """Text client for OpenAI-compatible chat-completions APIs.

    This adapter is intentionally generic. For DashScope/Bailian compatible
    mode, use base_url ``https://dashscope.aliyuncs.com/compatible-mode/v1``.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 1.0
    transport: JsonTransport | None = None
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = _clean_text(self.model, "model")
        self.base_url = _clean_text(self.base_url, "base_url").rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def text(self, prompt: str) -> str:
        prompt = _clean_text(prompt, "prompt")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        self.calls.append(prompt)
        data = self._post("/chat/completions", payload)
        return _extract_chat_content(data)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = _auth_headers(self.api_key)
        transport = self.transport or _urlopen_json
        return _post_with_retries(
            transport,
            f"{self.base_url}{path}",
            headers,
            payload,
            self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )


@dataclass
class OpenAICompatibleVLMClient:
    """Vision client for OpenAI-compatible multimodal chat-completions APIs."""

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    timeout: float = 120.0
    max_retries: int = 2
    retry_backoff: float = 1.0
    max_image_data_uri_bytes: int = 96_000
    image_preprocess_max_side: int = 768
    image_preprocess_min_quality: int = 45
    transport: JsonTransport | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = _clean_text(self.model, "model")
        self.base_url = _clean_text(self.base_url, "base_url").rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def vision(self, prompt: str, image_paths: list[str]) -> str:
        prompt = _clean_text(prompt, "prompt")
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list of strings")
        image_paths = [_clean_text(path, "image_path") for path in image_paths]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_payloads: list[dict[str, Any]] = []
        for image_path in image_paths:
            image_payload = _image_url_or_data_uri(
                image_path,
                max_data_uri_bytes=self.max_image_data_uri_bytes,
                max_side=self.image_preprocess_max_side,
                min_quality=self.image_preprocess_min_quality,
            )
            image_payloads.append(
                {
                    key: value
                    for key, value in image_payload.items()
                    if key != "url"
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_payload["url"]},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": deepcopy(image_paths),
                "image_payloads": deepcopy(image_payloads),
            }
        )
        data = self._post("/chat/completions", payload)
        return _extract_chat_content(data)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = _auth_headers(self.api_key)
        transport = self.transport or _urlopen_json
        return _post_with_retries(
            transport,
            f"{self.base_url}{path}",
            headers,
            payload,
            self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )


def client_from_env(
    *,
    kind: str,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.2,
) -> LLMClient | VLMClient:
    """Build an API client from an environment variable without logging the key."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"environment variable {api_key_env} is not set")
    if kind == "llm":
        return OpenAICompatibleLLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
    if kind == "vlm":
        return OpenAICompatibleVLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
    raise ValueError("kind must be 'llm' or 'vlm'")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        raise RuntimeError("api_key is required for real API clients")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _urlopen_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"API connection error: {exc.reason}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise RuntimeError("API response JSON must be an object")
    return parsed


def _post_with_retries(
    transport: JsonTransport,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
    *,
    max_retries: int,
    retry_backoff: float,
) -> Mapping[str, Any]:
    attempts = max(1, int(max_retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return transport(url, headers, payload, timeout)
        except RuntimeError as exc:
            if not _is_retryable_api_error(exc) or attempt >= attempts - 1:
                raise
            last_error = exc
            time.sleep(max(0.0, float(retry_backoff)) * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("API request failed without an exception")


def _is_retryable_api_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    retryable_needles = (
        "api connection error",
        "timed out",
        "timeout",
        "ssl",
        "temporarily unavailable",
        "connection reset",
        "remote end closed",
        "502",
        "503",
        "504",
        "rate limit",
    )
    return any(needle in text for needle in retryable_needles)


def _extract_chat_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("API response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("API response choice is not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("API response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise RuntimeError("API response message content is empty or unsupported")


def _image_url_or_data_uri(
    value: str,
    *,
    max_data_uri_bytes: int | None = None,
    max_side: int = 768,
    min_quality: int = 45,
) -> dict[str, Any]:
    if value.startswith(("http://", "https://", "data:")):
        return {
            "url": value,
            "source": "passthrough_url",
            "bytes": len(value.encode("utf-8")),
            "compressed": False,
        }
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"image path does not exist: {value}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    original_bytes = path.read_bytes()
    original_url = _data_uri(mime_type, original_bytes)
    max_bytes = int(max_data_uri_bytes or 0)
    if max_bytes <= 0 or len(original_url.encode("utf-8")) <= max_bytes:
        return {
            "url": original_url,
            "source": str(path),
            "mime_type": mime_type,
            "bytes": len(original_url.encode("utf-8")),
            "original_bytes": len(original_bytes),
            "compressed": False,
        }
    compressed = _compressed_image_data_uri(
        path,
        max_data_uri_bytes=max_bytes,
        max_side=max_side,
        min_quality=min_quality,
    )
    compressed.update(
        {
            "source": str(path),
            "original_mime_type": mime_type,
            "original_bytes": len(original_bytes),
            "original_data_uri_bytes": len(original_url.encode("utf-8")),
            "compressed": True,
        }
    )
    return compressed


def _data_uri(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _compressed_image_data_uri(
    path: Path,
    *,
    max_data_uri_bytes: int,
    max_side: int,
    min_quality: int,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except Exception:
        data = path.read_bytes()
        url = _data_uri(mimetypes.guess_type(path.name)[0] or "image/png", data)
        return {
            "url": url,
            "mime_type": mimetypes.guess_type(path.name)[0] or "image/png",
            "bytes": len(url.encode("utf-8")),
            "width": None,
            "height": None,
            "quality": None,
            "preprocess_error": "PIL is not available; sent original image",
        }

    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = image.size
        max_side = max(64, int(max_side or 768))
        min_quality = min(max(20, int(min_quality or 45)), 95)
        image.thumbnail((max_side, max_side))
        working = image.copy()

    best: dict[str, Any] | None = None
    quality = 85
    while quality >= min_quality:
        data = _jpeg_bytes(working, quality=quality)
        url = _data_uri("image/jpeg", data)
        record = {
            "url": url,
            "mime_type": "image/jpeg",
            "bytes": len(url.encode("utf-8")),
            "width": working.size[0],
            "height": working.size[1],
            "quality": quality,
            "original_size": [original_size[0], original_size[1]],
        }
        best = record
        if record["bytes"] <= max_data_uri_bytes:
            return record
        quality -= 10

    while best is not None and best["bytes"] > max_data_uri_bytes and max(working.size) > 160:
        next_size = (
            max(1, int(working.size[0] * 0.82)),
            max(1, int(working.size[1] * 0.82)),
        )
        working = working.resize(next_size)
        data = _jpeg_bytes(working, quality=min_quality)
        url = _data_uri("image/jpeg", data)
        best = {
            "url": url,
            "mime_type": "image/jpeg",
            "bytes": len(url.encode("utf-8")),
            "width": working.size[0],
            "height": working.size[1],
            "quality": min_quality,
            "original_size": [original_size[0], original_size[1]],
        }
        if best["bytes"] <= max_data_uri_bytes:
            return best
    if best is None:
        raise RuntimeError("image compression failed")
    best["warning"] = "compressed image still exceeds target payload budget"
    return best


def _jpeg_bytes(image: Any, *, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return buffer.getvalue()


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value
