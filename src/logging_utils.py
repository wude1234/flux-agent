"""Run-directory logging helpers for the unified M4 agent loop."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


def make_run_id() -> str:
    """Return a timestamp run id suitable for ``project/runs/<timestamp>``."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_run_dir(
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    *,
    run_id: str | None = None,
) -> Path:
    """Create and return a unique run directory."""

    base = Path(runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    stem = _clean_run_id(run_id or make_run_id())
    path = base / stem
    if not path.exists():
        path.mkdir(parents=True)
        return path

    for index in range(1, 1000):
        candidate = base / f"{stem}-{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise FileExistsError(f"could not allocate run directory for {stem}")


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Write JSON with stable indentation and UTF-8 encoding."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    _atomic_write_text(output_path, payload, encoding="utf-8")
    return output_path


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write text so interrupted runs do not corrupt prior artifacts."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output_path, text, encoding=encoding)
    return output_path


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding=encoding) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_state_snapshot(
    run_dir: str | Path,
    round_index: int,
    *,
    state: Mapping[str, Any],
    round_record: Mapping[str, Any],
    status: str,
) -> Path:
    """Write ``state_round_<n>.json`` for one orchestration round."""

    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    return write_json(
        Path(run_dir) / f"state_round_{round_index}.json",
        {
            "round": round_index,
            "status": status,
            "state": dict(state),
            "round_record": dict(round_record),
        },
    )


def write_final_report(
    run_dir: str | Path,
    *,
    run_id: str,
    status: str,
    mode: str,
    user_prompt: str,
    final_prompt: str,
    final_score: float | None,
    selected_image: str | None,
    round_records: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a compact Markdown report for a completed or paused run."""

    report = build_final_report(
        run_id=run_id,
        status=status,
        mode=mode,
        user_prompt=user_prompt,
        final_prompt=final_prompt,
        final_score=final_score,
        selected_image=selected_image,
        round_records=round_records,
    )
    path = Path(run_dir) / "final_report.md"
    atomic_write_text(path, report, encoding="utf-8")
    return path


def build_final_report(
    *,
    run_id: str,
    status: str,
    mode: str,
    user_prompt: str,
    final_prompt: str,
    final_score: float | None,
    selected_image: str | None,
    round_records: Sequence[Mapping[str, Any]],
) -> str:
    score_text = "n/a" if final_score is None else f"{final_score:.3f}"
    image_text = selected_image or "n/a"
    lines = [
        "# Multimodal T2I Agent Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Status: `{status}`",
        f"- Mode: `{mode}`",
        f"- User prompt: {user_prompt}",
        f"- Final prompt: {final_prompt}",
        f"- Final score: {score_text}",
        f"- Selected image: `{image_text}`",
        "",
        "## Rounds",
    ]
    if not round_records:
        lines.append("")
        lines.append("No generation rounds were completed.")
        return "\n".join(lines) + "\n"

    for record in round_records:
        feedback = record.get("feedback", {})
        score = _feedback_score(feedback)
        score_suffix = "" if score is None else f" score={score:.3f}"
        lines.extend(
            [
                "",
                f"### Round {record.get('round', 'n/a')}{score_suffix}",
                f"- Prompt: {record.get('prompt', '')}",
                f"- Selected image: `{record.get('selected_image', 'n/a')}`",
                f"- Revision hint: {_revision_hint(feedback)}",
                f"- Revised prompt: {record.get('revised_prompt', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _clean_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or make_run_id()


def _feedback_score(feedback: Any) -> float | None:
    if not isinstance(feedback, Mapping):
        return None
    try:
        return float(feedback.get("score"))
    except (TypeError, ValueError):
        return None


def _revision_hint(feedback: Any) -> str:
    if not isinstance(feedback, Mapping):
        return str(feedback or "")
    return str(feedback.get("revision_hint") or feedback.get("reason") or "")
