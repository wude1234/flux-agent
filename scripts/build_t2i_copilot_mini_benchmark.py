from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVAL_ROOT = REPO_ROOT / "code" / "T2I-Copilot-master" / "eval_benchmark"
OUT_DIR = PROJECT_ROOT / "benchmarks"

GENAI_SKILL_CATEGORIES = [
    "attribute",
    "scene",
    "spatial relation",
    "action relation",
    "part relation",
    "counting",
    "comparison",
    "differentiation",
    "negation",
    "universal",
]


def normalize_category(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def build_drawbench_cases(per_category: int) -> list[dict[str, Any]]:
    path = EVAL_ROOT / "DrawBench_seed.txt"
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Unexpected DrawBench row {line_number}: {line!r}")
        prompt, seed, category = parts
        category_key = normalize_category(category)
        by_category[category_key].append(
            {
                "id": f"drawbench_{category_key}_{len(by_category[category_key]) + 1:03d}",
                "category": f"drawbench_{category_key}",
                "prompt": prompt.strip(),
                "focus": [category_key, "drawbench"],
                "source": "DrawBench",
                "source_category": category.strip(),
                "source_index": line_number - 1,
                "seed": int(seed),
            }
        )
    cases: list[dict[str, Any]] = []
    for category_key in sorted(by_category):
        cases.extend(by_category[category_key][:per_category])
    return cases


def build_genai_cases(per_category: int) -> list[dict[str, Any]]:
    root = EVAL_ROOT / "GenAIBenchmark"
    images = json.loads((root / "genai_image_seed.json").read_text(encoding="utf-8"))
    skills = json.loads((root / "genai_skills.json").read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    used_prompts: set[str] = set()

    for skill in GENAI_SKILL_CATEGORIES:
        category_key = normalize_category(skill)
        picked = 0
        for raw_index in skills[skill]:
            source_id = f"{int(raw_index):05d}"
            item = images[source_id]
            prompt = str(item["prompt"]).strip()
            # Keep categories distinct while avoiding duplicate prompts inside
            # the combined mini benchmark.
            prompt_key = prompt.casefold()
            if prompt_key in used_prompts:
                continue
            used_prompts.add(prompt_key)
            picked += 1
            cases.append(
                {
                    "id": f"genai_{category_key}_{picked:03d}",
                    "category": f"genai_{category_key}",
                    "prompt": prompt,
                    "focus": [category_key, "genai_bench"],
                    "source": "GenAI-Bench",
                    "source_category": skill,
                    "source_id": source_id,
                    "seed": int(item.get("random_seed", 7100 + int(raw_index))),
                }
            )
            if picked >= per_category:
                break
        if picked < per_category:
            raise ValueError(
                f"Only found {picked} unique prompts for GenAI-Bench skill {skill!r}"
            )
    return cases


def payload(version: str, description: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["category"] for case in cases)
    return {
        "version": version,
        "description": description,
        "categories": dict(sorted(counts.items())),
        "anti_overfit_policy": {
            "source": "T2I-Copilot paper-aligned benchmark files",
            "per_category_target": 5,
            "sampling": "first deterministic prompts per source category, preserving source seeds",
        },
        "cases": cases,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    per_category = 5
    drawbench_cases = build_drawbench_cases(per_category)
    genai_cases = build_genai_cases(per_category)
    combined_cases = [*drawbench_cases, *genai_cases]

    drawbench_payload = payload(
        "t2i_copilot_drawbench_5each_v1",
        "DrawBench mini benchmark sampled from T2I-Copilot eval_benchmark/DrawBench_seed.txt.",
        drawbench_cases,
    )
    genai_payload = payload(
        "t2i_copilot_genai_bench_5each_v1",
        "GenAI-Bench mini benchmark sampled from T2I-Copilot eval_benchmark/GenAIBenchmark by skill.",
        genai_cases,
    )
    combined_payload = payload(
        "t2i_copilot_paper_aligned_5each_v1",
        "Combined DrawBench and GenAI-Bench mini benchmark aligned with T2I-Copilot evaluation sources.",
        combined_cases,
    )

    drawbench_path = OUT_DIR / "t2i_copilot_drawbench_5each.json"
    genai_path = OUT_DIR / "t2i_copilot_genai_bench_5each.json"
    combined_path = OUT_DIR / "t2i_copilot_paper_aligned_5each.json"
    write_json(drawbench_path, drawbench_payload)
    write_json(genai_path, genai_payload)
    write_json(combined_path, combined_payload)

    print(
        json.dumps(
            {
                "drawbench": str(drawbench_path),
                "drawbench_cases": len(drawbench_cases),
                "genai_bench": str(genai_path),
                "genai_bench_cases": len(genai_cases),
                "combined": str(combined_path),
                "combined_cases": len(combined_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
