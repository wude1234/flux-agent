# GitHub Upload Guide

This project snapshot is meant to publish the code, docs, and tests only.
Generated runs, caches, logs, and model weights should stay local.

## Keep

- `src/`
- `scripts/`
- `tests/`
- `docs/`
- `notes/`
- `README.md`
- `PLAN.md`
- `environment-m0.yml`
- `infer_mgrag_flux.py`

## Do not upload by default

- `runs_mini_benchmark/`
- `local_reports/`
- `.conda-pkgs/`
- `.pytest_cache/`
- `__pycache__/`
- `benchmarks/`
- model weights: `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`, `*.onnx`
- log files: `*.log`

## Recommended upload flow

```bash
cd /home/zrr/t2i_agent_papers_2024_2025/mult-t2i-agent/project_flux

# If this directory is not a git repo yet
git init
git branch -M main

# Set your GitHub remote
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
# If origin already exists, use:
# git remote set-url origin git@github.com:YOUR_NAME/YOUR_REPO.git

# Review what will be committed
git status --short

# Add code-only files
git add .gitignore README.md PLAN.md environment-m0.yml infer_mgrag_flux.py
git add src scripts tests docs notes

# Double-check
git status --short

# Commit and push
git commit -m "Snapshot FLUX agent code"
git push -u origin main
```

## Optional: include benchmark JSONs

If you later want to publish the small benchmark prompt sets too, remove
`benchmarks/` from `.gitignore` and add that directory explicitly.
