# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pack-a-Punch is a multi-class classifier that identifies the AI application sector of texts. It fine-tunes `answerdotai/ModernBERT-base` (7 classes) and serves predictions via a FastAPI server backed by ONNX Runtime with CUDA acceleration.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train from labeled .txt files (src/data/<class_slug>/*.txt)
python scripts/train.py --data-source txt

# Train from CSV file
python scripts/train.py --data-source csv --csv-path public/modernbert_final.csv

# Train with K-Fold cross validation
python scripts/train.py --data-source txt --kfold --kfold-splits 5

# Train via knowledge distillation from a local LLM
python scripts/train.py --data-source distillation --teacher-url http://localhost:1234/v1/chat/completions

# Export trained model to ONNX (standard or Optimum with kernel fusion)
python scripts/train.py --data-source txt --export-onnx
python scripts/train.py --data-source txt --export-optimum

# Start inference server
python scripts/serve.py                          # default port 8080
python scripts/serve.py --port 8080 --reload     # with hot-reload

# Run via Docker
docker compose -f docker/docker-compose.yml up classifier          # ONNX backend (port 8080)
docker compose -f docker/docker-compose.yml --profile pytorch up   # PyTorch backend (port 8081)

# Benchmark inference
python scripts/benchmark.py --url http://localhost:8080 --num-samples 10000 --batch-size 64 --concurrent-requests 10

# Run tests
pytest
pytest tests/test_inference.py           # single file
pytest tests/test_inference.py -k "test_tokenize"  # single test

# Lint & format
ruff check .
black .
```

## Architecture

### Configuration flow
All settings originate in `config/model_config.yml`, are loaded by `src/config_loader.py`, and exposed as typed Pydantic models in `src/config.py`. The global `settings` singleton is used everywhere. Environment variables with prefix `PAP_` and double-underscore nesting (e.g. `PAP_INFERENCE__BATCH_SIZE`) override YAML values.

### Training pipeline (`src/training/`)
- `trainer.py` — main Trainer class; supports both simple train/eval split and stratified K-Fold CV
- `dataset.py` — loads data from `.txt` directories, `.jsonl`, `.csv` files, or distilled output
- `distillation.py` — queries a teacher LLM (OpenAI-compatible API) to generate labeled training data
- `export_onnx.py` / `export_optimum.py` — two paths to convert PyTorch model to ONNX

### Inference pipeline (`src/inference/`)
- `engine.py` — ONNX Runtime inference engine with session pooling (`num_sessions`)
- `pytorch_engine.py` — alternative PyTorch CUDA backend (selected via `INFERENCE_BACKEND` env var)
- `batching.py` — async request batching with configurable queue and timeout
- `server.py` — FastAPI app with `/classify` and `/health` endpoints

### Scripts (`scripts/`)
CLI entrypoints: `train.py`, `serve.py`, `benchmark.py`, `benchmark_quality.py`, `distribute_data.py`, `generate_synthetic.py`, `generate_concepts.py`, `generate_expanded_dataset.py`. All add project root to `sys.path` manually.

### Docker
`docker/docker-compose.yml` defines three main services: `classifier` (ONNX, port 8080), `classifier-pytorch` (PyTorch backend, port 8081), and `trainer`. Non-default services require `--profile` flags (`pytorch` for PyTorch backend, `training` for training service). All GPU services require NVIDIA Container Toolkit.

#### DGX-A100 Compatibility
Docker Compose config is optimized for DGX-A100 with DGX OS (Ubuntu 20.04):
- **GPU pinning**: `device_ids: ['5']` — restrict all containers to GPU 5 (modify as needed)
- **Resource limits**: `cpus: 32`, `shm_size: '4g'` — prevent resource exhaustion and PyTorch DataLoader crashes
- **Requirements**: `docker compose` (v2 plugin) for `device_ids` support; if using old docker-compose v1, upgrade via `pip3 install docker-compose>=1.29`

## Key Details

- **Apple Silicon**: use the `apple-branch` git branch for macOS/Metal optimizations
- **GPU ID**: currently pinned to GPU 5 in `docker-compose.yml`. To change, edit `device_ids: ['5']` in the `deploy` section of each service
- **Formatting**: Black with 100-char line length; Ruff for linting (rules: E, F, I, N, W, UP; E501 ignored)
- **Testing**: pytest with `asyncio_mode = "auto"`; test paths under `tests/`
- **Training data**: plain `.txt` files in `src/data/<class_slug>/` directories (e.g. `ai_research/`, `data_science/`); test split in `src/data/Test/`
- **Model artifacts**: saved to `src/models/` (`.pt` and `.onnx` files)
- **Base model**: `answerdotai/ModernBERT-base` (eager mode, no Triton), max sequence length 3072, 7 labels

## Workflow Orchestration
### 1. Plan Node Default
-   Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
-   If something goes sideways, STOP and re-plan immediately - don't keep pushing
-   Use plan mode for verification steps, not just building
-   Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
-   Use subagents liberally to keep main contect window clean
-   Offload research, exploration, and parallel analysis to subagents
-   For complex problens, throw more compute at it via subagents
-   One tack per subagent for focused execution

### 3. Self-Improvement Loop
-   After ANY correction from the user: update 'tasks/lessons.md' with the pattern
-   Write rules for yourself that prevent the same mistake
-   Ruthlessly iterate on these lessons until mistake rate drops
-   Review lessons at session start for relevant project

### 4. Verification Before Done
-   Never mark a task complete without proving it works
-   Diff behavior between main and your changes when relevant
-   Ask yourself: "Would a staff engineer approve this?"
-   Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
-   For non-trivial changes: pause and ask "is there a more elegant way?"
-   If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
-   Skip this for simple, chvious fixes - don't over-engineer
-   Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
-   When given a bug report: just fix it. Don't ask for hand-holding
-   Point at logs,errors, failing tests - then resolve them
-   Zero context switching required from the user
-   Go fix failing CI tests without being told how

## Task Management
1.    **PLan First**: Write plan to 'tasks/todo.md' with checkable items
2.    **Verify Plan**: Check in before starting implementation
3.    **Track Progress**: Mark items complete as you go
4.    **Explain Changes**: High-level summary at each step
5.    **Document Results**: Add review section to 'tasks/todo.md"
6.    **Capture Lessons**: Update 'tasks/lessons. md' after corrections

## Core Principles
-   **Simplicity First**: Make every change as simple as possible. Inpact minimal code.
-   **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
-   **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
