# 🥊 Pack-a-Punch

> **ModernBERT Binary Classification System**  
> *High-performance AI text detection optimized for Italian language.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-GPU-orange)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE.md)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

**Pack-a-Punch** is a robust binary classification system designed to distinguish between **AI-generated** and **Human-written** text. Built on top of `Italian-ModernBERT`, it leverages **ONNX Runtime** with CUDA acceleration for ultra-low latency inference, making it suitable for high-throughput production environments.

## ✨ Key Features

- **🚀 High Performance**: Optimized ONNX Runtime inference pipeline delivering ~110 req/sec on consumer GPUs.
- **🇮🇹 Italian Optimized**: Fine-tuned on `DeepMount00/Italian-ModernBERT-base` for superior understanding of Italian context.
- **🍏 Apple Silicon Ready**: Native support for MPS (Metal Performance Shaders) and CoreML for high-performance training and inference on macOS.
- **🧠 Knowledge Distillation**: Built-in pipeline to distill knowledge from large LLMs (via LM Studio) into a compact, efficient classifier.
- **🐳 Production Ready**: Fully containerized with Docker and NVIDIA Container Toolkit support.
- **⚙️ Type-Safe Config**: Robust configuration management using `pydantic-settings` with environment variable overrides.

---

## 🚀 Quick Start

The fastest way to get up and running is via Docker.

### Prerequisites

- **Linux**: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and NVIDIA GPU.
- **macOS**: Apple Silicon (M1/M2/M3) for MPS acceleration.
- **Python 3.10+** (if running locally).

### Run Inference Server (Docker)

```bash
# Start the classifier service (Defaults to ONNX)
docker compose -f docker/docker-compose.yml up classifier
```

### Run Inference Server (Local)

By default, the server uses **ONNX Runtime**. To use **PyTorch (with MPS/CUDA support)**, set the `INFERENCE_BACKEND` environment variable:

```bash
# Run with PyTorch (recommended for Mac MPS testing)
INFERENCE_BACKEND=pytorch python scripts/serve.py
```

> **Why ONNX by default?**  
> In `src/inference/server.py`, the backend is determined by:  
> `BACKEND = os.environ.get("INFERENCE_BACKEND", "onnx").lower()`  
> If not specified, it assumes `onnx` for production efficiency.

### API Usage Example

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "L'intelligenza artificiale sta rivoluzionando il mondo.",
      "Oggi sono andato al mercato e ho comprato le mele."
    ]
  }'
```

---

## ## ⚙️ Installation and Configuration

If you prefer running without Docker, you can install the dependencies locally.

```bash
# Clone the repository
git clone https://github.com/yourusername/Pack-a-Punch.git
cd Pack-a-Punch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For macOS Apple Silicon (optimized ONNX)
pip install onnxruntime-silicon
```

The application is configured via `src/config.py`. You can override any setting using environment variables with the prefix `PAP_`. Double underscores `__` denote nested configs.

**Common Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `PAP_SERVER__PORT` | API Server Port | `8080` |
| `PAP_INFERENCE__DEVICE` | Device (`cuda`, `mps`, `cpu`) | `mps` (on Mac) |
| `PAP_INFERENCE__BATCH_SIZE` | Inference Batch Size | `64` |
| `PAP_INFERENCE__NUM_SESSIONS` | Parallel ONNX Sessions | `2` |
| `PAP_TRAINING__BATCH_SIZE` | Training Batch Size | `32` |
| `PAP_TRAINING__NUM_EPOCHS` | Training Epochs | `3` |

---

## 🔧 Training & Development

Pack-a-Punch supports multiple training modes.

### Option 1: Dataset Training
Place your labeled txt files in the data directory:
- `src/data/ai/*.txt`
- `src/data/non_ai/*.txt`

Then run the training script:

```bash
python scripts/train.py --data-source txt
```

### Option 2: Knowledge Distillation
Train by distilling knowledge from a larger Teacher LLM (e.g., via LM Studio).

1. Start your Local LLM server (compatible with OpenAI API).
2. Run the distillation training:

```bash
python scripts/train.py \
  --data-source distillation \
  --teacher-url http://localhost:1234/v1/chat/completions
```

### Benchmarking

Test inference performance comparing ONNX Runtime vs PyTorch backends:

```bash
# Start PyTorch backend (porta 8081)
docker compose -f docker/docker-compose.yml --profile pytorch up classifier-pytorch

# Start ONNX backend (porta 8080)
docker compose -f docker/docker-compose.yml up classifier
```

Run performance benchmarks:

```bash
# Standard performance benchmark (synthetic data)
python scripts/benchmark.py --num-samples 1000 --batch-size 64

# Evaluation on real test set (accuracy + performance)
# 1. Ensure the server is running (e.g., with PyTorch)
#    INFERENCE_BACKEND=pytorch python scripts/serve.py
# 2. Run the benchmark against the server URL
python scripts/benchmark_test_set.py --url http://localhost:8080
```

> **Note**: `--concurrent-requests` simulates multiple HTTP clients. Requests are queued and processed sequentially on GPU/MPS.

---

## 📂 Project Structure

```bash
Pack-a-Punch/
├── docker/            # 🐳 Docker configurations
├── scripts/           # � CLI entrypoints (train, serve, benchmark)
├── src/
│   ├── config.py      # ⚙️ Pydantic configuration settings
│   ├── data/          # 💾 Raw training data
│   ├── inference/     # ⚡️ ONNX Runtime engine & logic
│   ├── models/        # 📦 Saved model artifacts (.pt, .onnx)
│   ├── training/      # 🏋️ Training pipeline & distillation
│   └── serve.py       # 🔌 FastAPI application (Internal)
└── tests/             # 🧪 Pytest suite
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
