# 🥊 Pack-a-Punch

![copertina](public/images/rdm1.png)

> **Italian BERT Binary Classification System**  
> *High-performance AI text detection optimized for Italian language.*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-666666?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Fedora](https://img.shields.io/badge/Fedora-51A2DA?style=for-the-badge&logo=fedora&logoColor=white)](https://getfedora.org/)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-44CC11?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE.md)

**Pack-a-Punch** is a robust binary classification system designed to distinguish between **AI-generated** and **Human-written** text. Built on top of `dbmdz/bert-base-italian-xxl-cased`, it leverages **ONNX Runtime** with CUDA acceleration for ultra-low latency inference, making it suitable for high-throughput production environments.

> 🍎 **Apple Users**: Please switch to the `apple-branch` for optimizations specific to macOS and Apple Silicon (M1/M2/M3) devices.

## ✨ Key Features

- **🚀 High Performance**: Optimized ONNX Runtime inference pipeline delivering ~110 req/sec on consumer GPUs.
- **🇮🇹 Italian Optimized**: Fine-tuned on `dbmdz/bert-base-italian-xxl-cased` for superior understanding of Italian context.
- **🧠 Knowledge Distillation**: Built-in pipeline to distill knowledge from large LLMs (via LM Studio) into a compact, efficient classifier.
- **🐳 Production Ready**: Fully containerized with Docker and NVIDIA Container Toolkit support.
- **⚙️ Type-Safe Config**: Robust configuration management using `pydantic-settings` with environment variable overrides.

---

## 🚀 Quick Start

The fastest way to get up and running is via Docker.

### Prerequisites

- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **NVIDIA GPU** with CUDA support (Tested on RTX 3060 Ti).

### Run Inference Server

```bash
# Start the classifier service
docker compose -f docker/docker-compose.yml up classifier
```

The API will be available at `http://localhost:8080`.

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
```

The application is configured via `src/config.py`. You can override any setting using environment variables with the prefix `PAP_`. Double underscores `__` denote nested configs.

**Common Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `PAP_SERVER__PORT` | API Server Port | `8080` |
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

Run benchmarks:

```bash
# ONNX Runtime benchmark (default, optimized)
python scripts/benchmark.py --url http://localhost:8080 --num-samples 10000 --batch-size 64 --concurrent-requests 10

# PyTorch CUDA benchmark
python scripts/benchmark.py --url http://localhost:8081 --num-samples 10000 --batch-size 64 --concurrent-requests 10

# Pure serial latency (no concurrent overhead)
python scripts/benchmark.py --url http://localhost:8081 --num-samples 10000 --batch-size 64 --concurrent-requests 1
```

> **Note**: `--concurrent-requests` simulates multiple HTTP clients. Requests are queued and processed sequentially on GPU.

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
