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
To test the inference performance on your hardware:

```bash
python scripts/benchmark.py --url http://localhost:8080 --num-samples 2000 --batch-size 64 --concurrent-requests 10
```

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
