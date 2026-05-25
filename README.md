# 🥊 Pack-a-Punch

![copertina](public/images/rdm1.png)

> **Italian BERT Binary Classification System**  
> *Classificatore AI ad alte prestazioni ottimizzato per la lingua italiana (Formazione vs Implementazione).*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-44CC11?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE.md)

**Pack-a-Punch** is a robust binary classification system designed to distinguish between **Formazione** and **Implementazione** in textual descriptions. Built on top of `dbmdz/bert-base-italian-xxl-cased`, it leverages **PyTorch** with CUDA acceleration for high-throughput production environments.

> 🍎 **Apple Users**: Please switch to the `apple-branch` for optimizations specific to macOS and Apple Silicon (M1/M2/M3) devices.

## ✨ Key Features

- **🚀 High Performance**: Optimized PyTorch inference pipeline for consumer GPUs.
- **🇮🇹 Italian Optimized**: Fine-tuned on `dbmdz/bert-base-italian-xxl-cased` for superior understanding of Italian context.
- **🧠 Knowledge Distillation**: Built-in pipeline to distill knowledge from large LLMs (via LM Studio) into a compact, efficient classifier.
- **🐳 Production Ready**: Fully containerized with Docker and NVIDIA Container Toolkit support.
- **⚙️ Type-Safe Config**: Robust configuration management using `pydantic-settings` with environment variable overrides.

---

## 🚀 Quick Start

The fastest way to get up and running is via Docker.

### Prerequisites

- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **NVIDIA GPU** with CUDA support (e.g., RTX 5070 Ti, RTX 3060 Ti).

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
      "Corso di formazione per dipendenti sull'uso di nuovi software aziendali.",
      "Sviluppo e implementazione di un nuovo sistema gestionale integrato ERP."
    ]
  }'
```

---

## ⚙️ Installation and Configuration

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
| `PAP_TRAINING__BATCH_SIZE` | Training Batch Size | `32` |
| `PAP_TRAINING__NUM_EPOCHS` | Training Epochs | `4` |

---

## 🔧 Training & Development

Pack-a-Punch supports multiple training modes.

### Option 1: Dataset Training
To train on the new dataset, first distribute the CSV files:

```bash
python scripts/distribute_data.py
```

This will extract the descriptions from `public/trainingset.csv` and `public/testset.csv` and place them in the correct directories:
- `src/data/formazione/*.txt`
- `src/data/implementazione/*.txt`
- `../data/Test/formazione/*.txt`
- `../data/Test/implementazione/*.txt`

Then run the training script via Docker:

```bash
docker compose -f docker/docker-compose.yml run --rm trainer
```
Or locally:
```bash
python scripts/train.py --data-source txt
```

### Option 2: Knowledge Distillation
Train by distilling knowledge from a larger Teacher LLM (e.g., via LM Studio).

1. Start your Local LLM server (compatible with OpenAI API).
2. Run the distillation training:

```bash
docker compose -f docker/docker-compose.yml run --rm distiller
```

### Benchmarking

Test inference performance:

```bash
# Start PyTorch backend 
docker compose -f docker/docker-compose.yml up classifier
```

Run benchmarks:

```bash
# PyTorch CUDA benchmark
python scripts/benchmark.py --url http://localhost:8080 --num-samples 10000 --batch-size 64 --concurrent-requests 10

# Pure serial latency (no concurrent overhead)
python scripts/benchmark.py --url http://localhost:8080 --num-samples 10000 --batch-size 64 --concurrent-requests 1
```

> **Note**: `--concurrent-requests` simulates multiple HTTP clients. Requests are queued and processed sequentially on GPU.

---

## 📂 Project Structure

```bash
Pack-a-Punch/
├── config/            # ⚙️ Configuration files (model_config.yml)
├── docker/            # 🐳 Docker configurations
├── scripts/           # 💻 CLI entrypoints (train, serve, distribute_data)
├── public/            # 📊 Source datasets and images
├── src/
│   ├── config.py      # ⚙️ Pydantic configuration settings
│   ├── data/          # 💾 Raw training data (distributed texts)
│   ├── inference/     # ⚡️ Inference engine & logic
│   ├── models/        # 📦 Saved model artifacts (.pt)
│   ├── training/      # 🏋️ Training pipeline & distillation
│   └── serve.py       # 🔌 FastAPI application (Internal)
└── tests/             # 🧪 Pytest suite
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
