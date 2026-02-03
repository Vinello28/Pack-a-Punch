# Pack-a-Punch: ModernBERT Binary Classification System

Sistema di classificazione binaria (AI/NON-AI) basato su Italian ModernBERT con inferenza ad alte prestazioni via ONNX Runtime.

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (tested on RTX 3060 Ti)

### Run Inference Server
```bash
docker-compose up classifier
```

### API Usage
```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Testo da classificare"]}'
```

## 📁 Project Structure

```
Pack-a-Punch/
├── src/
│   ├── data/          # Training datasets
│   ├── models/        # Saved models (PyTorch + ONNX)
│   ├── training/      # Training pipeline
│   ├── inference/     # Inference engine & API
│   └── config.py      # Configuration
├── scripts/           # CLI entrypoints
├── docker/            # Docker configuration
└── tests/             # Test suite
```

## 🔧 Training

### Option 1: From labeled TXT files
Place files in `src/data/ai/` and `src/data/non_ai/`:
```bash
python scripts/train.py --data-source txt
```

### Option 2: Distillation from LLM Teacher
Start LM Studio with your model, then:
```bash
python scripts/train.py --data-source distillation --teacher-url http://localhost:1234
```

## 📊 Performance

| Mode | Throughput | VRAM | Time (20M records) |
|------|------------|------|-------------------|
| Single session | ~3500/sec | 2GB | ~1.5h |
| Dual parallel | ~6000/sec | 4GB | ~55min |

## 📄 License

See [LICENSE.md](LICENSE.md)
