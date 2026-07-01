# Migrazione a NVIDIA Triton Inference Server

Obiettivo: aumentare il throughput servendo il modello con Triton (dynamic batching su GPU +
TensorRT), tenendo FastAPI come gateway sottile (tokenizzazione + softmax, API invariata).

## Implementazione
- [x] `model_repository/classifier/config.pbtxt` (backend onnxruntime + TensorRT accel + dynamic batching)
- [x] `scripts/export_triton.py` (export ONNX offline una volta sola, stampa input/output reali)
- [x] `src/inference/triton_engine.py` (client gRPC, stessa interfaccia degli altri engine, auto-adattante sugli input)
- [x] `src/inference/server.py` (swap import tensorrt_engine -> triton_engine, pulizia label AI/NON_AI)
- [x] `src/inference/__init__.py` (export TritonInferenceEngine / create_triton_engine)
- [x] `docker/docker-compose.yml` (nuovo servizio `triton`, GPU spostata, classifier -> TRITON_URL + depends_on)
- [x] `docker/Dockerfile.serve` (base python:3.11-slim, niente GPU/CUDA, + tritonclient[grpc])
- [x] `requirements.txt` (rimosso onnxruntime/tensorrt/optimum dal runtime, + tritonclient[grpc])
- [x] Syntax check Python (py_compile OK)

## Verifica (da eseguire su macchina con GPU)
- [ ] `python scripts/export_triton.py` -> genera `model_repository/classifier/1/model.onnx`;
      controllare gli input stampati e allineare `config.pbtxt` (probabile: solo input_ids/attention_mask)
- [ ] `docker compose -f docker/docker-compose.yml up --build triton classifier`
- [ ] `curl localhost:8000/v2/health/ready` -> 200 (Triton pronto)
- [ ] `curl localhost:8080/health` -> model_loaded:true, backend:"triton"
- [ ] `POST /classify` con frasi note -> confronto label/confidence con engine precedente (tolleranza FP16)
- [ ] `scripts/benchmark.py` -> confronto req/s e p95 vs baseline, alzando la concorrenza

## Note / rischi
- GPU reale: RTX 5070 Ti (Blackwell, sm_120), driver 595.71.05 (CUDA runtime fino a 13.2).
  Serve un Triton su CUDA 13 -> scelto `nvcr.io/nvidia/tritonserver:26.01-py3` (CUDA 13.1.1).
  Le tag CUDA-13 partono da `25.08-py3`; `26.06-py3` e' CUDA 13.3 (richiede driver >= 13.3).
- Prima inferenza: build del motore TensorRT lenta; opzionale abilitare la cache TRT in config.pbtxt.

## Review — verifica end-to-end OK (RTX 5070 Ti, 2026-06-28)
- Export: `model.onnx` (442MB, pesi inclusi); input reali `input_ids`/`attention_mask`,
  output `logits` -> combaciano con `config.pbtxt` (nessun token_type_ids, come previsto).
- `docker compose up`: `pap-triton` healthy (image 26.01-py3, CUDA 13.1.1), `pap-classifier`
  healthy. Bug trovato e risolto: `__init__.py` importava eagerly `pytorch_engine` (torch),
  assente nell'immagine slim -> reso lazy con `__getattr__`.
- `/health`: backend "triton", server_live true, model_ready true.
- Correttezza: "tracciabilità filiera blockchain" -> tracciabilita 0.998; "corso public
  speaking" -> altro 0.999. Coerente.
- Latenza a caldo ~0.10s/call (primo call ~12s = build TensorRT per shape, una tantum).
- Throughput a caldo: 800 classificazioni in 0.54s -> ~1483 cls/s, ~185 req/s (concorrenza 20).
- Dynamic batching confermato dalle stats Triton: esecuzioni accorpate in batch 80 e 160
  (`batch_stats`: (160,7),(80,1),(8,6),...), ~19 exec del modello per 100 richieste.

### Fix post-verifica: TensorRT EP -> CUDA EP (2026-06-28)
- Bug col workload reale del notebook (`traceability_classification.ipynb`, batch 512): 500
  Internal Server Error, "TensorRT EP failed to create engine / No engine is found".
- Causa: il TensorRT EP del backend onnxruntime ricostruisce un engine per ogni shape (batch +
  seq len dinamici) e sul batch 512x512 esauriva il workspace (1GB < 1.5GB richiesti).
- Fix: rimosso l'accel TensorRT da `config.pbtxt` -> ONNX Runtime **CUDA EP** (shape dinamiche
  in una sola sessione, niente rebuild). Triton dynamic batching invariato.
- Risultato: batch 512 OK; ~1265 rec/s su lunghezze variabili (0 errori). 929962 record ~12 min
  (prima: 24 rec/s + crash). Il notebook NON richiede modifiche.

### Fix #2: OOM su batch grandi + fp16 (2026-06-28)
- Sintomo nel notebook: 500 intermittenti + throughput ~466 rec/s. Causa: GPU **OOM**
  (`Failed to allocate ... 5.5-6.4 GB` sul nodo attention). La memoria attention scala come
  `batch * heads * seq^2`: batch 512 con seq ~512 in fp32 = ~6.4GB -> OOM quando il batch
  contiene testi lunghi (quindi intermittente).
- Fix A: cap del batch per chiamata Triton (`config.pbtxt` max_batch_size + env
  `TRITON_MAX_BATCH`), la `predict_parallel` spezza in chunk -> memoria limitata.
- Fix B: modello **fp16** (export `--fp16`, optimizer transformer di ORT, non
  onnxconverter_common che produce un grafo Cast non valido che Triton rifiuta). Logit
  identici a fp32 (diff 0.0006, argmax uguale). Memoria dimezzata -> cap rialzato a **256**.
- Risultato: 0 errori, **~2100 rec/s** su mix realistico, caso peggiore (512 testi lunghi)
  0.36s. Stima 929962 record ~7.4 min (era: crash / 24 rec/s).
- Config finale: model.onnx fp16, max_batch_size 256, TRITON_MAX_BATCH 256, CUDA EP.

### Note operative
- venv di export usa-e-getta in `.venv-export/` (torch CPU + optimum-onnx); serve solo per
  ri-generare `model.onnx`, non per il runtime. Rimovibile.
- optimum 2.x ha spostato il backend ONNX: per l'export servono `optimum>=2.0` + `optimum-onnx`.

## Fix modello "etichetta tutto tracciabilità" — riaddestramento sul dataset giusto (2026-06-28)
Root cause: il modello deployato NON era addestrato sulla tracciabilità. Il trainer
(`src/training/dataset.py`) legge `src/data/implementazione|formazione` (vecchio task
formazione-vs-implementazione, 5000/5000 = `cv_results.json` AI/NON_AI). Il dataset corretto
`data/traceability/training/train_traceability.csv` (5265/5264, label `tracciabilita`/`altro`)
esisteva ma non era agganciato a nessun loader. Decisioni utente: input = TITOLO+DESCRIZIONE
(`f"{tit}: {desc}"`, già usato dal notebook), fine-tune leggero UmBERTo (encoder congelato).

- [x] `config/model_config.yml`: opzioni `freeze_encoder` + `unfreeze_top_layers` nel blocco training
- [x] `src/config.py`: campi `freeze_encoder` / `unfreeze_top_layers` in `TrainingConfig`
- [x] `src/training/dataset.py`: `load_dataset_from_csv` (TITOLO+DESC -> testo, label str->id, stdlib csv)
- [x] `src/training/trainer.py`: `_apply_freezing` (congela encoder, allena testa + top N layer),
      optimizer solo sui param trainable. Validato sul modello reale: 14.7M/110.6M (13.3%) allenabili.
- [x] `scripts/train.py`: `--data-source csv` + path train/test traceability, eval su test set
- [x] `docker/docker-compose.yml`: mount `data/traceability/training` nel servizio `trainer`
- [x] py_compile OK su tutti i file; freezing validato (base_model_prefix='roberta', 12 layer)
- [x] Eseguire il training su GPU (servizio docker `trainer`, torch cu130) e validare su test set
- [ ] Riesportare ONNX (`scripts/export_triton.py --fp16`) -> Triton, validare su righe reali

## Fix pipeline dati + training reale eseguito (2026-07-01)
Prima di eseguire il training sono emersi due bug residui nel lavoro precedente e un problema
di qualità dati serio, tutti risolti in questa sessione:

- **Bug 1**: `docker-compose.yml` (`trainer`) girava ancora `--data-source auto`, che con lo
  storico `implementazione/formazione` presente in `src/data/` avrebbe rifatto in silenzio
  l'errore originale (stesso identico bug della rilabeling). Fix: comando esplicito
  `--data-source csv --csv-path ... --val-csv-path ... --test-csv-path ...`.
- **Bug 2**: il mount `../../../data/traceability/training:/app/data/traceability:ro` montava
  la cartella dati condivisa del monorepo direttamente nel container. Rimosso: i CSV vengono ora
  preparati con `scripts/prepare_traceability_data.py` (venv di repo, fuori Docker) e copiati in
  `src/data/traceability/{train,val,test}.csv`, dentro il mount `rw` gia' esistente.
- **Scoperta dati**: unendo `train_traceability.csv` + `test_traceability.csv` (13.162 righe) e
  deduplicando per (titolo, descrizione) restano solo 6.170 righe uniche, sbilanciate
  ~79% tracciabilita / 21% altro — il "50/50" originale era ottenuto duplicando le righe
  "altro" (solo ~1.037-1.441 uniche per file). C'erano anche 348 righe duplicate esatte tra
  train e test originali (leakage gia' presente nello split di partenza).
  **Fix**: `prepare_traceability_data.py` ora deduplica sempre, poi pesca automaticamente
  campioni "altro" aggiuntivi e unici da `data/technology_mapping/` (dataset enorme e non
  correlato, stesso schema aiuti di stato) fino a pareggiare il conteggio di tracciabilita,
  escludendo righe con keyword di tracciabilita (`tracciabil|rintracciabil|filiera|blockchain`)
  per non introdurre falsi negativi. Risultato: 9.800 righe, 50/50 esatto, split 70/20/10
  (train 6.860 / val 1.960 / test 980).
- **Training eseguito** (`docker compose --profile training up trainer`, RTX 5070 Ti, ~10 min):
  5-Fold CV su train+val (8.820 campioni) F1 0.9777 ± 0.0044 (per-fold 0.9697-0.9826,
  precision/recall entrambi alti e vicini, nessun collasso su una classe). Retrain finale +
  **HELD-OUT TEST (980 campioni mai visti)**: acc 0.9724, P 0.9602, R 0.9857, F1 0.9728.
  Risultato reale e non degenere, a differenza del bug originale (~1.00 di confidenza su
  "tutto tracciabilita").
- Tuning hardware (Ryzen 9 9900X 12c/24t, RTX 5070 Ti 16GB): `num_workers` 4->8,
  `batch_size` 16->32, aggiunto `persistent_workers`/`prefetch_factor=4` ai DataLoader,
  `shm_size: 2gb` sul servizio `trainer`.
- Non incluso in questa sessione (resta il prossimo passo): riesportare ONNX e ridistribuire su
  Triton con il nuovo modello.
