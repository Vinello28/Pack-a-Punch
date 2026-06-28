# Lessons

## GPU reale: RTX 5070 Ti (Blackwell), non la 3060 Ti del README
- **Pattern:** il README e i commenti citano "RTX 3060 Ti" (Ampere), ma la macchina di sviluppo
  monta una **RTX 5070 Ti (Blackwell, sm_120)**, driver 595.71.05, CUDA runtime fino a 13.2.
- **Regola:** non fidarsi delle GPU citate nei doc; verificare sempre con `nvidia-smi -L` e
  `nvidia-smi --query-gpu=driver_version --format=csv` prima di scegliere immagini CUDA/Triton.
- **Conseguenza:** Blackwell richiede CUDA 13. Le immagini `tritonserver` su CUDA 13 partono da
  `25.08-py3`. Per stare entro il CUDA 13.2 del driver -> `26.01-py3` (CUDA 13.1.1). Evitare tag
  con CUDA > driver (es. `26.06-py3` = CUDA 13.3).

## L'utente vuole CUDA 13, non si transige
- **Pattern:** richiesta esplicita e non negoziabile di CUDA 13 (coerente con la Blackwell).
- **Regola:** per questa macchina, qualsiasi immagine GPU (Triton, base CUDA, ecc.) deve essere
  CUDA 13.x compatibile col driver; non proporre tag CUDA 12.x.

## ORT TensorRT EP + shape dinamiche = rebuild continui e crash
- **Pattern:** con l'accel TensorRT del backend onnxruntime, ogni nuova shape (batch size E
  lunghezza sequenza) fa ricostruire un engine TRT (20-40s). Col padding dinamico ogni batch
  ha lunghezza diversa -> rebuild continui (24 rec/s nel notebook). Inoltre il batch grande
  512x512 falliva: workspace 1GB < ~1.5GB richiesti -> "failed to create engine / No engine".
- **Sintomo nei log Triton:** `UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory`
  + `Error Code 10: Could not find any implementation for node`.
- **Regola:** per servire shape pienamente dinamiche, usare il **CUDA EP** (nessun rebuild per
  shape), tenendo il dynamic batching di Triton per il throughput. Risultato: batch 512 OK,
  ~1265 rec/s su dati a lunghezza variabile, 0 errori. Il TRT va usato solo con engine
  precompilato + profili min/opt/max espliciti + workspace adeguato, non col EP lazy.
- **Lezione di metodo:** il benchmark "a caldo" su shape identiche nascondeva il problema;
  testare sempre col workload reale (batch grandi + lunghezze variabili) prima di dire "fatto".

## Memoria attention = batch * heads * seq^2 -> cap il batch
- **Pattern:** batch grandi (512) con seq lunga (512) in fp32 = ~6.4GB per un buffer attention
  -> GPU OOM intermittente (solo quando il batch contiene testi lunghi). Errore ORT:
  `BFCArena ... Failed to allocate memory for requested buffer of size ~6e9`.
- **Regola:** per modelli transformer servire con un cap sul batch per chiamata
  (`config.pbtxt` max_batch_size + chunking lato engine). Il throughput viene comunque dal
  dynamic batching di Triton, non da un singolo batch enorme.

## fp16: usare l'optimizer transformer di ORT, NON onnxconverter_common
- **Pattern:** `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)` su
  questo modello produce un grafo con un nodo `Cast` di tipo incoerente (output float16
  dichiarato float) -> Triton non lo carica (`Type Error ... does not match expected type`).
- **Regola:** convertire in fp16 via `onnxruntime.transformers.optimizer.optimize_model(
  model_type="bert", num_heads=12, hidden_size=768)` poi `.convert_float_to_float16(
  keep_io_types=True)`. Grafo valido, logit ~identici a fp32 (diff 1e-3), ~2x throughput.
- **Metodo:** validare il grafo fp16 in locale con una `onnxruntime.InferenceSession` (e
  confronto logit vs fp32) PRIMA di deployarlo su Triton, per non causare cicli di unhealthy.

## Il modello "tracciabilità" è in realtà "implementazione vs formazione" mal-etichettato
- **Sintomo riportato:** il modello etichetta "tutto tracciabilità" sui CSV reali
  (`reclassified_multiclass_*.csv`), con confidenza ~1.00.
- **Causa (root, non superficiale):** il training set in `src/data/implementazione/` (label 1,
  rinominata `tracciabilita`) e `src/data/formazione/` (label 0, `altro`) NON riguarda la
  tracciabilità. Su 5000 file "tracciabilita" solo 20 citano `traccia`, 44 `filiera`, 4
  `blockchain`: sono descrizioni generiche di progetti. La classe "altro" è per 2/3 corsi.
  Qualcuno ha rinominato il `label_map` in `config/model_config.yml` (1->tracciabilita,
  0->altro) SENZA riaddestrare né rietichettare. Il modello classifica "progetto vs corso"
  (CV F1 0.96, std 0.005 -> generalizza bene, NON overfitta), e siccome i CSV sono quasi tutti
  progetti -> sempre classe 1 -> stampa "tracciabilita".
- **Regola:** prima di diagnosticare qualità/architettura/overfit, VERIFICARE che i dati di
  training corrispondano semanticamente alle etichette dichiarate. Apri qualche esempio per
  classe e fai un grep dei termini-chiave del concetto. Un `label_map` rinominato non riaddestra
  il modello. CV alta su dati sintetici/repurposed != correttezza sul dominio reale.
- **Conseguenza:** nessun tweak di freezing/input/architettura insegna un concetto assente dai
  label. Serve ground-truth reale di tracciabilità sul dominio target (i CSV) prima di tutto.

## Verificare i tag delle immagini invece di indovinare
- **Pattern:** avevo messo `24.10-py3` (CUDA 12.6) "a memoria"; sbagliato per Blackwell.
- **Regola:** verificare i tag reali via registry NGC (token guest `nvcr.io/proxy_auth`,
  `/v2/nvidia/tritonserver/tags/list`) e ispezionare `CUDA_VERSION` nel config blob dell'immagine.
