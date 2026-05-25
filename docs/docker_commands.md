# Comandi Rapidi Docker per Pack-a-Punch

Questo file funge da "cheat sheet" per avviare rapidamente i servizi di addestramento e inferenza tramite Docker Compose per il task `Formazione vs Implementazione`. Tutti i comandi presuppongono che tu ti trovi all'interno della cartella `src/inference-usage`.

## 1. Preparazione dei Dati

Prima di avviare il training, assicurati di aver distribuito i dati dai file CSV (presenti in `public/`) alle cartelle di destinazione corrette:

```bash
python scripts/distribute_data.py
```

## 2. Avviare l'Inferenza (Server API PyTorch)

Per avviare il classificatore ed esporre l'API sulla porta `8080`:

```bash
docker compose -f docker/docker-compose.yml up classifier
```

*Nota: aggiungi il flag `-d` alla fine del comando se desideri eseguire il container in background (detached mode).*

## 3. Avviare il Training (PyTorch)

Per eseguire il processo di addestramento (il modello salverà gli step nella cartella `src/models/`):

```bash
docker compose -f docker/docker-compose.yml run --rm trainer
```

*Nota: Se modifichi il `Dockerfile.train`, ricorda di eseguire prima `docker compose -f docker/docker-compose.yml build trainer`.*

## 4. Distillation (Opzionale)

Se desideri usare la knowledge distillation contattando un Teacher LLM in esecuzione localmente (es. LM Studio su `host.docker.internal:1234`):

```bash
docker compose -f docker/docker-compose.yml run --rm distiller
```

## 5. Fermare i Container

Se hai avviato i container in background (`-d`) e vuoi spegnerli e rimuoverli:

```bash
docker compose -f docker/docker-compose.yml down
```
