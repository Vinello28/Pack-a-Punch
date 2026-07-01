#!/usr/bin/env python3
"""
Prepare the traceability dataset for training.

Combines the two raw source CSVs (data/traceability/training/{train,test}_traceability.csv),
drops duplicate/invalid rows, augments the minority 'altro' class with additional unique
descriptions sourced from data/technology_mapping/ (a much larger, unrelated AI/non-AI
classification dataset over the same "aiuti di stato" records), and writes a fresh stratified
70/20/10 train/val/test split into this project's own dataset directory
(src/data/traceability/), mirroring the existing implementazione/formazione/Test convention.

Deduplication key is DESCRIZIONE_PROGETTO alone (not the (title, description) pair) — the
project's established convention (see src/regex_multiprocessing.py), since many rows repeat an
identical description under different, often cosmetic, titles (e.g. the same funding-scheme
boilerplate attached to different beneficiary names).

Run with the repo-root venv (NOT inside Docker):
    .venv/bin/python src/inference-usage/scripts/prepare_traceability_data.py
"""

import random
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMODULE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SUBMODULE_ROOT.parent.parent

# Read label_map straight from the YAML config rather than importing src.config: that module
# pulls in pydantic_settings, which belongs to the training image's deps and isn't guaranteed to
# be installed in the repo-root venv this script is meant to run in.
_model_cfg = yaml.safe_load((SUBMODULE_ROOT / "config" / "model_config.yml").read_text())
LABEL_MAP = _model_cfg["model"]["label_map"]

SOURCE_DIR = REPO_ROOT / "data" / "traceability" / "training"
SOURCE_TRAIN_CSV = SOURCE_DIR / "train_traceability.csv"
SOURCE_TEST_CSV = SOURCE_DIR / "test_traceability.csv"

# Extra source of unique 'altro' (non-traceability) descriptions, used to correct the class
# imbalance revealed after deduplication (raw files fake a 50/50 balance by duplicating the
# minority 'altro' rows many times over; the real unique data skews heavily towards
# tracciabilita).
TECH_MAPPING_DIR = REPO_ROOT / "data" / "technology_mapping"
TRACEABILITY_KEYWORD_RE = re.compile(
    r"tracciabil|rintracciabil|filiera|blockchain", re.IGNORECASE
)

OUT_DIR = SUBMODULE_ROOT / "src" / "data" / "traceability"

TITLE_COL = "TITOLO_PROGETTO"
DESC_COL = "DESCRIZIONE_PROGETTO"
LABEL_COL = "label"

SEED = 42


def _collect_file_candidates(filepath: Path) -> pd.DataFrame:
    """
    Worker (one process per file): return unique, keyword-filtered (title, description) rows
    from a single data/technology_mapping/*.csv file, deduplicated by DESCRIZIONE_PROGETTO alone.
    Reads in chunks since some of these files are multiple GB.
    """
    rows = []
    try:
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=200_000,
            usecols=lambda c: c in (TITLE_COL, DESC_COL),
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
        )
        seen_in_file: set[str] = set()
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=[DESC_COL])
            chunk[TITLE_COL] = chunk[TITLE_COL].fillna("")
            chunk = chunk[~chunk[DESC_COL].str.contains(TRACEABILITY_KEYWORD_RE, na=False)]
            chunk = chunk[~chunk[TITLE_COL].str.contains(TRACEABILITY_KEYWORD_RE, na=False)]
            chunk = chunk.drop_duplicates(subset=[DESC_COL])
            chunk = chunk[~chunk[DESC_COL].isin(seen_in_file)]
            seen_in_file.update(chunk[DESC_COL].tolist())
            if not chunk.empty:
                rows.append(chunk[[TITLE_COL, DESC_COL]])
    except Exception as e:
        print(f"  Errore su {filepath}: {e}")

    if not rows:
        return pd.DataFrame(columns=[TITLE_COL, DESC_COL])
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=[DESC_COL])


def collect_technology_mapping_candidates() -> pd.DataFrame:
    """
    Scan every data/technology_mapping/*.csv file (in parallel, one worker per file) for unique
    (by description) rows safe to use as additional 'altro' examples.
    """
    files = sorted(TECH_MAPPING_DIR.glob("reclassified_multiclass_aiuti_*.csv")) + [
        TECH_MAPPING_DIR / "ai_records_full_export.csv"
    ]
    files = [f for f in files if f.exists()]

    num_workers = min(cpu_count(), len(files)) or 1
    print(f"  Scansione di {len(files)} file con {num_workers} worker (multiprocessing)...")
    with Pool(num_workers) as pool:
        parts = list(pool.imap_unordered(_collect_file_candidates, files))

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=[DESC_COL])
    return combined


def main() -> int:
    for path in (SOURCE_TRAIN_CSV, SOURCE_TEST_CSV):
        if not path.exists():
            print(f"ERROR: source CSV not found: {path}")
            return 1

    df = pd.concat(
        [pd.read_csv(SOURCE_TRAIN_CSV), pd.read_csv(SOURCE_TEST_CSV)],
        ignore_index=True,
    )
    total_read = len(df)

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()
    valid_labels = set(LABEL_MAP.values())
    df = df[df[LABEL_COL].isin(valid_labels)]

    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str).str.strip()
    df[DESC_COL] = df[DESC_COL].fillna("").astype(str).str.strip()
    df = df[(df[TITLE_COL] != "") | (df[DESC_COL] != "")]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=[DESC_COL], keep="first")
    duplicates_dropped = before_dedup - len(df)

    invalid_or_empty_dropped = total_read - before_dedup
    print(
        f"Combined {total_read} rows from source CSVs "
        f"(dropped {invalid_or_empty_dropped} invalid/empty, "
        f"{duplicates_dropped} duplicate descriptions) -> {len(df)} usable rows"
    )
    counts = df[LABEL_COL].value_counts().to_dict()
    print(f"  Label balance after dedup: {counts}")

    # The raw files fake a 50/50 balance by duplicating 'altro' rows; after deduplication the
    # real unique data skews heavily towards tracciabilita. Augment 'altro' with unique,
    # traceability-keyword-filtered descriptions from data/technology_mapping/ up to parity
    # with the tracciabilita count.
    tracciabilita_count = counts.get("tracciabilita", 0)
    altro_count = counts.get("altro", 0)
    deficit = max(0, tracciabilita_count - altro_count)
    if deficit > 0:
        print(f"Collecting 'altro' augmentation candidates from {TECH_MAPPING_DIR} ...")
        candidates_df = collect_technology_mapping_candidates()
        existing_descs = set(df[DESC_COL])
        candidates_df = candidates_df[~candidates_df[DESC_COL].isin(existing_descs)]

        rng = random.Random(SEED)
        idx = list(candidates_df.index)
        rng.shuffle(idx)
        sampled_idx = idx[:deficit]
        aug_df = candidates_df.loc[sampled_idx, [TITLE_COL, DESC_COL]].reset_index(drop=True)
        aug_df[LABEL_COL] = "altro"
        print(
            f"  {len(candidates_df)} unique safe candidates available; "
            f"added {len(aug_df)} augmented 'altro' rows (target deficit: {deficit})"
        )
        df = pd.concat([df, aug_df], ignore_index=True)
        assert df[DESC_COL].is_unique, "duplicate descriptions leaked into the final dataset"
        counts = df[LABEL_COL].value_counts().to_dict()
        print(f"  Label balance after augmentation: {counts}")

    # 70% train, then split the remaining 30% into 20%/10% (2/3 val, 1/3 test).
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[LABEL_COL], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1 / 3, stratify=temp_df[LABEL_COL], random_state=SEED
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = OUT_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        split_counts = split_df[LABEL_COL].value_counts().to_dict()
        print(f"  {name}.csv: {len(split_df)} rows, {split_counts} -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
