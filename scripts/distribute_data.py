"""
Distribute labeled data from CSV into per-class TXT directories.

Reads public/modernbert_final.csv and writes one .txt file per row
into src/data/<class_slug>/, where the slug is derived from the label name
using the same _slugify function that the dataset loader uses.
"""

import os
import sys
from collections import Counter

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings


def _slugify(name: str) -> str:
    """Convert a label name to a directory slug (lowercase, underscores)."""
    import re

    slug = name.lower()
    slug = slug.replace("&", "").replace(",", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return re.sub(r"_+", "_", slug)


def distribute_data():
    csv_path = "public/modernbert_final.csv"
    base_data_path = "src/data"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    text_col = "description" if "description" in df.columns else "Descrizione"
    required_columns = [text_col, "Label"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing columns. Expected {required_columns}, found {list(df.columns)}")
        return

    # Build label -> slug mapping from config
    name_to_slug = {name: _slugify(name) for name in settings.model.label_map.values()}
    # Also build case-insensitive lookup
    name_lower_to_slug = {name.lower(): slug for name, slug in name_to_slug.items()}

    counts = Counter()
    skipped = 0

    for index, row in df.iterrows():
        description = str(row[text_col]).strip()
        label = str(row["Label"]).strip()

        if not description or description == "nan":
            skipped += 1
            continue

        slug = name_lower_to_slug.get(label.lower())
        if slug is None:
            print(f"Warning: Unknown label '{label}' at row {index}. Skipping.")
            skipped += 1
            continue

        target_dir = os.path.join(base_data_path, slug)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"tbc_{index}.txt"
        filepath = os.path.join(target_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(description)
            counts[slug] += 1
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")

    print("Distribution complete:")
    for slug in sorted(counts):
        print(f"  {slug}: {counts[slug]} files")
    print(f"  Skipped rows: {skipped}")
    print(f"  Total: {sum(counts.values())} files")


if __name__ == "__main__":
    distribute_data()
