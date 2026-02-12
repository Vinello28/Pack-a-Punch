import pandas as pd
import os
import uuid

def distribute_data():
    # Paths
    excel_path = "src/data/tbc_classificata.xlsx"
    base_data_path = "src/data"

    # Verify input file exists
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    # Read Excel file
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Check columns
    required_columns = ["Descrizione", "Label"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing columns. Expected {required_columns}, found {df.columns}")
        return

    # Counters
    counts = {"ai": 0, "non_ai": 0, "skipped": 0}

    # Iterate rows
    for index, row in df.iterrows():
        description = row["Descrizione"]
        label = row["Label"]

        # Normalize label just in case
        if isinstance(label, str):
            label = label.lower().strip()
        
        # Check if valid label
        target_dir = None
        if label == "ai":
            target_dir = os.path.join(base_data_path, "ai")
        elif label == "non_ai":
            target_dir = os.path.join(base_data_path, "non_ai")
        else:
            # Handle variations if necessary or skip
            # Check for possible variations
            normalized_label = label.lower().strip()
            if normalized_label == "ai":
                target_dir = os.path.join(base_data_path, "ai")
            elif normalized_label == "non_ai" or normalized_label == "non-ai":
                target_dir = os.path.join(base_data_path, "non_ai")
            else:
                print(f"Warning: Unknown label '{label}' at row {index}. Skipping.")
                counts["skipped"] = counts.get("skipped", 0) + 1
                continue

        # Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Create filename
        # Use a deterministic name based on row index to avoid duplicates if run again
        filename = f"tbc_{index}.txt"
        filepath = os.path.join(target_dir, filename)

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(description))
            counts[label] += 1
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")

    print("Distribution complete:")
    print(f"  AI files created: {counts['ai']}")
    print(f"  Non-AI files created: {counts['non_ai']}")
    print(f"  Skipped rows: {counts['skipped']}")

if __name__ == "__main__":
    distribute_data()
