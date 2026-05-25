import pandas as pd
import os

def distribute_csv_data(csv_path, target_base_dir, counts_dict, split_name):
    print(f"Processing {split_name} from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check columns
    required_columns = ["DESCRIZIONE_PROGETTO", "label"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing columns. Expected {required_columns}, found {df.columns}")
        return

    # Iterate rows
    for index, row in df.iterrows():
        description = row["DESCRIZIONE_PROGETTO"]
        label = row["label"]

        if not isinstance(description, str) or pd.isna(description):
            counts_dict["skipped"] = counts_dict.get("skipped", 0) + 1
            continue

        if isinstance(label, str):
            label = label.lower().strip()
        
        target_dir = None
        if label == "formazione":
            target_dir = os.path.join(target_base_dir, "formazione")
            label_key = "formazione"
        elif label == "implementazione":
            target_dir = os.path.join(target_base_dir, "implementazione")
            label_key = "implementazione"
        else:
            print(f"Warning: Unknown label '{label}' at row {index} in {split_name}. Skipping.")
            counts_dict["skipped"] = counts_dict.get("skipped", 0) + 1
            continue

        # Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Create filename
        filename = f"{split_name}_{index}.txt"
        filepath = os.path.join(target_dir, filename)

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(description))
            counts_dict[label_key] = counts_dict.get(label_key, 0) + 1
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")

def distribute_data():
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_usage_dir = os.path.dirname(script_dir)
    src_dir = os.path.dirname(inference_usage_dir)
    project_root = os.path.dirname(src_dir)

    train_csv_path = os.path.join(inference_usage_dir, "public", "trainingset.csv")
    test_csv_path = os.path.join(inference_usage_dir, "public", "testset.csv")

    train_data_dir = os.path.join(inference_usage_dir, "src", "data")
    test_data_dir = os.path.join(project_root, "data", "Test")

    counts = {"formazione": 0, "implementazione": 0, "skipped": 0}
    distribute_csv_data(train_csv_path, train_data_dir, counts, "train")
    
    print(f"  Training Formazione files created: {counts['formazione']}")
    print(f"  Training Implementazione files created: {counts['implementazione']}")
    print(f"  Training Skipped rows: {counts['skipped']}")
    
    counts_test = {"formazione": 0, "implementazione": 0, "skipped": 0}
    distribute_csv_data(test_csv_path, test_data_dir, counts_test, "test")

    print(f"  Test Formazione files created: {counts_test['formazione']}")
    print(f"  Test Implementazione files created: {counts_test['implementazione']}")
    print(f"  Test Skipped rows: {counts_test['skipped']}")

if __name__ == "__main__":
    distribute_data()
