import pandas as pd
import os
import argparse

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
        else:
            counts_dict["skipped"] = counts_dict.get("skipped", 0) + 1
            continue

        target_dir = os.path.join(target_base_dir, label)
        
        # Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Create filename
        filename = f"{split_name}_{index}.txt"
        filepath = os.path.join(target_dir, filename)

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(description))
            counts_dict[label] = counts_dict.get(label, 0) + 1
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")

def distribute_data():
    parser = argparse.ArgumentParser(description="Distribute CSV data into text files by label.")
    
    # Resolve default paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_usage_dir = os.path.dirname(script_dir)
    src_dir = os.path.dirname(inference_usage_dir)
    project_root = os.path.dirname(src_dir)
    
    default_train_csv = os.path.join(inference_usage_dir, "public", "trainingset.csv")
    default_test_csv = os.path.join(inference_usage_dir, "public", "testset.csv")
    default_train_dir = os.path.join(inference_usage_dir, "src", "data")
    default_test_dir = os.path.join(project_root, "data", "Test")

    parser.add_argument("--train-csv", type=str, default=default_train_csv, help="Path to training CSV file")
    parser.add_argument("--test-csv", type=str, default=default_test_csv, help="Path to test CSV file")
    parser.add_argument("--train-dir", type=str, default=default_train_dir, help="Directory to save training data")
    parser.add_argument("--test-dir", type=str, default=default_test_dir, help="Directory to save test data")

    args = parser.parse_args()

    # Train Data
    counts = {"skipped": 0}
    distribute_csv_data(args.train_csv, args.train_dir, counts, "train")
    
    print("--- Training Data Summary ---")
    for label, count in counts.items():
        if label != "skipped":
            print(f"  {label.capitalize()} files created: {count}")
    print(f"  Skipped rows: {counts.get('skipped', 0)}")
    
    # Test Data
    counts_test = {"skipped": 0}
    distribute_csv_data(args.test_csv, args.test_dir, counts_test, "test")

    print("\n--- Test Data Summary ---")
    for label, count in counts_test.items():
        if label != "skipped":
            print(f"  {label.capitalize()} files created: {count}")
    print(f"  Skipped rows: {counts_test.get('skipped', 0)}")

if __name__ == "__main__":
    distribute_data()
