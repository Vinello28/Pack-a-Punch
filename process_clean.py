import pandas as pd
import os
import uuid

file_path = "public/train_5.xlsx"
df = pd.read_excel(file_path)

print("Columns:", df.columns.tolist())
head_df = df.head()
print(head_df)

ai_dir = "src/data/ai"
non_ai_dir = "src/data/non_ai"

os.makedirs(ai_dir, exist_ok=True)
os.makedirs(non_ai_dir, exist_ok=True)

# I want to see the label column. Let's find it.
label_col = df.columns[1]
print(f"Label col is {label_col}")

# Check unique values in label column
print("Labels:", df[label_col].unique())

def process():
    for idx, row in df.iterrows():
        desc = str(row[df.columns[0]])
        label = str(row[label_col]).strip().lower()
        
        # Decide directory
        if label in ['ai', '1', '1.0', 'true']:
            target_dir = ai_dir
        elif label in ['non_ai', 'non-ai', '0', '0.0', 'false']:
            target_dir = non_ai_dir
        else:
            # Let's say if the label contains 'ai' or 'non'
            if 'non_ai' in label or 'non' in label:
                target_dir = non_ai_dir
            elif 'ai' in label:
                target_dir = ai_dir
            else:
                print(f"Skipping row {idx} due to unknown label: {label}")
                continue
                
        file_name = f"clean_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join(target_dir, file_name), "w", encoding="utf-8") as f:
            f.write(desc)

process()
print("Done processing")
