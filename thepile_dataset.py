from datasets import load_dataset

# Load the full dataset (only 'train' exists)
dataset = load_dataset("monology/pile-uncopyrighted", split="train")

# Shuffle and split off 2% for val+test
dataset = dataset.shuffle(seed=42)
val_test_split = dataset.train_test_split(test_size=0.02)

# Now split that 2% in half (1% val, 1% test)
val_test = val_test_split["test"].train_test_split(test_size=0.5)

# Extract validation and test splits
val_dataset = val_test["train"]
test_dataset = val_test["test"]

# Save to disk as JSONL
val_dataset.to_json("pile_uncopyrighted_val.jsonl")
test_dataset.to_json("pile_uncopyrighted_test.jsonl")
