from datasets import load_dataset
import json

def inspect_schema():
    """
    Loads the Parquet dataset and prints its schema (features).
    """
    try:
        print("Loading dataset from parquet file")
        dataset = load_dataset(
            "parquet", 
            data_files={"train": "GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet"}
        )['train']
        
        print("\nDataset loaded successfully!")
        
        # Print the features (schema)
        print("\nSchema (Features):")
        print(dataset.features)
        
        print("\nFirst example:")

        # Json pretty-print the first example
        print(json.dumps(dataset[0], indent=2))

    except Exception as e:
        print(f"Error inspecting dataset: {e}")

if __name__ == "__main__":
    inspect_schema()
