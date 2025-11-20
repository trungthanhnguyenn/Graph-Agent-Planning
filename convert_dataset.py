
from datasets import load_from_disk

# Path to the dataset directory
dataset_path = "GAP-MHQA-RL-Dataset"

# Path to output parquet file
output_path = "GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet"

try:
    # Load dataset from disk (it will automatically find the 'train' folder)
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Convert to Pandas DataFrame
    print("Converting to Pandas DataFrame...")
    df = dataset['train'].to_pandas()

    # Save as parquet file
    print(f"Saving to Parquet file at: {output_path}")
    df.to_parquet(output_path)

    print("\nConversion successful!")
    print(f"File saved at: {output_path}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure the 'datasets' and 'pyarrow' libraries are installed in your 'parallel-agent' conda environment.")
    print("You can install them with: pip install datasets pyarrow pandas")

