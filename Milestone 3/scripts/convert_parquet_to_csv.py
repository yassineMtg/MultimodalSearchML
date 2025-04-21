import pandas as pd
import os

# Define input/output paths
RAW_DIR = os.path.join("data", "raw")

# Parquet to CSV conversion mapping
files_to_convert = {
    "query_features.parquet": "query_features.csv",
    "product_features.parquet": "product_features.csv"
}

def convert_parquet_to_csv():
    for parquet_file, csv_file in files_to_convert.items():
        parquet_path = os.path.join(RAW_DIR, parquet_file)
        csv_path = os.path.join(RAW_DIR, csv_file)

        if not os.path.exists(parquet_path):
            print(f"❌ File not found: {parquet_path}")
            continue

        print(f"🔄 Converting {parquet_file} to CSV...")
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

if __name__ == "__main__":
    convert_parquet_to_csv()
