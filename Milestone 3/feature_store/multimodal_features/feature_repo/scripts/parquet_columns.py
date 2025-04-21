import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/raw"))

product_features_path = os.path.join(BASE_DIR, "product_features_flat.parquet")

# Load the flattened Parquet file
df = pd.read_parquet(product_features_path)

# Show the first few rows
print(df.head())

# Print column names to verify all features
print("\nTotal columns:", len(df.columns))
print("Column names preview:", df.columns[:10].tolist(), "...", df.columns[-10:].tolist())
