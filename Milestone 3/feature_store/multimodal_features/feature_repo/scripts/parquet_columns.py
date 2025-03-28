import pandas as pd

# Load the flattened Parquet file
df = pd.read_parquet("data/product_features_flat.parquet")

# Show the first few rows
print(df.head())

# Print column names to verify all features
print("\nTotal columns:", len(df.columns))
print("Column names preview:", df.columns[:10].tolist(), "...", df.columns[-10:].tolist())
