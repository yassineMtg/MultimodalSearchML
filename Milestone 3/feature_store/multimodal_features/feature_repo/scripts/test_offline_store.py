import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/raw"))

query_features_path = os.path.join(BASE_DIR, "query_features_with_timestamp.parquet")

df = pd.read_parquet(query_features_path)
print(df.head())  # View first few rows
print(df.columns[:5])  # Confirm the feature names exist
