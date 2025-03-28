import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_parquet("data/query_features_with_timestamp.parquet")
print(df.head())  # View first few rows
print(df.columns[:5])  # Confirm the feature names exist
