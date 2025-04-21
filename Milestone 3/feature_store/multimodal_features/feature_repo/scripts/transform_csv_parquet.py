import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/raw"))

query_features_path = os.path.join(BASE_DIR, "query_features_with_timestamp.csv")
query_features_parquet_path = os.path.join(BASE_DIR, "query_features_flat.parquet")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_csv(query_features_path)
df["event_timestamp"] = pd.Timestamp.now()  # or actual timestamps if you have them
df.to_parquet(query_features_parquet_path, index=False)
