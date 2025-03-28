import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_csv("data/query_features_with_timestamp.csv")
df["event_timestamp"] = pd.Timestamp.now()  # or actual timestamps if you have them
df.to_parquet("data/query_features_flat.parquet", index=False)
