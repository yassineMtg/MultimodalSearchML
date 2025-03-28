import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_parquet("data/query_features_with_timestamp.parquet")
row = df[df["query_id"] == 113370]
print(row[["query_id", "event_timestamp"]])
