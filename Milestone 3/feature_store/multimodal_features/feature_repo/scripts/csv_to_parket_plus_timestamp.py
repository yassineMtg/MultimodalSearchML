import pandas as pd
from datetime import datetime
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/raw"))

file_path = os.path.join(BASE_DIR, "query_features_flat.csv")

parquet_path = os.path.join(BASE_DIR, "query_features_with_timestamp.parquet")

csv_path = os.path.join(BASE_DIR, "query_features_with_timestamp.csv")

# Load the full CSV
df = pd.read_csv(file_path)

# Add a valid timezone-aware timestamp column
df["event_timestamp"] = pd.to_datetime(datetime.utcnow()).tz_localize("UTC")

# Save to the proper file
df.to_parquet(parquet_file, index=False)

# Also save as CSV if needed (optional)
df.to_csv(csv_path, index=False)
