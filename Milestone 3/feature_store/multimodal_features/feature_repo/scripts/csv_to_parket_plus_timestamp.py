import pandas as pd
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load the full CSV
df = pd.read_csv("data/query_features_flat.csv")

# Add a valid timezone-aware timestamp column
df["event_timestamp"] = pd.to_datetime(datetime.utcnow()).tz_localize("UTC")

# Save to the proper file
df.to_parquet("data/query_features_with_timestamp.parquet", index=False)

# Also save as CSV if needed (optional)
df.to_csv("data/query_features_with_timestamp.csv", index=False)
