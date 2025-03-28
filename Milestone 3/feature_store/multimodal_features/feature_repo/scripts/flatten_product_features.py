import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load the Parquet file
df = pd.read_parquet("data/product_features.parquet")

# Safe parsing function
def parse_features(val):
    if isinstance(val, str):
        return np.fromstring(val.strip("[]"), sep=" ")
    return val  # Already parsed or None

# Apply parsing
df['clip_text_features'] = df['clip_text_features'].apply(parse_features)
df['clip_image_features'] = df['clip_image_features'].apply(parse_features)

# Drop rows with missing features (or handle them differently if needed)
df = df.dropna(subset=['clip_text_features', 'clip_image_features'])

# Detect dimensions dynamically
text_dim = len(df['clip_text_features'].iloc[0])
# Flatten both arrays into separate columns
clip_text_df = pd.DataFrame(df['clip_text_features'].tolist(), columns=[f"text_f{i}" for i in range(text_dim)])

# Detect dimensions dynamically
image_dim = len(df['clip_image_features'].iloc[0])
clip_image_df = pd.DataFrame(df['clip_image_features'].tolist(), columns=[f"image_f{i}" for i in range(image_dim)])

# Combine all into one dataframe
flattened_df = pd.concat([df['product_id'], clip_text_df, clip_image_df], axis=1)

start_time = datetime(2025, 3, 1)
flattened_df['event_timestamp'] = [
    start_time + timedelta(seconds=i * 10) for i in range(len(flattened_df))
]

flattened_df['product_id'] = flattened_df['product_id'].astype(str)

# Save to Parquet
flattened_df.to_parquet("data/product_features_flat.parquet", index=False)
