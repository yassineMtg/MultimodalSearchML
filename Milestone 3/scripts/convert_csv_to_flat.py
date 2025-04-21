import pandas as pd
import numpy as np

# Load original CSV with query_id
df = pd.read_csv("data/raw/query_features.csv")

# Parse the CLIP vector from the string column
parsed = df["clip_text_features"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Turn list of vectors into columns f0 to f767
parsed_df = pd.DataFrame(parsed.tolist(), columns=[f"f{i}" for i in range(768)])

# Add back query_id as the first column
parsed_df.insert(0, "query_id", df["query_id"])

# Save it
parsed_df.to_csv("data/raw/query_features_flat.csv", index=False)
