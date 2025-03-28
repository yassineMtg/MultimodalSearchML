import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/query_features.csv")
sample = df["clip_text_features"].iloc[0]

# Convert string to list
values = np.array(sample.strip("[]").replace("\n", " ").split(), dtype=np.float32)
print(values.shape)
