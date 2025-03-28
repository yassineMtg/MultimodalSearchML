# import pandas as pd

# df = pd.read_parquet("data/product_features_flat.parquet")
# print("Number of rows:", len(df))


import pandas as pd

df = pd.read_parquet("data/product_features_flat.parquet")
print("Min timestamp:", df["event_timestamp"].min())
print("Max timestamp:", df["event_timestamp"].max())
