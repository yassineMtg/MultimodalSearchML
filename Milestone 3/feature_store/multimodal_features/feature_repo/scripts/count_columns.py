import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/raw"))

file_path = os.path.join(BASE_DIR, "query_features_with_timestamp.parquet")


df = pd.read_parquet(file_path)
print("Number of rows:", len(df))


#import pandas as pd

#df = pd.read_parquet("data/product_features_flat.parquet")
#print("Min timestamp:", df["event_timestamp"].min())
#print("Max timestamp:", df["event_timestamp"].max())
