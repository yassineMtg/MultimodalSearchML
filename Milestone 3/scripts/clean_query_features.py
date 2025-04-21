# import pandas as pd
# import os

# # Paths
# RAW_DIR = os.path.join("data", "raw")
# INPUT_FILE = os.path.join(RAW_DIR, "query_features.csv")
# OUTPUT_FILE = os.path.join(RAW_DIR, "query_features_cleaned.csv")

# def clean_clip_vectors():
#     print("🚀 Cleaning clip_text_features...")
#     df = pd.read_csv(INPUT_FILE)

#     # Replace newlines and excessive spaces inside vector strings
#     df["clip_text_features"] = (
#         df["clip_text_features"]
#         .astype(str)
#         .str.replace(r"\s+", " ", regex=True)
#         .str.strip()
#     )

#     # Save cleaned version
#     df.to_csv(OUTPUT_FILE, index=False)
#     print(f"✅ Cleaned CSV saved to: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     clean_clip_vectors()


import pandas as pd

df = pd.read_csv("data/raw/query_features.csv")
df.drop(columns=["query_id"], inplace=True)
df.to_csv("data/raw/query_features_final.csv", index=False)
