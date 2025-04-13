import pandas as pd

# Load files
sqid_df = pd.read_csv("../../Milestone 3/data/processed/query_features_with_timestamp.csv")
esci_df = pd.read_parquet("../data/raw/shopping_queries_dataset_examples.parquet")

# Drop duplicates just in case
sqid_df = sqid_df.drop_duplicates(subset=["query_id"])
esci_df = esci_df.drop_duplicates(subset=["query_id"])

# Map labels
label_map = {"I": 0, "C": 1, "S": 2, "E": 3}
esci_df["label"] = esci_df["esci_label"].map(label_map)

# Merge by query_id
merged_df = pd.merge(sqid_df, esci_df[["query_id", "label"]], on="query_id", how="inner")

# Save merged dataset
merged_df.to_csv("../data/processed/merged_queries.csv", index=False)
print("✅ Merged dataset saved with shape:", merged_df.shape)
