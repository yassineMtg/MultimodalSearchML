import pandas as pd

# === Paths ===
SQID_QUERY_PATH     = "../app/data/query_features.parquet"
SQID_PRODUCT_PATH   = "../app/data/product_features.parquet"
ESCI_EXAMPLES_PATH  = "../data/esci/shopping_queries_dataset_examples.parquet"
ESCI_PRODUCTS_PATH  = "../data/esci/shopping_queries_dataset_products.parquet"

# === Load Datasets ===
query_df = pd.read_parquet(SQID_QUERY_PATH)
product_df = pd.read_parquet(SQID_PRODUCT_PATH)
examples_df = pd.read_parquet(ESCI_EXAMPLES_PATH)
product_meta_df = pd.read_parquet(ESCI_PRODUCTS_PATH)

# === Merge 1: Examples with Query Embeddings ===
merged_df = examples_df.merge(query_df, on="query_id", how="inner", suffixes=('', '_query_embed'))

# === Merge 2: Add Product Embeddings ===
merged_df = merged_df.merge(product_df, on="product_id", how="inner", suffixes=('', '_product_embed'))

# === Merge 3: Add Product Metadata (title, desc, etc.) ===
merged_df = merged_df.merge(product_meta_df, on="product_id", how="left")

# === Save Final Merged Dataset for Training/Serving ===
output_path = "../data/merged/merged_dataset.parquet"
merged_df.to_parquet(output_path)

print(f"✅ Merged dataset saved to: {output_path}")
print(f"📦 Shape: {merged_df.shape}")
print(f"📊 ESCI Label distribution:\n{merged_df['esci_label'].value_counts()}")
