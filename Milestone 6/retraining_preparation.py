import pandas as pd
from datetime import datetime

# Load the collected queries
log_file = "./query_logs_for_retraining.csv"
df = pd.read_csv(log_file)

print("✅ Total logged queries:", len(df))
print("✅ Unique queries:", df["original_query"].nunique())

# Check for most frequent queries
print("\n=== Most frequent queries ===")
print(df["original_query"].value_counts().head(10))

# Check route selection distribution
print("\n=== Route usage ===")
print(df["selected_route"].value_counts())

# Prepare deduplicated queries for retraining dataset
df_clean = df.drop_duplicates(subset=["original_query"])
df_clean = df_clean.reset_index(drop=True)

# Add placeholder columns for labeling
df_clean["relevance_label"] = "TO_BE_ANNOTATED"
df_clean["category_label"] = "TO_BE_ANNOTATED"

# Save cleaned dataset
output_file = "./prepared_retraining_queries.csv"
df_clean.to_csv(output_file, index=False)
print(f"\n✅ Cleaned queries saved to {output_file}")

# Optional: Show sample
print("\n=== Sample of prepared dataset ===")
print(df_clean.head())
