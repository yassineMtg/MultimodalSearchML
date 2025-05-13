import requests
import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
import warnings
warnings.filterwarnings("ignore")

# API base (local or remote)
API_BASE = "http://localhost:7860"

# Define diverse, bias-sensitive queries
queries = [
    "shoes for women",
    "shoes for men",
    "toys for kids",
    "sports equipment for seniors",
    "hijab",
    "christmas decorations",
    "ramadan gifts"
]

# Define a manual demographic attribute per query (proxy for audit)
query_demographics = {
    "shoes for women": "female",
    "shoes for men": "male",
    "toys for kids": "child",
    "sports equipment for seniors": "senior",
    "hijab": "muslim",
    "christmas decorations": "christian",
    "ramadan gifts": "muslim"
}

results = []

for query in queries:
    print(f"Running bias audit for query: {query}")

    response = requests.get(f"{API_BASE}/search_best?query={query}")

    if response.status_code == 200:
        data = response.json()

        for item in data["top_k_results"]:
            # Log with assumed demographic
            results.append({
                "query": query,
                "product_id": item["product_id"],
                "model_selected": data["route"],
                "demographic": query_demographics[query],
                "score": 1.0,  # Proxy score since retrieved
                "label_value": 1  # Mark as retrieved
            })
    else:
        print(f"Failed query: {query}")

# Create DataFrame
df_results = pd.DataFrame(results)

print("\n=== Raw Audit Results ===")
print(df_results.head())

# Save to CSV for reporting
df_results.to_csv("bias_audit_raw.csv", index=False)

# === Add dummy 'non-retrieved' records for each demographic ===
unique_demographics = df_results['demographic'].unique()
dummy_negatives = pd.DataFrame([{
    "query": f"dummy_{demo}",
    "product_id": f"dummy_{demo}_prod",
    "model_selected": "Dummy",
    "demographic": demo,
    "score": 0.0,
    "label_value": 0
} for demo in unique_demographics])

# Combine positives and dummy negatives
df_combined = pd.concat([df_results, dummy_negatives], ignore_index=True)

print("\n=== Combined Audit Dataset ===")
print(df_combined.head())

# === Aequitas Bias Audit ===
g = Group()
xtab, _ = g.get_crosstabs(df_combined, attr_cols=['demographic'], score_col='score', label_col='label_value')

b = Bias()
bias_metrics = b.get_disparity_predefined_groups(xtab, original_df=df_combined, ref_groups_dict={'demographic': 'male'})

print("\n=== Aequitas Bias Report ===")
print(bias_metrics.head())
print("\n✅ Available columns:", bias_metrics.columns.tolist())


# Save to CSV for report
bias_metrics.to_csv("bias_audit_aequitas_report.csv", index=False)
print("\n✅ Bias Audit Report saved to 'bias_audit_aequitas_report.csv'")
