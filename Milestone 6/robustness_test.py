import requests
import pandas as pd

API_BASE = "http://localhost:7860"

# Define robustness test queries (perturbed, adversarial, edge cases)
queries = [
    "smart watch",            # normal
    "smaart wach",            # typo
    "sm@rt watch!!!",         # special chars
    "smartwatch ",            # trailing space
    "    smart    watch",     # multiple spaces
    "what is the best watch", # long natural query
    "asdfghjkl",              # gibberish
    "",                       # empty
    "!!!???",                 # only symbols
    "شاشة ذكية"               # Arabic (different language)
]

results = []

for query in queries:
    print(f"\nTesting robustness for query: '{query}'")

    response = requests.get(f"{API_BASE}/search_best?query={query}")

    if response.status_code == 200:
        data = response.json()

        results.append({
            "query": query,
            "rewritten_query": data.get("rewritten_query", ""),
            "route": data.get("route", ""),
            "top_product_ids": [item["product_id"] for item in data["top_k_results"]],
            "num_results": len(data["top_k_results"]),
            "latency_ms": data.get("latency_ms", 0)
        })
    else:
        print(f"Failed query: {query}")
        results.append({
            "query": query,
            "rewritten_query": "",
            "route": "",
            "top_product_ids": [],
            "num_results": 0,
            "latency_ms": 0
        })

# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(results)
print("\n=== Robustness Test Summary ===")
print(df_results)

# Save for reporting
df_results.to_csv("robustness_test_results.csv", index=False)
print("\n✅ Robustness test results saved to 'robustness_test_results.csv'")
