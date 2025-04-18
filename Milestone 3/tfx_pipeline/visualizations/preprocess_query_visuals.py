# tfx_pipeline/visualizations/preprocess_query_visuals.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tfx.orchestration import metadata
from tfx.types import standard_artifacts
from tfx.types.artifact_utils import get_split_uri

# Dynamically locate artifacts
ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts/multimodal_search_pipeline/"))

def visualize_statistics():
    csv_examplegen_dir = os.path.join(ARTIFACT_DIR, "CsvExampleGen", "examples")
    if not os.path.exists(csv_examplegen_dir):
        print("❌ CsvExampleGen artifacts not found.")
        return

    # Example: load one CSV generated by CsvExampleGen (will vary depending on splits)
    for split in os.listdir(csv_examplegen_dir):
        split_path = os.path.join(csv_examplegen_dir, split)
        if os.path.isdir(split_path):
            example_file = os.path.join(split_path, os.listdir(split_path)[0])
            df = pd.read_csv(example_file)
            print("✅ Loaded example CSV:", example_file)
            print(df.head())
            df.describe().to_csv("data_summary.csv")
            df.hist(figsize=(20, 20))
            plt.tight_layout()
            plt.savefig("data_distribution.png")
            print("✅ Statistics and histogram saved.")
            break
    else:
        print("❌ No split found inside CsvExampleGen.")

if __name__ == "__main__":
    visualize_statistics()
