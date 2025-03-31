

import os
import pandas as pd
import tensorflow_data_validation as tfdv
import matplotlib.pyplot as plt

# Define input paths
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/query/query_features_with_timestamp.csv"))
stats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts/StatisticsGen/statistics"))
schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts/SchemaGen/schema"))
anomalies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts/ExampleValidator/anomalies"))

# -----------------------
# Load Dataset for Preview
# -----------------------
df = pd.read_csv(data_path)
print("✅ Dataset Loaded")
print(df.info())
print(df.head())

# -----------------------
# Visualize Statistics
# -----------------------
print("✅ Loading Statistics")
stats = tfdv.load_statistics(stats_path)
tfdv.visualize_statistics(stats)
plt.show()

# -----------------------
# Visualize Schema
# -----------------------
print("✅ Loading Schema")
schema = tfdv.load_schema_text(os.path.join(schema_path, "schema.pbtxt"))
tfdv.display_schema(schema)
plt.show()

# -----------------------
# Visualize Anomalies
# -----------------------
print("✅ Loading Anomalies")
anomalies = tfdv.load_anomalies(anomalies_path)
tfdv.display_anomalies(anomalies)
plt.show()
