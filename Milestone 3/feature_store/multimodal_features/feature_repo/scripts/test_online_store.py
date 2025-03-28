from feast import FeatureStore
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

store = FeatureStore(".")

# 🔹 Try a valid ID (113370) and an invalid one (999999)
entity_df = pd.DataFrame({
    "query_id": [113370, 113379, 113381]
})

features = store.get_online_features(
    features=[
        "query_features_view:f0",
        "query_features_view:f1",
        "query_features_view:f2"
    ],
    entity_rows=entity_df.to_dict(orient="records")
).to_df()

print(features)
