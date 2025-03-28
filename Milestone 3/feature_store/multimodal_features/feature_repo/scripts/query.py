from feast import FeatureStore


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

store = FeatureStore(repo_path=".")

features = store.get_online_features(
    features=[
        "query_features_view:f0",
        "query_features_view:f1",
        "query_features_view:f2",
    ],
    entity_rows=[{"query_id": 113370}]
).to_df()

print(features)

