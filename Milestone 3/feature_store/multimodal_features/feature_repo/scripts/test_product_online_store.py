from feast import FeatureStore
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

store = FeatureStore(".")

# Replace these product_ids with valid ones from your dataset
sample_product_ids = ["B00N2E1UF0", "B0026RHYO8", "B00BU1WMM0"]

features = store.get_online_features(
    features=[
        "product_features_full_view:text_f0",
        "product_features_full_view:text_f1",
        "product_features_full_view:image_f0",
    ],
    entity_rows=[{"product_id": pid} for pid in sample_product_ids]
).to_df()

print(features)
