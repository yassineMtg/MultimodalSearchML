from feast import FileSource, FeatureView, Entity, Field
from feast.types import Float32
from feast.types import ValueType
from datetime import timedelta
import pandas as pd

# Entity
product = Entity(name="product_id", join_keys=["product_id"], value_type=ValueType.STRING)

# File source
product_features_source = FileSource(
    path="data/product_features_flat.parquet",
    timestamp_field="event_timestamp"
)
# Dynamically build schema
sample = pd.read_parquet("data/product_features_flat.parquet", engine="pyarrow")
text_dim = len([col for col in sample.columns if col.startswith("text_f")])
image_dim = len([col for col in sample.columns if col.startswith("image_f")])
schema = [Field(name=f"text_f{i}", dtype=Float32) for i in range(text_dim)] + \
         [Field(name=f"image_f{i}", dtype=Float32) for i in range(image_dim)]

print("✅ Total schema fields registered:", len(schema))
print("✅ Sample fields:", schema[:5])

# Feature View
product_features_view = FeatureView(
    name="product_features_full_view",  # temporary name
    entities=[product],
    ttl=timedelta(days=365),
    schema=schema,
    source=product_features_source,
    online=True,
)
