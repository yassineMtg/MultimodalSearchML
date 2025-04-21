from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.types import ValueType
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/raw"))

query_features_path = os.path.join(BASE_DIR, "query_features_with_timestamp.parquet")

query = Entity(name="query_id", join_keys=["query_id"], value_type=ValueType.INT64)

query_features_source = FileSource(
    path=query_features_path,
    timestamp_field="event_timestamp"
)

schema = [Field(name=f"f{i}", dtype=Float32) for i in range(768)]

query_features_view = FeatureView(
    name="query_features_view",
    entities=[query],
    ttl=None,
    schema=schema,
    source=query_features_source,
    online=True,
)
