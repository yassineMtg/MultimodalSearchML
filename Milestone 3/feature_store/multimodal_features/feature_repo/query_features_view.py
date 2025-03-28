from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

query = Entity(name="query_id", join_keys=["query_id"])

query_features_source = FileSource(
    path="data/query_features_with_timestamp.parquet",
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
