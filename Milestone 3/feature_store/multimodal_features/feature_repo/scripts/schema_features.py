from feast import FeatureStore

store = FeatureStore(".")
fv = store.get_feature_view("product_features_view")
print([field.name for field in fv.schema])
