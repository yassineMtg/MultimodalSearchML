from tensorflow_metadata.proto.v0 import anomalies_pb2
from google.protobuf import text_format

with open("artifacts/multimodal_search_pipeline/ExampleValidator/anomalies/17/Split-eval/SchemaDiff.pb", "rb") as f:
    anomalies = anomalies_pb2.Anomalies()
    anomalies.ParseFromString(f.read())
    print(text_format.MessageToString(anomalies))

print('\n##################################\n')

with open("artifacts/multimodal_search_pipeline/ExampleValidator/anomalies/17/Split-train/SchemaDiff.pb", "rb") as f:
    anomalies = anomalies_pb2.Anomalies()
    anomalies.ParseFromString(f.read())
    print(text_format.MessageToString(anomalies))
