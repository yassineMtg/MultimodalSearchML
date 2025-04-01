
# Multimodal Search ML

---

# Milestone 3

## Overview

This milestone focuses on building the data pipeline for the MultimodalSearchML project. We automated the ingestion, validation, preprocessing, and preparation of the dataset using TFX, ensured proper data versioning with DVC, and managed features with Feast as our feature store.

---

## Data Ingestion

- **Library used**: TFX ExampleGen

- **Dataset used:** query_features_with_timestamp.csv

- **Task**: Ingest raw data into the pipeline for future steps (statistics, schema generation, validation, and transformation).

I created a TFX pipeline component for ingestion using CsvExampleGen. Then, I loaded the dataset from:

```bash
data/query/query_features_with_timestamp.csv
```

The dataset is already prepared with query_id, f0 to f767 features, and an event_timestamp column.

```python
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2

input_config = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name='train', pattern='query_features_with_timestamp.csv'),
])

example_gen = CsvExampleGen(input_base=data_path, input_config=input_config)
```

- The CsvExampleGen is used to automatically:

	- Ingest the CSV file

	- Convert it into TFRecords

	- Automatically split into train and eval sets

	- Store generated artifacts under artifacts/CsvExampleGen

- Results:

You can observe the ingested data under:

```bash
tfx_pipeline/artifacts/CsvExampleGen/examples/
```

and inside:

```mathematica
Split-train/
Split-eval/
```

Each containing generated data_tfrecord-* files.


