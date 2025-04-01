
# Multimodal Search ML

---

# Milestone 3

## Overview

This milestone focuses on building the data pipeline for the MultimodalSearchML project. We automated the ingestion, validation, preprocessing, and preparation of the dataset using TFX, ensured proper data versioning with DVC, and managed features with Feast as our feature store.

---

## Ingestion of Raw Data

- **Library used**: TFX ExampleGen

- **Dataset used:** query_features_with_timestamp.csv

- **Task**: Ingest raw data into the pipeline for future steps (statistics, schema generation, validation, and transformation).

We created a custom query_pipeline.py where the pipeline logic is written.

We used CsvExampleGen() to ingest the dataset located at:
