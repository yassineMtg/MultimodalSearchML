# Multimodal Search ML  

---

# Milestone 1

## Business Idea as an ML Problem

In the fast-growing landscape of e-commerce, customers expect fast, accurate, and personalized search experiences. However, traditional keyword-based search engines often fall short when handling vague, visual, or ambiguous queries — such as "Nike shoes with blue swoosh" or "Apple logo hoodie" — where intent cannot be fully captured by text alone.

This project addresses this challenge by leveraging multimodal machine learning techniques to improve product search. Specifically, it uses both textual and visual data from the Shopping Queries Image Dataset (SQID) and extracts semantic embeddings via CLIP, a vision-language model.

The machine learning problem is framed as a multimodal ranking/retrieval task, where the system computes a semantic similarity score between a user query and multiple product candidates, each represented through both their text description and image. The goal is to return a ranked list of products most relevant to the user's intent.

This approach enables improved understanding of visual and textual attributes, making the search more intuitive, robust to vocabulary gaps, and capable of handling open-ended or visually grounded queries.

---

## Business Case

Modern e-commerce platforms face increasing pressure to deliver accurate and engaging search experiences. When users enter queries, they expect results that match both their functional and visual expectations. For example, a search for "slim fit striped shirt" should return products that actually have the right cut and visual pattern — not just any shirt with matching keywords.

However, most search engines still rely on traditional information retrieval methods that focus only on textual metadata. These methods often fail when:

- The product image contradicts the description.

- Visual details are not fully captured by the product title or description.

- The user’s intent is visual or ambiguous.

This project aims to improve the product search experience by using multimodal machine learning, combining product titles, descriptions, and images to better understand relevance. It leverages the Shopping Queries Image Dataset (SQID), an enriched dataset built on top of Amazon’s Shopping Queries Dataset (SQD), which includes product images and pretrained CLIP embeddings.

By understanding both the visual and textual context of each product, we can return more meaningful and satisfying search results — enhancing customer experience, increasing conversion rates, and reducing bounce rates.

---

## Business Value 

Using machine learning, particularly multimodal models, provides substantial business value for modern e-commerce platforms. Traditional search engines based on keyword matching often fail to capture subtle user intent or visual expectations, leading to irrelevant results and poor user experience.

By introducing CLIP-based multimodal search, the system gains the ability to:

- Interpret complex queries more semantically, not just syntactically.

- Leverage both product images and text to rank relevance more effectively.

- Adapt to user preferences over time without manual rule-based tuning.

In summary, applying machine learning in this context helps companies stay competitive by improving user satisfaction, driving revenue, and enabling smarter automation in product search and recommendation systems.

---

## Data Overview

This project uses the Shopping Queries Image Dataset (SQID), an extension of the original Shopping Queries Dataset (SQD) released by Amazon for the KDDCup’22 challenge. SQID enriches the SQD dataset by including product images and precomputed CLIP embeddings, allowing for multimodal learning and ranking tasks.

-----

Key dataset components:

- Queries: User search terms in natural language (e.g., "men’s red sneakers").

- Products: Product listings with titles, descriptions, brand, color, etc.

- Images: Main product image for each item, linked via URL.

- Labels (ESCI): Relevance judgments:

  - E – Exact match

  - S – Substitute

  - C – Complement

  - I – Irrelevant

- Embeddings: Precomputed CLIP text and image embeddings for each product.

Dataset stats:

- ~190,000 product listings

- ~1.1M query-product pairs

- Covers US, ES, and JP locales (this project focuses on the US locale)

The dataset is publicly available at:
🔗 [https://github.com/Crossing-Minds/shopping-queries-image-dataset](https://github.com/Crossing-Minds/shopping-queries-image-dataset)

---

## Project Archetype

Based on the archetype framework taught in class, this project falls under the Software 2.0 category.

Software 2.0 refers to systems where traditional code is replaced or augmented by machine-learned logic — typically models trained on data. Instead of writing explicit rules for ranking products or understanding user queries, the system learns from examples and uses embeddings (text + image) to drive behavior.

In this project:

- The ranking function is learned through semantic similarity between query and product embeddings.

- The rules for what is relevant are not hardcoded — they are inferred from CLIP’s understanding of visual-textual alignment.

- The system adapts and improves based on the distribution of queries and product listings.

This is in contrast to:

- Autonomous systems: which act in the world (e.g., robotics).

- Human-in-the-loop: where a person is part of the feedback cycle (e.g., labeling data or approving results).

Thus, this project is a clear example of Software 2.0 — where ML replaces handcrafted relevance ranking logic.

---

## Feasibility Analysis

### Literature Review

This project is inspired by two key papers:

- Shopping Queries Dataset (SQD) – [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)

  - Introduced by Amazon, this dataset provides a large-scale benchmark for product search using query-product pairs labeled by relevance. It defined the ESCI labels (Exact, Substitute, Complement, Irrelevant) and proposed ranking, classification, and substitution detection tasks.

- Shopping Queries Image Dataset (SQID) – [arXiv:2405.15190](https://arxiv.org/abs/2405.15190)

  - SQID builds on SQD by enriching it with product images and pretrained embeddings (text and image) using CLIP. It supports multimodal learning and highlights the value of combining text + image for better product ranking.

### Baseline Model

**Baseline Model Summary**

| Model Name                             | Developer                                    | Purpose                                      | Performance (NDCG) |
|----------------------------------------|----------------------------------------------|----------------------------------------------|--------------------|
| CLIP (ViT-L/14)                        | OpenAI                                       | Multimodal embedding for image-text matching | ~0.82 (SQID paper) |
| SBERT (MiniLM-L12-v2)                  | Hugging Face                                 | Text-only semantic similarity model          | ~0.83              |
| ESCI Baseline (MS MARCO Cross-Encoder) | Amazon                                       | Fine-tuned text-only ranker for SQD          | **0.85+**          |

The baseline model used in this project is CLIP (Contrastive Language-Image Pretraining), specifically the clip-vit-large-patch14 variant, available via Hugging Face.

- It provides joint embeddings for both text (user queries) and images (product photos).

- Embeddings can be compared using cosine similarity to estimate semantic relevance.

This model is:

- Open-source

- Available as a pretrained binary

- Does not require retraining to get useful results (zero-shot setup)

---

## Evaluation Metrics

This project focuses on ranking product search results by relevance to a given user query. Therefore, the most appropriate evaluation metrics are:

### 1. Normalized Discounted Cumulative Gain (NDCG)

- Why?: NDCG measures the quality of a ranked list by rewarding correct items at higher positions.

- How it works:

  - Takes into account the position of relevant items in the list.

  - ESCI labels are mapped to graded relevance scores:

    - E (Exact) = 1.0

    - S (Substitute) = 0.1

    - C (Complement) = 0.01

    - I (Irrelevant) = 0.0

  - Used in: The official SQD and SQID benchmarks.

### 2. Precision @ K (Optional)

- Measures how many of the top-K retrieved products are truly relevant (e.g., Exact or Substitute).

- Useful for evaluating short result lists, like top 5 or top 10.

### 3. Cosine Similarity (Intermediate)

- Not a final metric, but used internally to compare query and product embeddings (via CLIP).

- It’s how we score and rank product candidates before computing NDCG.

These metrics provide a robust framework to evaluate how well the system understands and ranks product relevance based on multimodal inputs.

---

# Milestone 2 [Demo](https://huggingface.co/spaces/yassinemtg/MultimodalSearchML)

This milestone implements a functional **Streamlit** prototype **(PoC)** of the multimodal product search engine using OpenAI's CLIP model. The system supports natural language queries and returns visually and semantically relevant product images.

## **Live Demo**

[Hugging Face MultimodalSearchML Space](https://huggingface.co/spaces/yassinemtg/MultimodalSearchML)

---

## Objectives

- Build an interactive UI to test query-to-image search
- Use real product data from the SQID dataset
- Deploy the app publicly on Hugging Face Spaces

---

## Model & Embedding

- **Model**: CLIP (`ViT-L/14`) from OpenAI
- **Embedding**:
  - Product image features: precomputed and stored in `products_sample.csv`
  - Query text: encoded in real-time using CLIP

---

## Dataset

- Source: [SQID (Shopping Queries Image Dataset)](https://github.com/Crossing-Minds/shopping-queries-image-dataset)
- Sample size: 15,000 products
- Format:
  - `product_id`
  - `product_title`
  - `image_url`
  - `clip_image_features` (768-dim)

---

## 💻 PoC UI Overview

- Built with **Streamlit**
- Accepts a free-text user query
- Encodes the query with CLIP
- Computes cosine similarity between query and all image embeddings (batched)
- Ranks top 10 matches and displays:
  - Product image
  - Title / ID
  - Normalized match score (%)

---

## Deployment

The working PoC is live on Hugging Face Spaces:

[View Demo](https://huggingface.co/spaces/yassinemtg/MultimodalSearchML)

---

## End-to-End Validation

| Component           | Status |
|---------------------|--------|
| Query input         | ✅     |
| CLIP query encoding | ✅     |
| Embedding similarity| ✅     |
| Image ranking       | ✅     |
| Visual output       | ✅     |

### Test Case

![Image 1](./Milestone%202/images/test1.png)

---

![Image 2](./Milestone%202/images/test2.png)

---

![Image 3](./Milestone%202/images/test3.png)

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

Results:

- You can observe the ingested data under:

```bash
tfx_pipeline/artifacts/CsvExampleGen/examples/
```

and inside:

```mathematica
Split-train/
Split-eval/
```

Each containing generated data_tfrecord-* files.

---

## Data Validation 

To automatically generate statistics, infer a data schema, and detect anomalies in the ingested query dataset before applying transformations or training models.

I added three essential components:

1. StatisticsGen: Computes descriptive statistics on the ingested data.

2. SchemaGen: Automatically infers the schema from the statistics.

3. ExampleValidator: Detects data anomalies and schema drift.

```python
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator

# Compute statistics
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# Infer schema
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

# Validate dataset against the inferred schema
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
```

Pipeline Flow:

- Statistics are automatically computed from the TFRecords generated by ExampleGen.

- SchemaGen infers a schema without any manual intervention.

- ExampleValidator checks if:

    - The data conforms to the inferred schema.

    - There are missing values, unusual distributions, or unexpected types.

Results:

- Statistics are stored under:

```bash
tfx_pipeline/artifacts/StatisticsGen/statistics/
```

- Schema is stored under:

```bash
tfx_pipeline/artifacts/SchemaGen/schema/
```

- This is the output of our schema: (schame.pbtxt)

```mathematica
feature {
  name: "event_timestamp"
  type: BYTES
  domain: "event_timestamp"
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "f0"
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "f1"
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
.....
.....
feature {
  name: "f767"
  type: FLOAT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
feature {
  name: "query_id"
  type: INT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
  shape {
    dim {
      size: 1
    }
  }
}
string_domain {
  name: "event_timestamp"
  value: "2025-03-27 07:01:04.332726+00:00"
}
```

- Anomalies report is stored under:

```bash
tfx_pipeline/artifacts/ExampleValidator/anomalies/
```

NB:

- This process is fully automated thanks to TensorFlow Data Validation (TFDV).
- The schema file (schema.pbtxt) will serve as input for the next Transform step.
- Any anomalies detected can be analyzed in TensorBoard or directly via the generated artifacts.


---

## Data Preprocessing and Feature Engineering

Prepare the dataset for model training by applying transformations, selecting useful features, and managing missing data using TensorFlow Transform (TFX Transform).

I built a dedicated preprocessing module responsible for:

- Selecting only useful columns (768 embedding features).

- Handling potential missing data.

- Preparing the data for modeling in a scalable and production-friendly way. 

Preprocessing Module (preprocessing.py):

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    # Keep only feature columns f0 to f767
    outputs = {}
    for i in range(768):
        col = f"f{i}"
        outputs[col] = inputs[col]

    # Keep the event timestamp as is
    outputs['event_timestamp'] = inputs['event_timestamp']

    return outputs
```

TFX Transform Component (in the pipeline):

```python
from tfx.components import Transform

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=module_file,  # preprocessing.py script
)
```

What this step achieves:

- Reduces the dataset to only the needed features (f0 - f767 + timestamp).
- Ensures data is formatted consistently.
- Scales and prepares data in a reproducible way for future serving.
- The transformed dataset is automatically split into:
	- Split-train/
	- Split-eval/

Output Directories:

- Transform Graph:

```bash
tfx_pipeline/artifacts/Transform/transform_graph/
```

- Transformed Dataset:

```bash
tfx_pipeline/artifacts/Transform/transformed_examples/
```

---

## Feature Store

In this step, I integrated Feast as a feature store to manage and serve features for the MultimodalSearchML project. The feature store plays a crucial role in centralizing, versioning, and serving feature data for both training and serving machine learning models.

Objectives:

- Organize, version, and serve query and product dataset features.
- Enable consistent feature access during both training and inference.
- Prepare for later integration with the future TFX pipeline model training phase.

Data Used:

- query_features_flat.csv
- product_features_flat.parquet

These datasets contain extracted features representing query embeddings and product embeddings computed using CLIP models.

I installed Feast in a dedicated environment:

```bash
pip install feast==0.40.1
```

Then, initialized the feature repository:

```bash
feast init feature_store/multimodal_features
```

I defined two main FeatureViews:

1. Query Feature View from query_features_flat.csv
2. Product Feature View from product_features_flat.parquet

Example from query_features_view.py:

```python
query_feature_view = FeatureView(
    name="query_features_view",
    entities=[query_entity],
    ttl=Duration(seconds=86400),
    schema=[Field(name=f"f{i}", dtype=Float32) for i in range(768)],
    source=query_source
)
```

Example from product_features_view.py:

```python
product_feature_view = FeatureView(
    name="product_features_view",
    entities=[product_entity],
    ttl=Duration(seconds=86400),
    schema=[Field(name=f"f{i}", dtype=Float32) for i in range(768)],
    source=product_source
)
```
 
#### Feature Store Registration

Once the FeatureViews were defined, we applied them:

```bash
feast apply
```

Feast registered:

- 2 Entities (query_id and product_id)
- 2 FeatureViews
- Features with embedding dimensions: (f0 to f767 for queries) and (f0 to f1535 for products)

Feature Materialization: 

I materialized features to be ready for offline and online serving:

```bash
feast materialize-incremental $(date -u +%Y-%m-%d)
```

Feast then generated:

- Offline feature data (for training pipelines)
- Online feature data (for future model serving)

Summary: 

- Both datasets were successfully managed in Feast.
- Redis was configured as an online store (development purpose).
- The setup ensures that during model training or serving, the exact same feature values are retrieved as during feature generation.

---

## Data Versioning (DVC)

In this step, I integrated DVC (Data Version Control) to manage and version the raw data used throughout the milestone. DVC provides a Git-like workflow for datasets, making sure that every experiment or pipeline execution always uses the correct version of the data.

Objectives

    Track the query and product datasets systematically.

    Avoid redundant storage.

    Enable reproducibility of experiments.

    Prepare the project for collaborative work and future experiments.

I versioned the raw data folder containing:

    query_features_flat.csv

    query_features_with_timestamp.csv

    product_features_flat.parquet

    product_image_urls.csv

    supp_product_image_urls.csv

I initialized DVC inside the project directory:

```bash
dvc init
```

This created the .dvc/ folder and the configuration files necessary to start versioning data.

Then, for tracking the dataset, I use the following command to track the whole folder:

```bash
dvc add Milestone\ 3/data/raw/
```

DVC created:

- raw.dvc file to track changes.
- .dvc/cache/ for storing dataset versions efficiently.

I also configured a local remote to store data artifacts:

```bash
dvc remote add -d localremote ~/dvcstore
dvc remote modify localremote type local
```

This ensures all future dataset versions will be pushed to ~/dvcstore safely.

The datasets were pushed into the local remote:

```bach
dvc push
```

This command:

- Stored the dataset version into the ~/dvcstore directory.

- Allowed us to share datasets without duplicating the data.

---

## Ingestion and Preprocessing Pipeline (TFX)

In this step, I designed and implemented a fully functional TFX (TensorFlow Extended) pipeline for ingesting, validating, and preparing the query dataset for modeling. We focused on separating ingestion, validation, and preprocessing clearly while ensuring the pipeline is clean, professional, and reproducible.

### Pipeline Overview

This pipeline included the following TFX components:

1. CsvExampleGen

2. StatisticsGen

3. SchemaGen

4. ExampleValidator

5. Transform

Each component generated useful artifacts under the artifacts/ directory automatically, maintaining the TFX structure of numbered runs and split directories (train/eval).

### CsvExampleGen

This component ingests the query_features_with_timestamp.csv dataset.

```python
input_config = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name="train", pattern="query_features_with_timestamp.csv"),
])
example_gen = CsvExampleGen(input_base=data_path, input_config=input_config)
```

It automatically:

- Split the dataset.

- Converted it into TFRecords under:

```swift
tfx_pipeline/artifacts/CsvExampleGen/examples/1/Split-train/
tfx_pipeline/artifacts/CsvExampleGen/examples/1/Split-eval/
```

### StatisticsGen

It computed descriptive statistics of the dataset:

```python
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
```

Generated:

- FeatureStats.pb containing complete statistics of all columns.
- Split under:

```swift
artifacts/StatisticsGen/statistics/{run_id}/Split-train/FeatureStats.pb
```

### SchemaGen

Automatically inferred the data schema:

```python
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
```

Generated:

- schema.pbtxt defining data types, expected values, and feature constraints.

- Saved under:

```bash
artifacts/SchemaGen/schema/{run_id}/schema.pbtxt
```

### ExampleValidator

Validated dataset against the inferred schema:

```python
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
```

Generated:

- SchemaDiff.pb showing anomalies and schema drifts.

- Stored under:

```swift
artifacts/ExampleValidator/anomalies/{run_id}/Split-train/SchemaDiff.pb
```

### Transform (Preprocessing)

I used a custom preprocessing module to:

- Remove unwanted features.

- Scale and normalize input features.

- Keep only the f0 to f767 (768 clip-text embeddings).

Example code:

```python
def preprocessing_fn(inputs):
    return {
        f"f{i}": inputs[f"f{i}"] for i in range(768)
    }
```

TFX Transform component:

```python
transform = Transform(
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    module_file=module_file,
)
```

Generated:

- transformed_examples/ (TFRecords ready for modeling)

- transform_graph/ containing transformation logic for serving


### Full Pipeline Execution

We executed the pipeline with:

```bash
python tfx_pipeline/scripts/run_ingest_query_pipeline.py
```

It completed successfully and populated the following:

- All component artifacts

- Automatically versioned runs (run_id folders)

- Split-aware artifacts (train/eval)

Example Final Pipeline Structure (auto-generated by TFX):

```css
artifacts/
    CsvExampleGen/
    StatisticsGen/
    SchemaGen/
    ExampleValidator/
    Transform/
```

---

## Data Visualization [Notebook](https://github.com/yassineMtg/MultimodalSearchML/blob/main/Milestone%203/notebooks/pipeline_summary.ipynb)

After running the ingestion and preprocessing pipeline, we prepared a dedicated Jupyter Notebook to:

1. Explore TFX-generated artifacts.

2. Visualize statistics, schema, and anomalies.

3. Confirm data correctness before training.

Notebook Objective:

Load pipeline artifacts automatically.

- Show summary statistics.

- Visualize feature distributions.

- Display schema.

- Report anomalies detected by ExampleValidator.

- Prepare the pipeline for potential monitoring (TensorBoard or other tools).

As It showed, every component's artifact is generated under:

```swift
artifacts/
    CsvExampleGen/examples/{run_id}/Split-*/ 
    StatisticsGen/statistics/{run_id}/Split-*/FeatureStats.pb
    SchemaGen/schema/{run_id}/schema.pbtxt
    ExampleValidator/anomalies/{run_id}/Split-*/SchemaDiff.pb
    Transform/...
```

This allowed us to dynamically load latest runs using helper functions like:

```python
def get_latest_subdir(directory):
    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {directory}")
    return max(subdirs, key=os.path.getmtime)
```

### Visualization of Dataset Statistics

I used tensorflow_data_validation (tfdv) to load and visualize statistics:

```python
train_stats = tfdv.load_statistics(os.path.join(stats_path, "Split-train", "FeatureStats.pb"))
tfdv.visualize_statistics(train_stats)
```

Output:

- Rich interactive charts

- Feature distributions

- Missing value analysis

- Data type distributions

### Schema Visualization

We also displayed the schema:

```python
schema = tfdv.load_schema_text(os.path.join(schema_path, "schema.pbtxt"))
tfdv.display_schema(schema)
```

Schema gave us:

- Inferred types

- Domains (min/max)

- Presence constraints

### Anomalies Detection

ExampleValidator anomalies were visualized as:

```python
anomalies = tfdv.load_anomalies_text(os.path.join(anomalies_path, "SchemaDiff.pb"))
tfdv.display_anomalies(anomalies)
```

Result:

- Quickly spotted missing values, type mismatches, or distribution drifts

---

# Milestone 4 (Model Training and Evaluation)

## Overview

This milestone focuses on the training and evaluation phase of the MultimodalSearchML pipeline using real-world relevance labels from the Amazon ESCI dataset. It builds on Milestone 3, which handled data ingestion and preprocessing, and advances the pipeline with a robust training setup using TensorFlow Extended (TFX), MLflow, and CodeCarbon.

I used multi-class labels (Exact, Substitute, Complement, Irrelevant) from the ESCI dataset. This allowed for meaningful model training, accurate performance evaluation, and more realistic experimentation.

---

## Tools and Technologies Used

This milestone integrates several tools from the MLOps ecosystem to manage the training, tracking, and monitoring of the model lifecycle.

| Tools                         | Role in the Project                                                          |
|-------------------------------|------------------------------------------------------------------------------|
| **TFX (TensorFlow Extended)** | Powers the end-to-end training pipeline and handles modular orchestration    |
| **MLflow**                    | Tracks experiments, logs training metrics, and stores model versions         |
| **CodeCarbon**                | Logs energy usage and carbon emissions of model training                     |
| **Git & GitHub**              | Maintains version control and collaborative development                      |
| **DVC**                       | Tracks and manages dataset versions (e.g., ESCI data)                        |
| **Python 3.9**                | Used to implement preprocessing, training logic, and orchestration code      |

---

##  Project Structure & Cookiecutter Template

This project was initially scaffolded using a modified version of the **Cookiecutter Data Science** structure, then adapted to fit milestone-based developmen>

The core ideas from Cookiecutter — such as separation of `data/`, `notebooks/`, `scripts/`, and `models/` — are reflected across the milestones.

---

## Dataset Overview and Preprocessing

The dataset used in this milestone is the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset), which provides real multi-class relevance labels for query-product pairs. The ESCI labels include:

- **E**: Exact match

- **S**: Substitute

- **C**: Complement

- **I**: Irrelevant

This allowed us to simulate a realistic product search environment, using actual labels rather than synthetic ones.

We combined this dataset with the query CLIP embeddings from the Shopping Queries Image Dataset (SQID) used in Milestone 3. The following files were used:

- query_features_with_timestamp.csv (from Milestone 3): CLIP text embeddings for shopping queries

- shopping_queries_dataset_examples.parquet: Query–product pairs with real esci_labels

### Merging Strategy

- The two datasets were merged on the common query_id field.

- The esci_label column was mapped to integer labels as follows:

```python
label_map = {'I': 0, 'C': 1, 'S': 2, 'E': 3}
```

- The resulting file (merged_queries.csv) was saved in the data/processed/ directory and used as input to the pipeline.

### Preprocessing Logic

The Transform component in TFX was updated to:

- Pass all 768 features (f0 to f767) unchanged

- Accept the real label column directly without synthetic generation

- Ensure compatibility with sparse_categorical_crossentropy for multi-class classification

This marks a significant upgrade over the earlier synthetic-label strategy, making training outcomes more interpretable and relevant to the underlying task.

---

## Model Architecture and Training Setup

We designed a multi-class classification model using TensorFlow Keras to predict the ESCI relevance label (E, S, C, I) based on the 768-dimensional CLIP text embeddings of shopping queries.

### Model Structure

- **Input**: 768-dimensional CLIP embedding vector (f0 to f767)

- **Architecture**:

    - Dense layer with 256 units, ReLU activation

    - Dropout layer (0.3)

    - Dense layer with 128 units, ReLU activation

    - Output Layer: Dense layer with 4 units, Softmax activation

### Model Compilation

- Loss Function: sparse_categorical_crossentropy (used with integer labels 0–3)

- Optimizer: adam

- Metrics: accuracy

### Training Configuration

- **Batch size:** 32
- **Epochs:** 50 (early stopping applied with patience = 5)
- **Steps per epoch**: 200 (controlled via TrainArgs)
- **Validation split:** Validation: Provided via **eval_dataset** in the TFX pipeline
- **Checkpoints:** Saved after every epoch via `ModelCheckpoint` callback

This architecture balances simplicity and scalability, offering a strong baseline while keeping the model TFX-friendly and resource-efficient.

---

## Experiment Tracking with MLflow

MLflow was integrated into the Trainer component of the TFX pipeline to enable end-to-end experiment tracking for model development. The tracking setup captures all key aspects of the training run.

### What Was Logged

- Training and Validation Metrics

    - accuracy, val_accuracy, loss, val_loss

- Hyperparameters

    - batch_size, epochs, input_dim, optimizer

- Model versions

    - Saved under individual MLflow run IDs for comparison
    
Both manual logging (mlflow.log_param) and autologging (mlflow.tensorflow.autolog()) were used to ensure full visibility of model behavior.

### MLflow Setup

- Experiment name: `milestone4_training`
- Tracking UI hosted locally via:
  ```bash
  mlflow ui --host 0.0.0.0
  ```
- Tracking Source: Model training was launched via LocalDagRunner inside the TFX pipeline.

### Sample Results from MLflow

|Metric       |	Value   |
|-------------|---------|
|Accuracy     |	0.6756  |
|Val Accuracy |	0.7652  |
|Loss         |	0.6471  |
|Val Loss     |	0.5114  |

These metrics reflect training with real ESCI labels and indicate the model's ability to generalize relevance decisions across multiple product-query pairs.

---

## Energy Efficiency Logging with CodeCarbon

To measure the environmental impact of the training process, the [CodeCarbon](https://mlco2.github.io/codecarbon/) library was integrated into the `run_fn()` method of the TFX Trainer.

The `EmissionsTracker` was configured to output logs into the pipeline’s artifact directory for each training run. This included detailed metrics such as energy consumed, CO₂ emissions, CPU power usage, and location-based carbon impact estimation.

### Key Results from emissions.csv

| Metric              | Value                      |
|---------------------|----------------------------|
| Duration            | ~6 seconds                 |
| Emissions           | **0.000056 kg CO₂**        |
| CPU Power Used      | 42.5 W                     |
| RAM Power Used      | 11.4 W                     |
| CPU Energy Consumed | 0.00007 kWh                |
| Region              | Casablanca, Morocco        |
| CPU Model           | AMD Ryzen 7 7700           |
| Tracking Mode       | `machine` (local hardware) |

This demonstrates a growing awareness of sustainable ML practices and carbon-aware experimentation.

The output of the carboncode can be found in the following path
  ```mathematics
  ./tfx_pipeline/artifacts/training_pipeline/Trainer/model_run/6/emissions.csv
  ```

---

# Milestone 5 (Model Deployment and Serving)

## Overview

The goal of this milestone is to productionize the MultimodalSearchML system built in earlier milestones. The system allows users to search for relevant products based on natural language queries using CLIP-based multimodal embeddings. Rather than training a new model, the system leverages a pretrained CLIP model (ViT-B/32) to encode both product information and user queries, and serves predictions via a FastAPI backend and a React-based frontend.

The entire system is modular, automated, and integrates tools like TFX, Feast, and DVC to handle data processing, feature storage, and versioning.

---

## ML System Architecture

The following architecture diagram shows the full machine learning system, including all tools and services involved in the pipeline and serving infrastructure:

![Architecture](./Milestone%205/images/ML_architecture.png)

### Component Descriptions :

- **CSV / Parquet Files**: Raw query and product data in tabular format. These files contain embeddings, metadata, and product image links.

- **DVC (Data Version Control)**: Tracks versions of input data and metadata. Connected to Google Drive for remote storage.

- **Google Drive**: Remote DVC storage used to store raw data files for reproducibility and sharing.

- **Redis**: Serves as the online feature store for fast retrieval of product embeddings.

- **TFX (TensorFlow Extended)**: Used for building the end-to-end ML pipeline: ingestion, validation, transformation, trainer, and pusher. It outputs the final model to be deployed.

- **MLflow**: Handles model tracking, experiment logging, and metric visualization. Integrated inside the TFX trainer.

- **GitHub**: Hosts the source code and CI/CD pipeline.

- **GitHub Actions**: Automates the deployment of the backend and frontend to Hugging Face and cPanel respectively.

- **FastAPI**: Lightweight Python API framework used to serve predictions and expose endpoints.

- **CLIP (ViT-B/32)**: Pretrained model used for encoding queries and product metadata into embeddings.

- **Gemini LLM**: Used for query rewriting to enhance semantic understanding before matching.

- **Docker**: Containers are used to package the FastAPI service for deployment.

- **Hugging Face Spaces**: Hosts the Dockerized FastAPI backend.

- **ReactJS**: Frontend application that users interact with.

- **cPanel**: Used to host the React frontend publicly.

Each arrow in the diagram illustrates data or control flow — from dataset versioning to real-time inference served on Hugging Face and visualized in the React app.

---

## Model Serving Mode

In this system, we implemented two serving modes:

🔹 1. On-Demand Serving to Humans (via UI)

    A React.js frontend lets users submit:

        Text queries

        Image uploads

    These are served to the backend for inference, and matching products are returned instantly.

🔹 2. On-Demand Serving to Machines (via API)

    A FastAPI backend exposes two main endpoints:

        POST /predict for text queries

        POST /predict-image for image-based search

    The backend is accessible as a public API hosted on Hugging Face Spaces.

    Can be queried programmatically using tools like curl or axios.

🔧 Deployment Setup

    The FastAPI backend is containerized using Docker.

    The API is deployed on Hugging Face via Dockerfile and .huggingface.yml.

    The frontend is deployed separately on a cloud server (cPanel).

---

## Model Service Development

The model service was designed to serve on-demand, low-latency predictions based on multimodal embeddings from the CLIP model. Instead of training a new model, we use the pretrained ViT-B/32 CLIP model to encode queries and products. This design ensures a consistent, production-ready service for image–text matching tasks.

### Backend Implementation (FastAPI)

- **Framework**: The FastAPI framework was chosen for its speed, simplicity, and OpenAPI integration.

- **Endpoints**:

    - POST /predict: Accepts a query string and returns the top-k matching products using cosine similarity over embeddings.

    - POST /predict-image: Accepts an image upload, embeds it via CLIP, and returns ranked product matches.

- **LLM Rewriter**: The backend optionally integrates Gemini LLM to rewrite natural language queries before encoding to improve semantic matching.

- **Redis**: Used for temporary embedding caching and fast retrieval operations.

- **DVC + Google Drive**: Data access and versioning are managed through DVC, with remote storage pointing to Google Drive.

### CLIP Model Integration

- **Model**: ViT-B/32, loaded via the clip Python package.

- **Usage**:

    - Both product descriptions and queries/images are encoded using CLIP.

    - Embeddings are precomputed and compared using cosine similarity.

### Model Packaging

- The full backend is containerized using Docker and deployed on Hugging Face Spaces.

- The Docker image includes:

    - All Python dependencies via requirements.lock

    - FastAPI app in app/main.py

    - CLIP model loading and pre/post-processing utils

- The container is automatically built and redeployed via GitHub Actions CI/CD on every push.

---

## Model Serving Runtime

The deployed MultimodalSearchML system leverages cloud infrastructure to serve predictions in real-time:

### Backend Runtime Environment

- **Hosting Platform**: Hugging Face Spaces

- **Serving Framework**: FastAPI with automatic OpenAPI docs

- **Containerization**: Docker is used to isolate and run the backend service

### Runtime Setup

- The service is launched using a custom Dockerfile and requirements.txt.

- The runtime environment pulls the latest code and models from GitHub on each deploy.

- On every new push to main, GitHub Actions triggers automatic container rebuild and deployment.

### Performance Features

- Low-latency serving with Redis caching for embeddings

- On-the-fly embedding using CLIP for both image and text

- LLM-based query rewriting to improve semantic accuracy before matching

This setup enables both human (React UI) and machine (public API) clients to interact with the model in a scalable and production-ready runtime environment.

---

## Front-End Client Development

The front-end application provides an interactive user interface that allows users to perform multimodal search through both text and image inputs. It was built using ReactJS with modern best practices in component-based architecture and state management.

### Core Features

- **Text-based Search**: Users can input natural language queries describing the product they are looking for. The input is processed and sent to the FastAPI backend for similarity-based retrieval using CLIP embeddings.

- **Image-based Search**: Users can upload an image to find visually similar products. The image is embedded using the CLIP model and matched against product embeddings.

- **Result Display**: Retrieved products are displayed as responsive cards containing:

    Product title

    Thumbnail image

    Relevance score (cosine similarity)

    “More Info” button for detailed view

- **Search Customization Controls**: The UI includes toggles for setting:

    top_k (number of results)

    threshold (minimum similarity score)

    Whether to allow text/image input independently

- **Navigation and Routing**: Implemented using **react-router-dom**, allowing clean navigation between the search results page and the product detail page.

### Tech Stack

| Technology           | Purpose                                  |
| -------------------- | ---------------------------------------- |
| **ReactJS**          | Front-end framework                      |
| **Axios**            | API requests to FastAPI                  |
| **React Router DOM** | Client-side routing                      |
| **Tailwind CSS**     | Styling and responsive layout            |
| **React Icons**      | Icons for UI feedback                    |
| **State Hooks**      | Search input and result state management |

### Hosting

The frontend is hosted via cPanel, deployed manually through file upload and auto-refresh. The final UI is fully responsive and publicly accessible [Here](https://smartsearchml.yassinemaatougui.tech/)

The backend is hosted via Hugging Face spaces. It is accessible [Here](https://huggingface.co/spaces/yassinemtg/smartsearch-api)

- Screenshot Example

![screenshot](./Milestone%205/images/ui.png)

---

## Packaging and Containerization

To ensure portability, reproducibility, and isolation of the backend inference service, the FastAPI server was packaged into a Docker container.

### Dockerization Workflow

1. Dockerfile

A custom Dockerfile was written to:

- Set up a minimal Python environment using an official python:3.10-slim base image

- Install all dependencies from a pinned requirements.txt

- Copy the application codebase into the container

- Set the working directory and expose the serving port

- Launch the FastAPI server using uvicorn

Key Dockerfile commands:

```bash
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

2. .dockerignore

Unnecessary files such as local datasets, **.git**, and cache directories are excluded from the container image build context.

3. Testing Locally

The Docker image was tested using:

```bash
docker build -t multimodal-backend .
docker run -p 7860:7860 multimodal-backend
```

The API was successfully served on http://localhost:7860.

### Benefits of Containerization

- Ensures consistent runtime environment across development, testing, and deployment

- Makes backend easily deployable to cloud platforms like Hugging Face or DigitalOcean

- Reduces environment conflicts and simplifies dependency management

---

## CI/CD Integration and Deployment

The MultimodalSearchML project supports continuous integration and deployment (CI/CD) to streamline the release of updates to both the backend and frontend services.

### CI/CD for Backend (FastAPI)

- **Platform**: GitHub Actions

- **Target**: Hugging Face Spaces

Every push to the main branch automatically triggers the backend deployment pipeline via a .github/workflows/deploy.yml file.

**Key Steps in CI/CD Workflow:**

1. Checkout repository

2. Set up Python environment

3. Install dependencies from requirements.txt

4. Build and push Docker container

5. Trigger Hugging Face deployment using .huggingface.yml

**Hugging Face Integration:**

```yaml
# .huggingface.yml
title: MultimodalSearchML
sdk: docker
emoji: 🔍
```

This automation ensures that any committed changes to the main branch result in a refreshed, containerized FastAPI app being redeployed live on Hugging Face Spaces.

---

### Deployment of Frontend (React)

The React frontend was deployed manually via cPanel:

- The build output from npm run build was uploaded to the public_html folder on the hosting server.

- The interface is now publicly available via your domain (insert link if available).

- Any frontend update is handled by rebuilding locally and replacing the previous build on the server.

### Authentication & Security Notes

- Environment secrets should be defined in GitHub repository settings under Settings > Secrets and variables.

- Hugging Face deployment uses a personal access token (HF_TOKEN) securely stored in the repo secrets.

---

# Milestone 6 (Model Testing, Evaluation, Monitoring and Continual Learning)

This milestone focuses on evaluating, auditing, and analyzing the deployed MultimodalSearchML system beyond accuracy metrics. It covers unseen data evaluation, online A/B testing, bias auditing, robustness testing, and model explainability.

---

## Model Evaluation on Unseen Data

### Objective:

Evaluate the system using **unseen product data split (80/20 train/test split)**.

### Methodology:

- Used unseen data split from our dataset.
- Performed similarity search using ViT-B/32.
- Calculated:
  - **Top-5 Accuracy**
  - **Mean Reciprocal Rank (MRR)**

### Results:

- Top-5 Accuracy: 14.77%
- Mean Reciprocal Rank (MRR): 14.56%

### Conclusion:

Model shows **moderate accuracy on unseen data**, with room for improvement.

---

## Online A/B Testing

### Objective:

Evaluate two models online using **A/B Testing framework integrated in the API**:
- **Model A:** ViT-B/32
- **Model B:** ViT-L/14

### Methodology:

- Created a FastAPI endpoint `/search_best` that dynamically selects the best model per query using **z-score normalized top-k similarity**.
- Integrated the endpoint with the deployed **React frontend**.
- User can see which model was used per query.

### Links:

- [A/B Testing App (FastAPI on HF Space)](https://yassinemtg-ab-testing.hf.space)

- [A/B Testing App (React on cloud)](https://smartsearchml.yassinemaatougui.tech/ab-test)

### Conclusion:

The system can dynamically select the best model **per user query, based on online similarity evaluation**.

---

## Testing Beyond Accuracy

### Bias Auditing

#### Methodology:

- Used **Aequitas toolkit**.
- Sent predefined demographic-sensitive queries (e.g., `"shoes for women"`, `"ramadan gifts"`, `"toys for kids"`).
- Created proxy positives (`label_value=1`) and added **dummy negatives (`label_value=0`)**.
- Analyzed representation and coverage per demographic.

#### Results:

- See: `bias_audit_aequitas_report.csv`
- Observations:
    - Underrepresentation observed for certain demographics.
    - Junk queries still returned products.
    - Need for **input filtering and demographic-aware training**.

#### Script:

- `bias_audit_aequitas.py`

---

### Robustness Testing

#### Methodology:

- Tested the system under perturbations:
    - Typos, slang, spacing.
    - Gibberish.
    - Irrelevant queries.
    - Multilingual input.

#### Results:

- See: `robustness_test_results.csv`
- Observations:
    - Model is **robust to typos, spacing, and verbose queries**.
    - **Fails to reject gibberish or junk inputs.**
    - Acceptable **multilingual handling**.

#### Script:

- `robustness_test.py`

---

### Model Explainability & Interpretability

#### Methodology:

- Used **TSNE projection** of query and product title embeddings (text side only).
- Created **cosine similarity heatmaps**.
- Used CLIP ViT-L/14 text encoder.

#### Results:

| Figure                                    | Description                                                    |
|-------------------------------------------|----------------------------------------------------------------|
| ![Heatmap](./Milestone%206/explainability_heatmap.png)  | Similarity heatmap between queries and products                |
| ![TSNE](./Milestone%206/explainability_tsne.png)        | TSNE projection of queries and products in the embedding space |

#### Observations:

- Clear clustering between similar queries and products.
- **Gibberish queries placed isolated, validating model behavior**.
- The visualization supports model explainability for semantic search.

#### Script:

- `explainability_tsne_umap.py`

---

## 4. MLflow Experiment Logging (Optional, Integrated)

### Actions:

- Integrated MLflow tracking inside the **A/B testing API for query logging and model decision recording**.
- Logged metrics:
    - Query.
    - Selected route.
    - Latency.
    - Top returned products.

#### MLflow UI (Local):

- Run using: `mlflow ui`
- Access at: `http://localhost:5000`

---

## Model Monitoring and Continual Learning

### Data Monitoring

#### Methodology:

- Used MLflow as the main tool to monitor:

    - Query lengths.

    - Model selection (route).

    - Latency per request.

    - Top-k results returned.

- Tracked every user query through the API directly integrated with MLflow.

#### Observations:

- Observed variations in latency depending on the model selected (ViT-L/14 being slower).

- Query length monitoring showed input variations between short and verbose queries.

---

## Continual Trigger (CT) and Continuous Delivery (CD)

### Objective:

Simulate a Continual Learning Trigger and Delivery pipeline based on the logged queries.

### Methodology:

- Collected all queries into a structured CSV log (query_logs_for_retraining.csv).

- Implemented automatic retraining trigger (continual_trigger_retrain.py) that:

    - Checks if the number of logged queries exceeds 100 queries threshold.

    - Simulates the retraining process by generating a marker file with timestamp (simulated_retrained_model_YYYYMMDD_HHMMSS.txt).

    - Clears the query log file for the next cycle.

### Results:

- Successfully simulated automatic retraining triggering and model version update.

- Query logs are cleared after every simulated retraining to maintain clean data flow.

### Script:

- continual_trigger_retrain.py

### Logs:

- Queries logged in: query_logs_for_retraining.csv

- Simulated retrained model files saved in: simulated_models/

---

## Conclusion

- The MultimodalSearchML system was extended to support advanced testing, evaluation, monitoring, and lifecycle management.

- Successfully fulfilled all Milestone 6 requirements, including:

    - Unseen data evaluation.

    - Online A/B Testing.

    - Bias auditing.

    - Robustness testing.

    - Explainability and interpretability.

    - Query and model monitoring with MLflow.

    - Simulated CT/CD retraining trigger and model delivery.

- The project is now ready for future enhancements, like integrating real retraining pipelines and advanced drift detection tools.

---
