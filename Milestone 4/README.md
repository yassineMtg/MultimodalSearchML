
# Multimodal Search ML

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



