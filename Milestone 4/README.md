
# Multimodal Search ML

---

# Milestone 4 (Model Training and Evaluation)

## Overview

This milestone focuses on the training and evaluation phase of the MultimodalSearchML pipeline. It builds on the output of Milestone 3 (data ingestion and preprocessing) and introduces a full MLOps training workflow using TensorFlow Extended (TFX).

The goal was not to achieve high accuracy, but to validate that the model training, evaluation, logging, and model versioning infrastructure work reliably and are modular, reproducible, and trackable.

Despite the absence of labeled data in the Shopping Queries Image Dataset (SQID), we generated synthetic binary labels to simulate a supervised training setting. This allowed us to demonstrate model development, metric tracking with MLflow, and energy usage tracking with CodeCarbon.

---

## Tools and Technologies Used

This milestone integrates several tools from the MLOps ecosystem to manage the training, tracking, and monitoring of the model lifecycle.

| Tools                         | Role in the Project                                                          |
|-------------------------------|------------------------------------------------------------------------------|
| **TFX (TensorFlow Extended)** | Used to build the training pipeline and run the `Trainer` component          |
| **MLflow**                    | Enabled experiment tracking for hyperparameters, metrics, and model versions |
| **CodeCarbon**                | Tracked the carbon emissions and energy consumption of the training process  |
| **Git & GitHub**              | Used for source code version control and collaborative development workflow  |
| **Python 3.9**                | Programming language used throughout the implementation                      |


---

## Dataset Overview and Preprocessing

The dataset used in this milestone is the [Shopping Queries Image Dataset (SQID)](https://github.com/Crossing-Minds/shopping-queries-image-dataset), which builds upon the original SQD dataset by including product images and CLIP-based multimodal embeddings.

From the available files in SQID, we used:

- `query_features.parquet`: This file contains 768-dimensional CLIP text embeddings for shopping queries.

For Milestone 4, we selected only the first **100 features** (`f0` to `f99`) for model input, due to hardware limitations that led to memory overflow when attempting to process all 768 features during training.

### Preprocessing Logic

I reused the preprocessing pipeline from Milestone 3, which performs the following:

- Parses and passes numerical features (`f0` to `f99`) as-is
- Applies a **synthetic binary labeling strategy**:
- `label = 1` if `f0 > 0`, otherwise `label = 0`

This approach simulates a supervised learning setup for pipeline validation purposes, as the SQID dataset does not contain labeled relevance information.

---

## Model Architecture and Training Setup

The model used in this milestone is a simple **Multi-Layer Perceptron (MLP)** built using TensorFlow Keras. It is designed to take in 100 CLIP text embedding features (`f0` to `f99`) and output a binary classification label.

### Model Structure

- **Input:** 100-dimensional vector (concatenation of features `f0` to `f99`)
- **Hidden Layers:**
- Dense layer with 256 units, ReLU activation
  - Dropout layer (0.3)
  - Dense layer with 128 units, ReLU activation
- **Output:** Dense layer with 1 unit, Sigmoid activation (binary classification)

The model was compiled with:

- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `Adam`
- **Metrics:** `accuracy`

### Training Configuration

- **Batch size:** 32
- **Epochs:** 50 (early stopping applied with patience = 5)
- **Validation split:** Provided via TFX pipeline's `eval_dataset`
- **Runner:** `LocalDagRunner` from TFX
- **Checkpoints:** Saved after every epoch via `ModelCheckpoint` callback

---

## Experiment Tracking with MLflow

To enable experiment tracking, MLflow was integrated into the TFX `Trainer` component. This allowed us to automatically and manually log:

- Training and validation metrics (`accuracy`, `val_accuracy`, `loss`, `val_loss`)
- Model parameters (`batch_size`, `epochs`, `input_dim`, `optimizer`)
- Model versioning through MLflow run IDs
- Comparison between different training runs using the MLflow UI

### MLflow Setup

- Experiment name: `milestone4_training`
- Tracking UI hosted locally via:
  ```bash
  mlflow ui --host 0.0.0.0 --port 5000
  ```
- Metrics and parameters are visible in the MLflow dashboard and can be compared across runs
- Autologging (mlflow.tensorflow.autolog()) was used for seamless integration with Keras

Sample Metrics Logged:

|Metric       |	Value   |
|-------------|---------|
|Accuracy     |	0.6756  |
|Val Accuracy |	0.7652  |
|Loss         |	0.6471  |
|Val Loss     |	0.5114  |

---

## Energy Efficiency Logging with CodeCarbon

To measure the environmental impact of the training process, the [CodeCarbon](https://mlco2.github.io/codecarbon/) library was integrated into the `run_fn()` method of the TFX Trainer.

The `EmissionsTracker` was configured to output logs into the pipeline’s artifact directory for each training run. This included detailed metrics such as energy consumed, CO₂ emissions, CPU power usage, and location-based carbon impact estimation.

### Key Results from emissions.csv

| Metric              | Value                     |
|---------------------|---------------------------|
| Duration            | ~6 seconds                |
| Emissions           | **0.000056 kg CO₂**       |
| CPU Power Used      | 42.5 W                    |
| RAM Power Used      | 11.4 W                    |
| CPU Energy Consumed | 0.00007 kWh               |
| Region              | Casablanca, Morocco       |
| CPU Model           | AMD Ryzen 7 7700          |
| Tracking Mode       | `machine` (local hardware)|

This demonstrates a growing awareness of sustainable ML practices and carbon-aware experimentation.

The output of the carboncode can be found in the following path
  ```mathematics
  ./tfx_pipeline/artifacts/training_pipeline/Trainer/model_run/6/emissions.csv
  ```



