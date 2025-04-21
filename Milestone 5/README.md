
# Multimodal Search ML

---

# Milestone 5 (Model Deployment and Serving)

## Overview

The goal of this milestone is to productionize the MultimodalSearchML system built in earlier milestones. The system allows users to search for relevant products based on natural language queries using CLIP-based multimodal embeddings. Rather than training a new model, the system leverages a pretrained CLIP model (ViT-L/14) to encode both product information and user queries, and serves predictions via a FastAPI backend and a React-based frontend.

The entire system is modular, automated, and integrates tools like TFX, Feast, and DVC to handle data processing, feature storage, and versioning.

---

## ML System Architecture

The following architecture diagram shows the full machine learning system, including all tools and services involved in the pipeline and serving infrastructure:

IMAAAAAAGE

Diagram Key Components:

- DVC: Handles versioning of raw and processed datasets.

- TFX: Manages data ingestion and preprocessing using standard pipeline components.

- Feast: Stores and retrieves both online and offline features.

- CLIP: Encodes user queries and products into the same embedding space.

- FastAPI: Serves the model inference logic as an API.

- React UI: Provides an interactive user interface hosted on DigitalOcean.

- MLflow: Logs and tracks model configurations and metrics.

This architecture enables a reproducible, scalable, and modular MLOps pipeline for multimodal semantic search.

---

## Model Serving Mode

- Serving Type: On-demand
- Target: Human

The system uses an on-demand serving mode, where predictions are generated in real time as users interact with the React-based frontend. The model is served to humans through a FastAPI REST API. Each user query is encoded by the pretrained CLIP model (ViT-L/14) and compared to precomputed product embeddings using cosine similarity. The top relevant product matches are then returned.

This real-time interaction mode ensures minimal latency while maintaining high-quality retrieval without the need for additional model training.

