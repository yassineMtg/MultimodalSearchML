---
title: MultimodalSearchML
emoji: 🔍
colorFrom: pink
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: true
sdk_version: 1.43.2
---

# 🔍 Multimodal Product Search (CLIP-Powered)

This demo app lets you semantically search products using text queries like "black sneakers" or "white hoodie with logo"; Matching both **product images** and **CLIP-based embeddings**.

It uses the [CLIP ViT-L/14](https://github.com/openai/CLIP) model to compute similarity between your query and real product image features from the SQID dataset.

## 🚀 How It Works

- You enter a **text query**
- The app encodes it into a CLIP embedding
- It compares your query with precomputed **product image embeddings**
- Shows the **top 10 matched products** (image + match score)

## ⚙️ Tech Stack

- Streamlit for UI
- PyTorch + OpenAI CLIP (`ViT-L/14`)
- Preprocessed product data from the [SQID dataset](https://github.com/Crossing-Minds/shopping-queries-image-dataset)

## 📦 Dataset

The sample CSV used (`products_sample.csv`) includes:
- Product IDs
- Image URLs
- CLIP image embeddings

NB: Embeddings are precomputed to keep the demo lightweight.

## ✨ Try It Out

- Click the "Search" button after typing a query.
- Only top relevant results (based on cosine similarity) are shown.