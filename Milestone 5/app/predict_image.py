# app/predict_image.py

import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

def predict_from_image(image_bytes, model, preprocess, device, metadata_df, embeddings, k=20, threshold=0.05):
    try:
        print("🔍 Loading image")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("🔍 Preprocessing image")
        image_input = preprocess(image).unsqueeze(0).to(device)

        print("🔍 Encoding image")
        with torch.no_grad():
            image_embedding = model.encode_image(image_input).cpu().numpy()

        print("🔍 Computing cosine similarity")
        scores = cosine_similarity(image_embedding, embeddings)[0]
        print("✅ Scores computed. Shape:", scores.shape)

        print("🔍 Creating DataFrame and adding scores")
        df = metadata_df.copy()
        df["score"] = scores
        print("🔍 Top 10 raw scores:", df["score"].nlargest(10).tolist())  # Debug log

        print("🔍 Sorting by score")
        df = df.sort_values(by="score", ascending=False)

        print("🔍 Filtering by threshold")
        df = df[df["score"] >= threshold]
        if df.empty:
            print("❌ No items after threshold filtering.")
            return []

        print("🔍 Filtering by image URL")
        df = df[df["image_urls"].notna() & df["image_urls"].str.startswith("http")]

        print("🔍 Filtering by metadata presence")
        has_metadata = (
            df["product_title"].notna() |
            df["product_description"].notna() |
            df["product_bullet_point"].notna()
        )
        df = df[has_metadata]

        print("🔍 Filtering blacklisted keywords")
        blacklist_keywords = [
            "movie", "film", "season", "episode", "blu-ray", "dvd",
            "kindle", "book", "publisher", "documentary", "novel",
            "tune", "video", "watch now", "stream", "cinema", "fiction"
        ]

        def contains_blacklist(text):
            text = str(text).lower()
            return any(keyword in text for keyword in blacklist_keywords)

        df = df[~df["product_title"].apply(contains_blacklist)]

        print("✅ Final sorting and deduplication")
        df = df.sort_values(by="score", ascending=False).drop_duplicates(subset="product_id")
        df["score"] = df["score"].round(2)

        return df.head(k).to_dict(orient="records")

    except Exception as e:
        print("❌ Error during image prediction:", e)
        return []
