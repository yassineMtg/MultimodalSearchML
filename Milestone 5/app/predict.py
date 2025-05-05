# app/predict.py

import torch
import clip
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_product_embeddings(embedding_path: str, metadata_path: str):
    """
    Loads pre-parsed product embeddings and metadata separately.
    Embeddings are stored in .npy format, metadata in .parquet.
    """
    metadata_df = pd.read_parquet(metadata_path)
    embeddings = np.load(embedding_path)
    return metadata_df, embeddings


def predict_top_k(query: str, model, preprocess, device, metadata_df, embeddings, k=20, threshold=0.60):
    import torch
    import clip
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Encode the query
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        query_embedding = model.encode_text(text).cpu().numpy()

    # Cosine similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]
    metadata_df = metadata_df.copy()
    metadata_df["score"] = scores

    # Apply score threshold
    filtered = metadata_df[metadata_df["score"] >= threshold]

    # Require image
    filtered = filtered[filtered["image_urls"].notna() & filtered["image_urls"].str.startswith("http")]

    # Require at least one non-empty metadata field
    has_metadata = (
        filtered["product_title"].notna() |
        filtered["product_description"].notna() |
        filtered["product_bullet_point"].notna()
    )
    filtered = filtered[has_metadata]

    # ✂️ Remove entries with blacklisted keywords
    blacklist_keywords = [
        "movie", "film", "season", "episode", "blu-ray", "dvd",
        "kindle", "book", "publisher", "documentary", "novel",
        "tune", "video", "watch now", "stream", "cinema", "fiction"
    ]

    def contains_blacklist(text):
        text = str(text).lower()
        return any(keyword in text for keyword in blacklist_keywords)

    filtered = filtered[~filtered["product_title"].apply(contains_blacklist)]

    # Sort by score
    filtered = (
        filtered.sort_values(by="score", ascending=False)
        .drop_duplicates(subset="product_id")
    )
    filtered["score"] = filtered["score"].round(2)

    return filtered.to_dict(orient="records")