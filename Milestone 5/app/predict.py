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

def predict_top_k(query: str, model, preprocess, device, metadata_df, embeddings, k=5):

    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        query_embedding = model.encode_text(text).cpu().numpy()

    scores = cosine_similarity(query_embedding, embeddings)[0]
    metadata_df["score"] = scores

    top_k = metadata_df.sort_values(by="score", ascending=False).head(k)

    return top_k.to_dict(orient="records")


