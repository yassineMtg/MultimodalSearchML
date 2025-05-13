import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import sys
import os

# Add parent path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_test.clip_models import clip_b32, processor_b32, clip_l14, processor_l14, device
from utils.query_rewriter import rewrite_query_llm

# === Paths ===
product_metadata_path = "../../Milestone 5/data/processed/product_metadata_clean.parquet"
product_embeddings_b_path = "../../Milestone 5/data/processed/product_embeddings_clean.npy"
product_embeddings_l_path = "../../Milestone 5/data/processed/product_embeddings_l14.npy"

# === Load metadata and both sets of embeddings ===
print("📦 Loading product metadata and embeddings...")
df = pd.read_parquet(product_metadata_path)
embeddings_b = np.load(product_embeddings_b_path)  # ViT-B/32
embeddings_l = np.load(product_embeddings_l_path)  # ViT-L/14

# === Select 10 random queries for demo ===
demo_queries = df.sample(10, random_state=42).reset_index(drop=True)

# === Encode Query ===
def encode_query(model, processor, query: str):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77  # Safe limit for CLIP ViT-L/14
    ).to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy()[0]

# === Perform A/B Search ===
print("🔍 Running local A/B test with rewritten queries...")

for idx, row in demo_queries.iterrows():
    query = row["title"] if "title" in row else row["product_title"]
    rewritten_query = rewrite_query_llm(query)

    emb_a = encode_query(clip_b32, processor_b32, rewritten_query)
    emb_b = encode_query(clip_l14, processor_l14, rewritten_query)

    sim_a = cosine_similarity([emb_a], embeddings_b)[0]
    sim_b = cosine_similarity([emb_b], embeddings_l)[0]

    top_a_ids = df.iloc[np.argsort(sim_a)[::-1][:5]]["product_id"].tolist()
    top_b_ids = df.iloc[np.argsort(sim_b)[::-1][:5]]["product_id"].tolist()

    print(f"\n🔹 Original Query [{idx+1}]: {query}")
    print(f"✏️ Rewritten: {rewritten_query}")
    print(f"🔵 Top-5 from ViT-B/32 (A): {top_a_ids}")
    print(f"🟢 Top-5 from ViT-L/14 (B): {top_b_ids}")
