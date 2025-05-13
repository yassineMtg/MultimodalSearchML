import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIG ===
PRODUCT_EMBEDDINGS_PATH = "../../Milestone 5/data/processed/product_embeddings_clean.npy"
PRODUCT_METADATA_PATH = "../../Milestone 5/data/processed/product_metadata_clean.parquet"
TOP_K = 5  # for Top-K Accuracy

# === 1. Load Data ===
print("📦 Loading product metadata and embeddings...")
product_metadata = pd.read_parquet(PRODUCT_METADATA_PATH)
product_embeddings = np.load(PRODUCT_EMBEDDINGS_PATH)

assert len(product_metadata) == len(product_embeddings), "Mismatch in metadata and embeddings length"

# === 2. Train/Test Split ===
print("✂️ Splitting into train/test sets (80/20)...")
train_meta, test_meta, train_embeds, test_embeds = train_test_split(
    product_metadata,
    product_embeddings,
    test_size=0.2,
    random_state=42
)

# === 3. Simulate Query-Based Search ===
print("🔍 Running CLIP-based similarity search and evaluation...")

def compute_top_k_accuracy(test_embeddings, test_metadata, train_embeddings, train_metadata, k=5):
    correct = 0
    mrr = 0.0
    total = len(test_metadata)

    for i in tqdm(range(total), desc="Evaluating queries"):
        query_embedding = test_embeddings[i].reshape(1, -1)
        true_product_id = test_metadata.iloc[i]["product_id"]

        similarities = cosine_similarity(query_embedding, train_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_product_ids = train_metadata.iloc[top_k_indices]["product_id"].tolist()

        # Top-K Accuracy
        if true_product_id in top_k_product_ids:
            correct += 1

        # MRR (Reciprocal Rank)
        try:
            rank = top_k_product_ids.index(true_product_id) + 1
            mrr += 1 / rank
        except ValueError:
            pass  # true product not in top-K

    top_k_accuracy = correct / total
    mrr_score = mrr / total
    return top_k_accuracy, mrr_score

top_k_acc, mrr_score = compute_top_k_accuracy(
    test_embeds, test_meta, train_embeds, train_meta, k=TOP_K
)

print("\n📊 Evaluation Results:")
print(f"✅ Top-{TOP_K} Accuracy: {top_k_acc:.4f}")
print(f"✅ Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
