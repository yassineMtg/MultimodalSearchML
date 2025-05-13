import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root="/tmp")

# Define queries
queries = [
    "smart watch",
    "shoes for women",
    "toys for kids",
    "ramadan gifts",
    "hijab",
    "asdfghjkl"
]

# Encode queries as text
query_embeddings = []
query_labels = []

for query in queries:
    tokenized = clip.tokenize([query]).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokenized)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    query_embeddings.append(embedding.cpu().numpy()[0])
    query_labels.append(f"Query: {query}")

query_embeddings = np.array(query_embeddings)

# === Encode product titles as text ===
df = pd.read_parquet("../Milestone 5/data/processed/product_metadata_clean.parquet")  # Adjust path if needed
sampled_products = df.sample(10, random_state=42)  # Take 10 random products for demo

product_embeddings = []
product_labels = []

for _, row in sampled_products.iterrows():
    title = row.get("product_title", row.get("title", "unknown product"))
    tokenized = clip.tokenize([title]).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokenized)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    product_embeddings.append(embedding.cpu().numpy()[0])
    product_labels.append(f"Product {row['product_id']}")

product_embeddings = np.array(product_embeddings)

# Combine for TSNE
all_embeddings = np.vstack((query_embeddings, product_embeddings))
all_labels = query_labels + product_labels

# TSNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 8))
for i, label in enumerate(all_labels):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y, label=label if i < len(query_labels) else "", marker='o' if i < len(query_labels) else 'x')

plt.legend()
plt.title("TSNE Visualization of Queries and Products (Text side, ViT-L/14)")
plt.savefig("explainability_tsne.png")
plt.show()

# Similarity heatmap (query vs products)
sims = cosine_similarity(query_embeddings, product_embeddings)

plt.figure(figsize=(8, 6))
plt.imshow(sims, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(product_labels)), product_labels, rotation=90)
plt.yticks(range(len(query_labels)), query_labels)
plt.title("Similarity Heatmap (Text Queries vs Product Titles)")
plt.tight_layout()
plt.savefig("explainability_heatmap.png")
plt.show()

print("\n✅ Explainability visualizations saved as 'explainability_tsne.png' and 'explainability_heatmap.png'")
