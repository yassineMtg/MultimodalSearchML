import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Paths ===
INPUT_PARQUET = "../../Milestone 5/data/processed/product_metadata_clean.parquet"
OUTPUT_EMBEDDINGS = "../../Milestone 5/data/processed/product_embeddings_l14.npy"

# === Load model ===
print("🔧 Loading CLIP ViT-L/14...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

# === Load product metadata ===
print("📦 Loading product metadata...")
df = pd.read_parquet(INPUT_PARQUET)
texts = df["title"] if "title" in df.columns else df["product_title"]

# === Encode with CLIP ===
print("🚀 Encoding products using ViT-L/14...")
embeddings = []

with torch.no_grad():
    for text in tqdm(texts, desc="Embedding products"):
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        embeddings.append(emb.cpu().numpy()[0])

# === Save ===
embeddings = np.array(embeddings)
np.save(OUTPUT_EMBEDDINGS, embeddings)

print(f"✅ Saved {embeddings.shape[0]} embeddings to {OUTPUT_EMBEDDINGS}")
