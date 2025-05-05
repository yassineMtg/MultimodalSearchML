# scripts/prepare_embeddings.py

import os
import numpy as np
import pandas as pd
import clip
import torch
from tqdm import tqdm

BASE_DIR = "data"

CLEAN_META_PATH = os.path.join(BASE_DIR, "processed/product_metadata_reachable.parquet")
OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, "processed/product_embeddings_clean.npy")
OUTPUT_METADATA = os.path.join(BASE_DIR, "processed/product_metadata_clean.parquet")


def main():
    print("🔍 Loading CLEAN dataset columns only...")
    df_meta = pd.read_parquet(CLEAN_META_PATH, columns=[
        "product_id", "product_title", "product_description",
        "product_bullet_point", "product_brand", "product_color", "image_urls"
    ])
    print(f"✅ Loaded clean metadata: {df_meta.shape}")

    # ❌ Filter out movie-related titles using basic keyword filtering
    keywords_to_exclude = ["season", "episode", "trailer", "blu-ray", "film", "dvd", "tv", "series", "prime video"]
    before_filter = len(df_meta)
    df_meta = df_meta[~df_meta["product_title"].str.lower().str.contains("|".join(keywords_to_exclude), na=False)]
    after_filter = len(df_meta)
    print(f"🧹 Removed {before_filter - after_filter} suspected movie/media entries. Remaining: {after_filter}")

    print("🧠 Loading CLIP model (ViT-B/32) on cpu...")
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    titles = df_meta["product_title"].astype(str).tolist()

    print("🧹 Filtering titles using clip.tokenize()...")
    safe_titles = []
    safe_indices = []

    for idx, title in tqdm(enumerate(titles), total=len(titles), desc="Filtering titles"):
        try:
            tokens = clip.tokenize([title])
            if tokens.shape[1] <= 77:
                safe_titles.append(title)
                safe_indices.append(idx)
        except Exception:
            continue

    print(f"✅ {len(safe_titles)} safe titles selected.")

    df_meta_safe = df_meta.iloc[safe_indices].reset_index(drop=True)

    print("📝 Encoding safe titles using CLIP...")
    batch_size = 256
    all_embeddings = []

    for i in tqdm(range(0, len(safe_titles), batch_size), desc="Encoding batches"):
        batch = safe_titles[i:i + batch_size]
        tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens).cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)

    print("💾 Saving clean embeddings and metadata...")
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS, all_embeddings)
    df_meta_safe.to_parquet(OUTPUT_METADATA, index=False)

    print("✅ Done! 🎯 Clean embeddings and metadata are saved.")


if __name__ == "__main__":
    main()
