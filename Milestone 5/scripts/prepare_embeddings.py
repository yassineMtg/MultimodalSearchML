
# scripts/prepare_embeddings.py

import os
import numpy as np
import pandas as pd

BASE_DIR            = "data"

PRODUCTS_PATH       = os.path.join(BASE_DIR, "shopping_queries_dataset_products.parquet")
MERGED_PATH         = os.path.join(BASE_DIR, "merged/merged_dataset.parquet")
IMG_URL_PATH        = os.path.join(BASE_DIR, "product_image_urls.csv")
SUPP_IMG_URL_PATH   = os.path.join(BASE_DIR, "supp_product_image_urls.csv")

OUTPUT_EMBEDDINGS   = os.path.join(BASE_DIR, "processed/product_embeddings.npy")
OUTPUT_METADATA     = os.path.join(BASE_DIR, "processed/product_metadata.parquet")


def parse_embedding(embedding):
    """Parses a stringified NumPy array into actual NumPy array"""
    if isinstance(embedding, str):
        return np.fromstring(embedding.strip("[]").replace("\n", ""), sep=" ")
    elif isinstance(embedding, np.ndarray):
        return embedding
    else:
        return None


def main():
    print("🔍 Loading merged dataset...")
    df = pd.read_parquet(MERGED_PATH)

    print("🧠 Parsing CLIP text embeddings...")
    df["embedding"] = df["clip_text_features_product_embed"].apply(parse_embedding)
    df = df[df["embedding"].notnull()].reset_index(drop=True)

    print("📝 Selecting metadata columns...")
    metadata_cols = [
        "product_id", "product_title", "product_description", "product_bullet_point",
        "product_brand", "product_color"
    ]
    metadata_df = df[metadata_cols + ["embedding"]].copy()

    print("🖼️ Merging image URLs from product_image_urls and supp_product_image_urls...")
    if os.path.exists(IMG_URL_PATH) and os.path.exists(SUPP_IMG_URL_PATH):
        img_df = pd.read_csv(IMG_URL_PATH)
        supp_df = pd.read_csv(SUPP_IMG_URL_PATH)

        # Combine both image sources
        all_imgs = pd.concat([img_df, supp_df], ignore_index=True)

        # Group image URLs as list per product
        grouped_imgs = all_imgs.groupby("product_id")["image_url"].apply(list).reset_index()

        # Merge into metadata
        metadata_df = metadata_df.merge(grouped_imgs, on="product_id", how="left")

        # Optional: join into comma-separated string
        metadata_df["image_urls"] = metadata_df["image_url"].apply(
            lambda urls: ", ".join([str(u) for u in urls if isinstance(u, str)]) if isinstance(urls, list) else ""
        )
        metadata_df.drop(columns=["image_url"], inplace=True)
    else:
        print("⚠️ Warning: One or both image files missing. Skipping image URL merge.")

    print("💾 Saving embeddings and metadata...")
    embeddings = np.stack(metadata_df["embedding"].values)
    metadata_df.drop(columns=["embedding"], inplace=True)

    np.save(OUTPUT_EMBEDDINGS, embeddings)
    metadata_df.to_parquet(OUTPUT_METADATA, index=False)

    print("✅ Done! Embeddings and metadata saved.")


if __name__ == "__main__":
    main()
