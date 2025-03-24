import streamlit as st
import pandas as pd
import torch
import clip
import numpy as np
import torch.nn.functional as F
import time
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load dataset
df = pd.read_csv("products_sample.csv")
df = df.dropna(subset=["image_url", "clip_image_features"])

# Clean + parse embeddings
def parse_embedding(s):
    try:
        s = s.replace('\n', '').strip('"').strip()
        emb = np.fromstring(s.strip("[]"), sep=' ')
        return emb if len(emb) == 768 else None
    except:
        return None

products, embeddings = [], []
for _, row in df.iterrows():
    emb = parse_embedding(row["clip_image_features"])
    if emb is not None:
        products.append(row)
        embeddings.append(torch.tensor(emb, dtype=torch.float32))

if not embeddings:
    st.error("🚫 No valid image embeddings found.")
    st.stop()

# Stack embeddings to batch
image_tensor = torch.stack(embeddings).to(device)
image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

# Streamlit UI
st.title("🔍 Multimodal Product Search")
query = st.text_input("Enter your search query:", "black sneakers")

if st.button("Search"):
    with st.spinner("Searching for best matches..."):
        time.sleep(1)

        with torch.no_grad():
            # Encode and normalize the query
            tokens = clip.tokenize([query]).to(device)
            query_embed = model.encode_text(tokens)
            query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)

            # Batch cosine similarity
            scores = torch.matmul(query_embed, image_tensor.T).squeeze().cpu().numpy()

            # Normalize score to 0–100%
            min_s, max_s = scores.min(), scores.max()
            to_percent = lambda s: 100 * (s - min_s) / (max_s - min_s)

            # Rank and display top results
            top = sorted(zip(products, scores), key=lambda x: x[1], reverse=True)[:100]

            st.subheader("Top Matching Products:")
            displayed = False
            for product, raw_score in top:
                score_percent = to_percent(raw_score)
                if score_percent >= 60:
                    if not displayed:
                        st.subheader(f"Results for: {query}")
                        displayed = True

                    st.image(product["image_url"],
                            caption=f"{product.get('product_title', product['product_id'])} — Match: {score_percent:.2f}%",
                            width=400)
            if not displayed:
                st.warning("No strong matches found for this query.")


