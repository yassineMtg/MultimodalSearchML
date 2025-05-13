from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
import torch
import clip
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import requests
from io import BytesIO
from PIL import Image
import io
import mlflow
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import csv

# MLflow setup for local usage
mlflow.set_tracking_uri("file:./mlruns")

# === Load embeddings and metadata ===
def load_parquet_from_url(url: str) -> pd.DataFrame:
    print(f"📥 Loading parquet from {url}")
    response = requests.get(url)
    return pd.read_parquet(BytesIO(response.content))

def load_npy_from_url(url: str) -> np.ndarray:
    print(f"📥 Loading .npy from {url}")
    response = requests.get(url)
    return np.load(BytesIO(response.content), allow_pickle=True)

print("📦 Loading metadata and embeddings...")
df = load_parquet_from_url("https://huggingface.co/datasets/yassinemtg/amazon_smartsearch/resolve/main/product_metadata_clean.parquet")
embeddings_b = load_npy_from_url("https://huggingface.co/datasets/yassinemtg/amazon_smartsearch/resolve/main/product_embeddings_clean.npy")
embeddings_l = load_npy_from_url("https://huggingface.co/datasets/yassinemtg/amazon_smartsearch/resolve/main/product_embeddings_l14.npy")

# === Load CLIP models using clip library ===
print("🔧 Loading CLIP models using OpenAI's clip library...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_b32, preprocess_b32 = clip.load("ViT-B/32", device=device, download_root="/tmp")
clip_l14, preprocess_l14 = clip.load("ViT-L/14", device=device, download_root="/tmp")
print("✅ CLIP models loaded successfully on", device)

# === FastAPI App ===
app = FastAPI(title="A/B Search API")

# CORS setup for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Shared encoding logic ===
def encode_query(model, query: str):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]

# === Query rewriting stub ===
def rewrite_query_llm(query: str) -> str:
    return query.lower().strip()

# === Continual learning logging ===
def log_query_to_csv(query, rewritten, selected_route):
    log_file = "./query_logs_for_retraining.csv"
    header = ['timestamp', 'original_query', 'rewritten_query', 'selected_route']
    data = [datetime.now().isoformat(), query, rewritten, selected_route]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(data)
    except Exception as e:
        print(f"❌ Failed to log query: {e}")

# === Response schema ===
class SearchResponse(BaseModel):
    route: str
    rewritten_query: str
    top_k_results: list
    latency_ms: float

@app.get("/")
def read_root():
    return {"message": "API running, use /search_a, /search_b, /search_best, /search_best_image"}

# === Endpoint A: CLIP B/32 ===
@app.get("/search_a", response_model=SearchResponse)
def search_a(query: str = Query(..., min_length=3)):
    start = time.time()
    rewritten = rewrite_query_llm(query)
    emb = encode_query(clip_b32, rewritten)
    sims = cosine_similarity([emb], embeddings_b)[0]
    top_ids = df.iloc[np.argsort(sims)[::-1][:5]].to_dict(orient="records")
    latency = round((time.time() - start) * 1000, 2)

    with mlflow.start_run(run_name="Search A"):
        mlflow.log_param("query", query)
        mlflow.log_param("model", "ViT-B/32")
        mlflow.log_metric("latency_ms", latency)

    return SearchResponse(
        route="A (ViT-B/32)",
        rewritten_query=rewritten,
        top_k_results=top_ids,
        latency_ms=latency
    )

# === Endpoint B: CLIP L/14 ===
@app.get("/search_b", response_model=SearchResponse)
def search_b(query: str = Query(..., min_length=3)):
    start = time.time()
    rewritten = rewrite_query_llm(query)
    emb = encode_query(clip_l14, rewritten)
    sims = cosine_similarity([emb], embeddings_l)[0]
    top_ids = df.iloc[np.argsort(sims)[::-1][:5]].to_dict(orient="records")
    latency = round((time.time() - start) * 1000, 2)

    with mlflow.start_run(run_name="Search B"):
        mlflow.log_param("query", query)
        mlflow.log_param("model", "ViT-L/14")
        mlflow.log_metric("latency_ms", latency)

    return SearchResponse(
        route="B (ViT-L/14)",
        rewritten_query=rewritten,
        top_k_results=top_ids,
        latency_ms=latency
    )

# === Endpoint C: Best based on z-score ===
@app.get("/search_best", response_model=SearchResponse)
def search_best(query: str = Query(..., min_length=3)):
    start = time.time()
    rewritten = rewrite_query_llm(query)

    emb_a = encode_query(clip_b32, rewritten)
    emb_b = encode_query(clip_l14, rewritten)
    sims_a = cosine_similarity([emb_a], embeddings_b)[0]
    sims_b = cosine_similarity([emb_b], embeddings_l)[0]

    top_k = 5
    def z_score_top_k(sim_scores):
        top_k_vals = np.sort(sim_scores)[-top_k:]
        mean = np.mean(sim_scores)
        std = np.std(sim_scores)
        z_scores = (top_k_vals - mean) / (std + 1e-8)
        return np.mean(z_scores)

    score_a = z_score_top_k(sims_a)
    score_b = z_score_top_k(sims_b)

    if score_b > score_a:
        selected = "B (ViT-L/14)"
        sims = sims_b
    else:
        selected = "A (ViT-B/32)"
        sims = sims_a

    top_ids = df.iloc[np.argsort(sims)[::-1][:top_k]].to_dict(orient="records")
    latency = round((time.time() - start) * 1000, 2)

    # ✅ MLflow logging
    with mlflow.start_run(run_name="AB_Test_Query"):
        mlflow.log_param("query", query)
        mlflow.log_param("route", selected)
        mlflow.log_metric("score_a", score_a)
        mlflow.log_metric("score_b", score_b)
        mlflow.log_metric("latency_ms", latency)
        mlflow.log_metric("num_top_k_results", len(top_ids))

    # ✅ Log query for continual learning
    log_query_to_csv(query, rewritten, selected)

    return SearchResponse(
        route=selected,
        rewritten_query=rewritten,
        top_k_results=top_ids,
        latency_ms=latency
    )

# === Endpoint D: Best based on z-score (Image) ===
@app.post("/search_best_image", response_model=SearchResponse)
async def search_best_image(image: UploadFile = File(...)):
    start = time.time()

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor_b32 = preprocess_b32(pil_image).unsqueeze(0).to(device)
    image_tensor_l14 = preprocess_l14(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_a = clip_b32.encode_image(image_tensor_b32)
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_a = emb_a.cpu().numpy()[0]

        emb_b = clip_l14.encode_image(image_tensor_l14)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
        emb_b = emb_b.cpu().numpy()[0]

    sims_a = cosine_similarity([emb_a], embeddings_b)[0]
    sims_b = cosine_similarity([emb_b], embeddings_l)[0]

    top_k = 5

    def z_score_top_k(sim_scores):
        top_k_vals = np.sort(sim_scores)[-top_k:]
        mean = np.mean(sim_scores)
        std = np.std(sim_scores)
        z_scores = (top_k_vals - mean) / (std + 1e-8)
        return np.mean(z_scores)

    score_a = z_score_top_k(sims_a)
    score_b = z_score_top_k(sims_b)

    if score_b > score_a:
        selected = "B (ViT-L/14)"
        sims = sims_b
    else:
        selected = "A (ViT-B/32)"
        sims = sims_a

    top_ids = df.iloc[np.argsort(sims)[::-1][:top_k]].to_dict(orient="records")
    latency = round((time.time() - start) * 1000, 2)

    with mlflow.start_run(run_name="A/B Testing Search Image"):
        mlflow.log_param("query", "Image Query")
        mlflow.log_param("selected_route", selected)
        mlflow.log_metric("z_score_b32", score_a)
        mlflow.log_metric("z_score_l14", score_b)
        mlflow.log_metric("latency_ms", latency)

    return SearchResponse(
        route=selected,
        rewritten_query="Image Query",
        top_k_results=top_ids,
        latency_ms=latency
    )
