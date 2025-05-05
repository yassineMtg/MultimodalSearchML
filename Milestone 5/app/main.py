# app/main.py

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.model_loader import load_clip_model
from app.predict import load_product_embeddings, predict_top_k
from app.predict_image import predict_from_image 
from app.llm_utils import rewrite_query_gemini

# Load CLIP model + product data
model, preprocess, device = load_clip_model()
metadata_df, embeddings = load_product_embeddings(
    "data/processed/product_embeddings_clean.npy",
    "data/processed/product_metadata_clean.parquet"
)

# FastAPI setup
app = FastAPI()

# CORS setup for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QueryRequest(BaseModel):
    query: str
    k: int = 20
    threshold: float = 0.35

@app.post("/predict")
def predict(request: QueryRequest):
    rewritten_query = rewrite_query_gemini(request.query)

    results = predict_top_k(
        query=rewritten_query,
        model=model,
        preprocess=preprocess,
        device=device,
        metadata_df=metadata_df,
        embeddings=embeddings,
        k=request.k,
        threshold=request.threshold,
    )

    return {
        "original_query": request.query,
        "rewritten_query": rewritten_query,
        "results": results
    }


@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...), k: int = 20, threshold: float = 0.05):  # lowered threshold
    image_bytes = await image.read()
    results = predict_from_image(
        image_bytes=image_bytes,
        model=model,
        preprocess=preprocess,
        device=device,
        metadata_df=metadata_df,
        embeddings=embeddings,
        k=k,
        threshold=threshold
    )
    return {"results": results}
