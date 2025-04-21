from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_loader import load_clip_model
from app.predict import load_product_embeddings, predict_top_k

model, preprocess, device = load_clip_model()

metadata_df, embeddings = load_product_embeddings(
    "data/processed/product_embeddings.npy", "data/processed/product_metadata.parquet"
)

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/predict")
def predict(request: QueryRequest):
    results = predict_top_k(
        query=request.query,
        model=model,
        preprocess=preprocess,
        device=device,
        metadata_df=metadata_df,
        embeddings=embeddings,
        k=request.k,
    )
    return {"results": results}
