# app/model_loader.py

import torch
import clip
from sentence_transformers import SentenceTransformer

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_bert_model():
    model = SentenceTransformer('models/all-MiniLM-L6-v2')  # Light and fast BERT
    return model
