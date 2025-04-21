# app/model_loader.py
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device
