from transformers import CLIPProcessor, CLIPModel
import torch

print("🔧 Loading CLIP models...")

# Model A: CLIP ViT-B/32
clip_b32 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_b32 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Model B: CLIP ViT-L/14
clip_l14 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor_l14 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip_b32.eval()
clip_l14.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_b32.to(device)
clip_l14.to(device)

print("✅ CLIP models loaded successfully.")
