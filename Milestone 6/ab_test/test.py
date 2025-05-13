from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=["test"], return_tensors="pt", padding=True)
with torch.no_grad():
    emb = model.get_text_features(**inputs)

print(emb.shape)  # 🔁 OUTPUT: torch.Size([1, 512])
