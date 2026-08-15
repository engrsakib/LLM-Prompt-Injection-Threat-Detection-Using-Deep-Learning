from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

class DistilBERTWrapper:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=7, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts, batch_size=16):
        enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        return out.logits.cpu()

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

