from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def extract_mfcc(audio_path: str | Path, n_mfcc: int = 40, sample_rate: int = 16000) -> np.ndarray:
    import librosa

    signal, sr = librosa.load(audio_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)


def extract_bert_embeddings(texts: Iterable[str], model_name: str = "bert-base-uncased") -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    outputs = []
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(pooled.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(outputs)


def extract_resnet_frame_embeddings(frame_paths: Iterable[str | Path], model_name: str = "resnet50") -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.DEFAULT if model_name == "resnet50" else None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms() if weights else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            embeddings.append(model(tensor).squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(embeddings)
