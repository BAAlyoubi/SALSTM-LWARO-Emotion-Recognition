from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    for inputs, lengths, labels in loader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        logits, _ = model(inputs, lengths)
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    y_true = np.array(targets)
    y_pred = np.array(predictions)
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

