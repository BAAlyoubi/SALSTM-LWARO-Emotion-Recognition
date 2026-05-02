from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MultimodalNPZDataset, collate_samples
from .evaluate import Metrics, evaluate
from .model import SALSTMClassifier
from .optimizer import LWAROOptimizer
from .utils import ensure_dir, load_config, resolve_device, set_seed


def build_model(config: dict, input_dim: int) -> SALSTMClassifier:
    model_config = config["model"]
    return SALSTMClassifier(
        input_dim=input_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(config["data"]["num_classes"]),
        num_layers=int(model_config.get("num_layers", 1)),
        dropout=float(model_config.get("dropout", 0.25)),
        bidirectional=bool(model_config.get("bidirectional", True)),
    )


def train_once(config: dict, train_path: str | Path, valid_path: str | Path) -> Metrics:
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    train_dataset = MultimodalNPZDataset(train_path)
    valid_dataset = MultimodalNPZDataset(valid_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_samples,
    )

    model = build_model(config, train_dataset.input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    output_dir = ensure_dir(config["training"].get("output_dir", "outputs"))
    best_f1 = -1.0
    stale_epochs = 0

    for _ in tqdm(range(int(config["training"]["epochs"])), desc="Training"):
        model.train()
        for inputs, lengths, labels in train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        metrics = evaluate(model, valid_loader, device)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["training"].get("patience", 8)):
                break

    return evaluate(model, valid_loader, device)


def run_search(config: dict, train_path: str | Path, valid_path: str | Path) -> tuple[dict[str, float], Metrics]:
    search_config = config["search"]

    def objective(params: dict[str, float]) -> float:
        trial = dict(config)
        trial["model"] = dict(config["model"])
        trial["training"] = dict(config["training"])
        trial["model"]["hidden_dim"] = int(round(params["hidden_dim"] / 16) * 16)
        trial["model"]["dropout"] = float(params["dropout"])
        trial["training"]["learning_rate"] = float(params["learning_rate"])
        trial["training"]["weight_decay"] = float(params["weight_decay"])
        metrics = train_once(trial, train_path, valid_path)
        return 1.0 - metrics.f1

    optimizer = LWAROOptimizer(
        bounds=search_config["bounds"],
        population_size=int(search_config.get("population_size", 8)),
        iterations=int(search_config.get("iterations", 10)),
        seed=int(config.get("seed", 42)),
    )
    result = optimizer.optimize(objective)

    tuned = dict(config)
    tuned["model"] = dict(config["model"])
    tuned["training"] = dict(config["training"])
    tuned["model"]["hidden_dim"] = int(round(result.best_params["hidden_dim"] / 16) * 16)
    tuned["model"]["dropout"] = float(result.best_params["dropout"])
    tuned["training"]["learning_rate"] = float(result.best_params["learning_rate"])
    tuned["training"]["weight_decay"] = float(result.best_params["weight_decay"])
    metrics = train_once(tuned, train_path, valid_path)
    return result.best_params, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SALSTM-LWARO for multimodal emotion recognition.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    parser.add_argument("--train", required=True, help="Training split .npz file.")
    parser.add_argument("--valid", required=True, help="Validation split .npz file.")
    parser.add_argument("--search", action="store_true", help="Run LWARO hyperparameter search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.search:
        best_params, metrics = run_search(config, args.train, args.valid)
        print({"best_params": best_params, "metrics": asdict(metrics)})
    else:
        metrics = train_once(config, args.train, args.valid)
        print(asdict(metrics))


if __name__ == "__main__":
    main()

