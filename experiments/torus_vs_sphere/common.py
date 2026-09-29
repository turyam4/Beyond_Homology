"""Shared training and evaluation utilities for torus-vs-sphere baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Callable, Iterable

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
NEW_EXPERIMENTS_DIR = PACKAGE_ROOT.parent
SMCN_ROOT = NEW_EXPERIMENTS_DIR.parent
DATASET_CACHE_ROOT = PACKAGE_ROOT / "dataset_cache"
for path in (SMCN_ROOT.parent, SMCN_ROOT, NEW_EXPERIMENTS_DIR, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torus_vs_sphere_dataset import TorusVsSphereDataset  # noqa: E402
from baselines.count_mlp.model import CountMLP, count_features  # noqa: E402
from baselines.dist_mlp.model import (  # noqa: E402
    DistanceHistogramMLP,
    distance_histogram_features,
)
from baselines.boundary_mlp.model import (  # noqa: E402
    BoundaryLoopMLP,
    boundary_loop_features,
)


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for the torus-vs-sphere control experiment."""

    seed: int = 0
    torus_grid_sizes: tuple[tuple[int, int], ...] = ((4, 4), (5, 5), (6, 6), (7, 7))
    sphere_subdiv_levels: tuple[int, ...] = (0, 1, 2, 2)
    copies: int = 32
    epochs: int = 40
    mlp_epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-2
    mlp_learning_rate: float = 1e-2
    hidden_dim: int = 64
    num_layers: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def dataset_cache_root(config: RunnerConfig) -> Path:
    return DATASET_CACHE_ROOT / (
        f"seed{config.seed}_t{len(config.torus_grid_sizes)}_"
        f"s{len(config.sphere_subdiv_levels)}_copies{config.copies}"
    )


def validate_dataset(data: Iterable) -> None:
    """Validate labels and the rank tensors every baseline relies on."""

    counts = {0: 0, 1: 0}
    total = 0
    for item in data:
        total += 1
        if not hasattr(item, "y"):
            raise ValueError("dataset item is missing y")
        label = int(item.y.reshape(-1)[0].item())
        if label not in counts:
            raise ValueError(f"unexpected label {label}; expected 0 or 1")
        counts[label] += 1
        for rank in (0, 1, 2):
            if not hasattr(item, f"x_{rank}"):
                raise ValueError(f"dataset item is missing x_{rank}")
            if not hasattr(item, f"num_cells_{rank}"):
                raise ValueError(f"dataset item is missing num_cells_{rank}")
    if total == 0:
        raise ValueError("dataset is empty")
    if counts[0] != counts[1]:
        raise ValueError(f"expected balanced classes; got {counts}")


def validate_required_fields(data: Iterable, required_fields: tuple[str, ...]) -> None:
    for idx, item in enumerate(data):
        missing = [name for name in required_fields if not hasattr(item, name)]
        if missing:
            raise ValueError(f"item {idx} is missing fields required by model: {missing}")


def load_dataset(config: RunnerConfig) -> list:
    """Load or generate the cached dataset."""

    cache_root = dataset_cache_root(config)
    cache_root.mkdir(parents=True, exist_ok=True)
    processed_files = list((cache_root / "processed").glob("*.pt"))
    if processed_files:
        print(f"Reusing cached dataset from: {cache_root}")
    else:
        print(f"No cached dataset found; generating and caching in: {cache_root}")

    dataset = TorusVsSphereDataset(
        root=str(cache_root),
        torus_grid_sizes=list(config.torus_grid_sizes),
        sphere_subdiv_levels=list(config.sphere_subdiv_levels),
        number_of_permuted_copies=config.copies,
        seed=config.seed,
    )
    data = [
        item
        for item in tqdm(
            dataset,
            desc="Loading torus-vs-sphere data",
            unit="sample",
        )
    ]
    validate_dataset(data)
    return data


def stratified_split(data: list, seed: int) -> dict[str, list]:
    """Create a deterministic 60/20/20 split for each class."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor(
        [int(item.y.reshape(-1)[0].item()) for item in data], dtype=torch.long
    )
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for label in sorted(int(v) for v in labels.unique().tolist()):
        idx = torch.where(labels == label)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n_train = int(0.6 * idx.numel())
        n_val = int(0.2 * idx.numel())
        splits["train"].extend(data[int(i)] for i in idx[:n_train])
        splits["val"].extend(data[int(i)] for i in idx[n_train : n_train + n_val])
        splits["test"].extend(data[int(i)] for i in idx[n_train + n_val :])
    for name, items in splits.items():
        if not items:
            raise ValueError(f"empty {name} split")
    return splits


def prepare_splits(config: RunnerConfig) -> dict[str, list]:
    """Load, validate, and split the dataset."""

    set_seed(config.seed)
    return stratified_split(load_dataset(config), config.seed)


def labels_from_batch(batch, device: torch.device) -> torch.Tensor:
    if not hasattr(batch, "y") or batch.y is None:
        raise ValueError("batch is missing labels")
    return batch.y.reshape(-1).to(device=device, dtype=torch.long)


def _check_logits(logits: torch.Tensor, n: int, name: str) -> None:
    if logits.ndim != 2 or logits.size(0) != n or logits.size(1) != 2:
        raise ValueError(f"{name}: expected logits [{n}, 2], got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{name}: logits contain NaN or Inf")


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = labels_from_batch(batch, device)
            logits = model(batch)
            _check_logits(logits, labels.numel(), "evaluate")
            total_loss += float(loss_fn(logits, labels).item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    if total == 0:
        raise ValueError("cannot evaluate an empty split")
    return {"loss": total_loss / total, "accuracy": correct / total,
            "correct": correct, "total": total}


def _print_final(model_name: str, splits: dict[str, list], metrics: dict) -> None:
    sizes = {name: len(items) for name, items in splits.items()}
    print("\nFINAL RESULTS")
    print(f"model: {model_name}")
    print(f"split_sizes: {sizes}")
    for split in ("train", "val", "test"):
        row = metrics[split]
        print(f"{split:5s} accuracy: {row['accuracy']:.4f} ({row['correct']}/{row['total']})")


def train_graph_model(
    *,
    model_name: str,
    model_factory: Callable[[RunnerConfig, torch.device], nn.Module],
    follow_batch: list[str],
    required_fields: tuple[str, ...],
    splits: dict[str, list],
    config: RunnerConfig,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for items in splits.values():
        validate_required_fields(items, required_fields)

    def make_loader(items: list, shuffle: bool) -> DataLoader:
        return DataLoader(items, batch_size=config.batch_size, shuffle=shuffle,
                          follow_batch=follow_batch)

    loaders = {
        "train": make_loader(splits["train"], True),
        "eval_train": make_loader(splits["train"], False),
        "val": make_loader(splits["val"], False),
        "test": make_loader(splits["test"], False),
    }

    set_seed(config.seed)
    model = model_factory(config, device).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    epoch_bar = tqdm(
        range(1, config.epochs + 1),
        desc=f"{model_name} training",
        unit="epoch",
    )
    for _epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        epoch_total = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            labels = labels_from_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            _check_logits(logits, labels.numel(), model_name)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError(f"{model_name}: loss is NaN or Inf")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(labels.numel())
            epoch_total += int(labels.numel())
        val_metrics = evaluate(model, loaders["val"], loss_fn, device)
        epoch_bar.set_postfix(
            loss=f"{epoch_loss / epoch_total:.4f}",
            val=f"{val_metrics['accuracy']:.4f}",
        )
        if float(val_metrics["accuracy"]) >= best_val:
            best_val = float(val_metrics["accuracy"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    metrics = {
        "model": model_name,
        "train": evaluate(model, loaders["eval_train"], loss_fn, device),
        "val": evaluate(model, loaders["val"], loss_fn, device),
        "test": evaluate(model, loaders["test"], loss_fn, device),
        "best_val_accuracy": best_val,
        "parameter_count": parameter_count,
        "epochs": config.epochs,
        "device": str(device),
    }
    _print_final(model_name, splits, metrics)
    return metrics


PROBES: dict[str, tuple[Callable, Callable[[], nn.Module]]] = {
    "count_mlp": (count_features, CountMLP),
    "dist_mlp": (distance_histogram_features, DistanceHistogramMLP),
    "boundary_mlp": (boundary_loop_features, BoundaryLoopMLP),
}


def train_probe_mlp(
    *,
    model_name: str,
    splits: dict[str, list],
    config: RunnerConfig,
) -> dict:
    feature_fn, model_cls = PROBES[model_name]

    features, labels = {}, {}
    for name in ("train", "val", "test"):
        items = splits[name]
        # Compress feature scales before standardization while preserving signs.
        raw = torch.stack(
            [
                feature_fn(item)
                for item in tqdm(
                    items,
                    desc=f"{model_name} features [{name}]",
                    unit="sample",
                    leave=False,
                )
            ]
        )
        features[name] = torch.sign(raw) * torch.log1p(raw.abs())
        labels[name] = torch.tensor(
            [int(item.y.reshape(-1)[0].item()) for item in items], dtype=torch.long
        )
    mean = features["train"].mean(dim=0)
    std = features["train"].std(dim=0).clamp_min(1e-3)
    for name in features:
        features[name] = (features[name] - mean) / std

    set_seed(config.seed)
    model = model_cls()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.mlp_learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    def split_metrics(name: str) -> dict:
        model.eval()
        with torch.no_grad():
            logits = model(features[name])
            _check_logits(logits, labels[name].numel(), model_name)
            loss = loss_fn(logits, labels[name])
            correct = int((logits.argmax(dim=1) == labels[name]).sum().item())
            total = int(labels[name].numel())
        return {"loss": float(loss.item()), "accuracy": correct / total,
                "correct": correct, "total": total}

    best_val, best_state = -1.0, None
    epoch_bar = tqdm(
        range(1, config.mlp_epochs + 1),
        desc=f"{model_name} training",
        unit="epoch",
    )
    for _epoch in epoch_bar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features["train"])
        loss = loss_fn(logits, labels["train"])
        if not torch.isfinite(loss):
            raise ValueError(f"{model_name}: loss is NaN or Inf")
        loss.backward()
        optimizer.step()
        val_acc = split_metrics("val")["accuracy"]
        epoch_bar.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            val=f"{val_acc:.4f}",
        )
        if val_acc >= best_val:
            best_val = float(val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "model": model_name,
        "train": split_metrics("train"),
        "val": split_metrics("val"),
        "test": split_metrics("test"),
        "best_val_accuracy": best_val,
        "parameter_count": parameter_count,
        "epochs": config.mlp_epochs,
        "device": "cpu",
        "learning_rate": config.mlp_learning_rate,
    }
    _print_final(model_name, splits, metrics)
    return metrics
