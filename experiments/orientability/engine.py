"""Shared, memory-bounded BREC evaluator for saved orientability base pairs."""
from __future__ import annotations

import csv
import ctypes
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn import CosineEmbeddingLoss
from torch_geometric.loader import DataLoader

ORIENTABILITY_DIR = Path(__file__).resolve().parent
NEW_EXPERIMENTS_DIR = ORIENTABILITY_DIR.parent
PROJECT_ROOT = NEW_EXPERIMENTS_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for _path in (WORKSPACE_ROOT, PROJECT_ROOT, NEW_EXPERIMENTS_DIR, ORIENTABILITY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baselines.esan.model import build_esan_model  # noqa: E402
from baselines.gin.model import GINClassifier  # noqa: E402
from baselines.hasse.model import HasseGINClassifier  # noqa: E402
from baselines.boundary_mlp.model import (  # noqa: E402
    BoundaryLoopMLP,
    boundary_loop_features,
)
from baselines.count_mlp.model import CountMLP  # noqa: E402
from baselines.dist_mlp.model import (  # noqa: E402
    DistanceHistogramMLP,
    distance_histogram_features,
)
from orient_common import build_orientability_certificate  # noqa: E402

FOLLOW_BATCH = ["x_0", "x_1", "x_2"]


def sanitize_for_public_artifact(value):
    """Remove machine/user-specific absolute paths from saved artifacts."""
    replacements = (
        (str(PROJECT_ROOT), "<REPO_ROOT>"),
        (str(NEW_EXPERIMENTS_DIR), "<NEW_EXPERIMENTS_DIR>"),
        (str(ORIENTABILITY_DIR), "<ORIENTABILITY_DIR>"),
    )
    if isinstance(value, dict):
        return {key: sanitize_for_public_artifact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_public_artifact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_public_artifact(item) for item in value)
    if isinstance(value, str):
        sanitized = value
        for old, new in replacements:
            sanitized = sanitized.replace(old, new)
        return sanitized
    return value


class BatchedCountMLP(nn.Module):
    """The canonical CountMLP fed graph-wise counts from a ComplexData batch."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.model = CountMLP(hidden_dim=hidden_dim)

    @staticmethod
    def _counts(batch: torch.Tensor, size: int) -> torch.Tensor:
        return torch.bincount(batch, minlength=size).float()

    def forward(self, data) -> torch.Tensor:
        size = int(data.num_graphs)
        n0 = self._counts(data.x_0_batch, size)
        n1 = self._counts(data.x_1_batch, size)
        n2 = (
            self._counts(data.x_2_batch, size)
            if getattr(data, "x_2", None) is not None
            else torch.zeros_like(n0)
        )
        return self.model(torch.stack((n0 - n1 + n2, n0, n1, n2), dim=1))


class BatchedFeatureMLP(nn.Module):
    """Apply a scalar-feature baseline to each graph in a PyG ComplexData batch."""

    def __init__(self, model: nn.Module, feature_fn) -> None:
        super().__init__()
        self.model = model
        self.feature_fn = feature_fn

    def forward(self, data) -> torch.Tensor:
        device = data.x_0.device
        features = torch.stack(
            [self.feature_fn(item).to(device) for item in data.to_data_list()],
            dim=0,
        )
        return self.model(features)


def resolve_device(value: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def resolve_path(script_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (script_dir / path).resolve()


def load_complex(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def list_pairs(pair_dir: Path) -> list[tuple[int, int, Path, Path]]:
    pairs = []
    pattern = re.compile(r"h(\d+)_p(\d+)_first$")
    for first in pair_dir.glob("h*_p*_first.pt"):
        match = pattern.fullmatch(first.stem)
        if not match:
            continue
        h, p = map(int, match.groups())
        second = pair_dir / f"h{h}_p{p}_second.pt"
        if not second.is_file():
            raise FileNotFoundError(f"Missing second member of pair: {second}")
        pairs.append((h, p, first, second))
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class CopySet:
    """CPU permutation copies, collated and transferred one minibatch at a time."""

    def __init__(self, base, copies: int, permutation_seed: int, cfg=None) -> None:
        # ComplexData permutation uses torch's RNG. Model initialization has a
        # separate seed and cannot alter the paired data seen by the four runs.
        state = torch.random.get_rng_state()
        torch.manual_seed(permutation_seed)
        try:
            self.items = base.get_permuted_copies(num_copies=copies)
        finally:
            torch.random.set_rng_state(state)
        if cfg is not None and str(cfg.model_names[0]) == "smcn":
            model_cfg = cfg.models.smcn
            for item in self.items:
                item.compute_subcomplex_feature(
                    low_rk=int(model_cfg.low_rank),
                    high_rk=int(model_cfg.high_rank),
                    binary_marking=bool(model_cfg.binary_marking),
                )

    def batches(self, batch_size: int):
        loader = DataLoader(
            self.items,
            batch_size=batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            num_workers=0,
        )
        yield from loader

    def release(self) -> None:
        self.items.clear()


def build_model(cfg: DictConfig, name: str) -> nn.Module:
    model_cfg = cfg.models[name]
    if name == "mlp":
        model = BatchedCountMLP(int(model_cfg.hidden_dim))
    elif name == "dist_mlp":
        model = BatchedFeatureMLP(
            DistanceHistogramMLP(
                max_dist=int(model_cfg.max_dist),
                hidden_dim=int(model_cfg.hidden_dim),
            ),
            lambda item: distance_histogram_features(
                item,
                high_rank=int(model_cfg.high_rank),
                max_dist=int(model_cfg.max_dist),
            ),
        )
    elif name == "boundary_mlp":
        model = BatchedFeatureMLP(
            BoundaryLoopMLP(hidden_dim=int(model_cfg.hidden_dim)),
            boundary_loop_features,
        )
    elif name == "gin":
        model = GINClassifier(
            hidden_dim=int(model_cfg.hidden_dim),
            num_layers=int(model_cfg.num_layers),
            num_classes=2,
            dropout=float(model_cfg.dropout),
        )
    elif name == "hasse_gin":
        model = HasseGINClassifier(
            hidden_dim=int(model_cfg.hidden_dim),
            num_layers=int(model_cfg.num_layers),
            num_classes=2,
            dropout=float(model_cfg.dropout),
        )
    elif name == "esan":
        model = build_esan_model(
            hidden_dim=int(model_cfg.hidden_dim),
            num_layers=int(model_cfg.num_layers),
            num_classes=2,
            dropout=float(model_cfg.dropout),
            num_hops=int(model_cfg.num_hops),
            chunk_size=int(model_cfg.chunk_size),
            device=str(cfg.device),
        )
    elif name == "homp_cin":
        from baselines.cin.model import build_homp_model

        model = build_homp_model(
            number_of_layers=int(model_cfg.num_layers),
            embedding_dim=int(model_cfg.hidden_dim),
            device=str(cfg.device),
            num_classes=2,
            output_ranks=list(model_cfg.output_ranks),
            number_of_mlp_layers=int(model_cfg.number_of_mlp_layers),
            dropout=float(model_cfg.dropout),
            final_dropout=float(model_cfg.final_dropout),
            in_dropout=float(model_cfg.in_dropout),
            activation=str(model_cfg.activation),
            include_rank2_updates=bool(model_cfg.include_rank2_updates),
        )
    elif name == "smcn":
        from baselines.smcn.model import build_subcomplex_model

        model = build_subcomplex_model(
            embedding_dim=int(model_cfg.embedding_dim),
            number_cin_layers=int(model_cfg.number_cin_layers),
            number_subgraph_layers=int(model_cfg.number_subgraph_layers),
            device=str(cfg.device),
            output_ranks=list(model_cfg.output_ranks),
            number_of_mlp_layers=int(model_cfg.number_of_mlp_layers),
            dropout=float(model_cfg.dropout),
            in_dropout=float(model_cfg.in_dropout),
            residual=bool(model_cfg.residual),
            activation=str(model_cfg.activation),
            high_rank=int(model_cfg.high_rank),
            binary_marking=bool(model_cfg.binary_marking),
            num_classes=int(model_cfg.num_classes),
        )
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return model.to(cfg.device)


def _paired_batches(first: CopySet, second: CopySet, batch_size: int):
    yield from zip(first.batches(batch_size), second.batches(batch_size))


def effective_batch_size(cfg: DictConfig) -> int:
    name = str(cfg.model_names[0])
    model_cfg = cfg.models.get(name, {})
    return int(model_cfg.get("batch_size", cfg.batch_size))


def should_free_each_batch(cfg: DictConfig) -> bool:
    name = str(cfg.model_names[0])
    model_cfg = cfg.models.get(name, {})
    return bool(model_cfg.get("free_memory_each_batch", False))


def train_trial(cfg, first: CopySet, second: CopySet, model, optimizer, loss_fn) -> float:
    model.train()
    copies = int(cfg.number_of_permuted_copies)
    batch_size = effective_batch_size(cfg)
    free_each_batch = should_free_each_batch(cfg)
    final_loss = 0.0
    for _ in range(int(cfg.num_epochs)):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_a, batch_b in _paired_batches(first, second, batch_size):
            batch_a, batch_b = batch_a.to(cfg.device), batch_b.to(cfg.device)
            vec_a, vec_b = model(batch_a), model(batch_b)
            label = torch.full((len(vec_a),), -1.0, device=cfg.device)
            loss = loss_fn(vec_a, vec_b, label)
            weight = len(vec_a) / copies
            (loss * weight).backward()
            total_loss += float(loss.item()) * len(vec_a)
            del batch_a, batch_b, vec_a, vec_b, label, loss
            if free_each_batch:
                free_memory()
        if float(cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        # Exactly one update per epoch, irrespective of the minibatch size.
        optimizer.step()
        final_loss = total_loss / copies
    return final_loss


@torch.no_grad()
def embeddings(cfg, copies: CopySet, model) -> torch.Tensor:
    model.eval()
    result = []
    free_each_batch = should_free_each_batch(cfg)
    for batch in copies.batches(effective_batch_size(cfg)):
        batch = batch.to(cfg.device)
        result.append(model(batch).cpu())
        del batch
        if free_each_batch:
            free_memory()
    return torch.cat(result).to(cfg.device)


@torch.no_grad()
def t2_score(cfg, first: CopySet, second: CopySet, model) -> float:
    vec_a, vec_b = embeddings(cfg, first, model), embeddings(cfg, second, model)
    diff = vec_a - vec_b
    mean = diff.mean(0).reshape(-1, 1)
    covariance = torch.cov(diff.T)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    regularizer = torch.eye(diff.shape[1], device=cfg.device) * float(cfg.epsilon_matrix)
    score = mean.T @ torch.linalg.pinv(covariance + regularizer) @ mean
    return float(score.item())


def evaluate_trial(cfg, first, second, model, optimizer, loss_fn) -> dict:
    loss = train_trial(cfg, first, second, model, optimizer, loss_fn)
    score = t2_score(cfg, first, second, model)
    return {
        "loss": loss,
        "t2_score": score,
        "above_threshold": score > float(cfg.threshold),
        "close_to_threshold": abs(score - float(cfg.threshold)) <= float(cfg.epsilon_cmp),
    }


def _copy_pair(base_a, base_b, cfg, pair_index: int, phase: int):
    seed = int(cfg.permutation_seed) + pair_index * 10 + phase * 2
    return CopySet(base_a, int(cfg.number_of_permuted_copies), seed, cfg), CopySet(
        base_b, int(cfg.number_of_permuted_copies), seed + 1, cfg
    )


def run(cfg: DictConfig, script_dir: Path, family_label: str) -> None:
    start_time = time.time()
    if int(cfg.number_of_permuted_copies) < 2:
        raise ValueError("number_of_permuted_copies must be at least 2")
    if int(cfg.batch_size) < 1 or int(cfg.num_epochs) < 1:
        raise ValueError("batch_size and num_epochs must be positive")
    if int(cfg.batch_size) > int(cfg.number_of_permuted_copies):
        raise ValueError("batch_size cannot exceed number_of_permuted_copies")

    cfg.device = resolve_device(str(cfg.device))
    pair_dir = resolve_path(script_dir, str(cfg.base_pair_dir))
    output_dir = resolve_path(script_dir, str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    certificates_path = output_dir / "certificates.jsonl"
    certificates_path.write_text("", encoding="utf-8")
    pairs = list_pairs(pair_dir)
    if not pairs:
        raise RuntimeError(f"No h*_p*_first.pt pairs found in {pair_dir}")
    if cfg.get("max_pairs") is not None:
        pairs = pairs[: int(cfg.max_pairs)]

    name = str(cfg.model_names[0])
    if len(cfg.model_names) != 1:
        raise ValueError("The launcher/evaluator runs exactly one model per process")
    torch.manual_seed(int(cfg.model_seed))
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(int(cfg.model_seed))
    model = build_model(cfg, name)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    loss_fn = CosineEmbeddingLoss(margin=float(cfg.margin))

    rows, target_ids, reliability_ids, solved_ids = [], [], [], []
    target_scores, reliability_scores = [], []
    print(f"{family_label}: {len(pairs)} pairs, model={name}, device={cfg.device}")

    for pair_index, (h, p, first_path, second_path) in enumerate(pairs):
        certificate = build_orientability_certificate(
            family=str(cfg.family),
            family_label=str(cfg.get("family_label", family_label)),
            dataset=OmegaConf.to_container(cfg.get("dataset", {}), resolve=True),
            pair_index=pair_index,
            sequence_index=pair_index,
            h=h,
            p=p,
        )
        with certificates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(certificate, sort_keys=True) + "\n")
        first_base, second_base = load_complex(first_path), load_complex(second_path)
        if bool(cfg.fresh_model_per_pair) and pair_index:
            del model, optimizer
            free_memory()
            torch.manual_seed(int(cfg.model_seed) + pair_index)
            model = build_model(cfg, name)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(cfg.learning_rate),
                weight_decay=float(cfg.weight_decay),
            )

        target_a, target_b = _copy_pair(first_base, second_base, cfg, pair_index, 0)
        target = evaluate_trial(cfg, target_a, target_b, model, optimizer, loss_fn)
        target_pass = target["above_threshold"] and not target["close_to_threshold"]
        target_scores.append(target["t2_score"])
        if target_pass:
            target_ids.append(pair_index)
        rows.append(dict(pair_index=pair_index, h=h, p=p, trial_type="target",
                         passed=target_pass, **target))
        target_a.release()
        target_b.release()
        del target_a, target_b
        free_memory()

        # Reliability uses two independently permuted realizations of class A.
        rel_a, rel_b = _copy_pair(first_base, first_base, cfg, pair_index, 1)
        reliability = evaluate_trial(cfg, rel_a, rel_b, model, optimizer, loss_fn)
        reliability_pass = (
            not reliability["above_threshold"] and not reliability["close_to_threshold"]
        )
        reliability_scores.append(reliability["t2_score"])
        if reliability_pass:
            reliability_ids.append(pair_index)
        if target_pass and reliability_pass:
            solved_ids.append(pair_index)
        rows.append(dict(pair_index=pair_index, h=h, p=p, trial_type="reliability",
                         passed=reliability_pass, **reliability))
        rel_a.release()
        rel_b.release()
        del rel_a, rel_b, first_base, second_base
        free_memory()
        print(
            f"pair={pair_index} h={h} p={p} target_t2={target['t2_score']:.4f} "
            f"reliability_t2={reliability['t2_score']:.4f} solved={target_pass and reliability_pass}",
            flush=True,
        )

    resolved = OmegaConf.to_container(cfg, resolve=True)
    # Keep per-run config.json in the old, compact style even though the
    # launcher now uses one shared root config.  The scientific run metadata
    # remains the selected family/model/seed/hyperparameters, not the entire
    # multi-family config book.
    for key in (
        "families",
        "family",
        "family_dir",
        "family_label",
        "dataset",
        "dataset_policy",
        "label",
    ):
        resolved.pop(key, None)
    resolved["output_dir"] = str(output_dir)
    resolved["baseline_name"] = name
    resolved["baseline_model"] = OmegaConf.to_container(cfg.models[name], resolve=True)
    resolved["effective_batch_size"] = effective_batch_size(cfg)
    summary = {
        "architecture": name,
        "parameter_count": parameter_count,
        "runtime_seconds": time.time() - start_time,
        "target_success_count": len(target_ids),
        "target_total": len(pairs),
        "reliability_pass_count": len(reliability_ids),
        "reliability_total": len(pairs),
        "brec_solved_count": len(solved_ids),
        "brec_solved_total": len(pairs),
        "target_success_ids": target_ids,
        "reliability_pass_ids": reliability_ids,
        "brec_solved_ids": solved_ids,
        "mean_t2_target": float(np.mean(target_scores)),
        "mean_t2_reliability": float(np.mean(reliability_scores)),
    }
    resolved = sanitize_for_public_artifact(resolved)
    summary = sanitize_for_public_artifact(summary)
    (output_dir / "config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"FINAL {name}: {len(solved_ids)}/{len(pairs)} BREC pairs solved", flush=True)
