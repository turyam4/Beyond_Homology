"""Cache, loading, saving, and synchronization helpers for FG genus-2 runs."""
from __future__ import annotations

import json
import os
import time
import csv
import ctypes
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
FOLLOW_BATCH_FIELDS = ["x_0", "x_1", "x_2"]
BASELINE_KEEP_FIELDS = {
    "num_nodes",
    "dim",
    "x_0",
    "x_1",
    "x_2",
    "num_cells_0",
    "num_cells_1",
    "num_cells_2",
    "edge_index_incidence_0_1",
    "edge_index_incidence_0_2",
    "edge_index_incidence_1_2",
    "edge_index_adjacency_0_1",
    "bridge_index_adjacency_0_1",
    "edge_index_adjacency_1_2",
    "bridge_index_adjacency_1_2",
    "edge_index_coadjacency_1_0",
}

base = None


def ensure_base():
    """Return the local shared runtime API used by engine.py and cache helpers."""
    global base
    if base is None:
        base = SimpleNamespace(
            SCRIPT_DIR=SCRIPT_DIR,
            FOLLOW_BATCH_FIELDS=FOLLOW_BATCH_FIELDS,
            validate_cfg=validate_cfg,
            resolve_device=resolve_device,
            resolve_output_dir=resolve_output_dir,
            log_message=log_message,
            build_model=build_model,
            make_pair_batches=make_pair_batches,
            try_evaluate_pair=try_evaluate_pair,
            free_memory=free_memory,
            write_outputs=write_outputs,
        )
    return base


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def validate_cfg(cfg: DictConfig) -> None:
    if int(cfg.min_g) > int(cfg.max_g):
        raise ValueError("min_g must be <= max_g")
    if int(cfg.samples_per_g) < 1:
        raise ValueError("samples_per_g must be >= 1")
    if int(cfg.number_of_permuted_copies) < 2:
        raise ValueError("number_of_permuted_copies must be >= 2")
    if int(cfg.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(cfg.get("eval_batch_size", cfg.batch_size)) < 1:
        raise ValueError("eval_batch_size must be >= 1")
    if int(cfg.low_rk) < 0 or int(cfg.high_rk) < int(cfg.low_rk):
        raise ValueError("invalid low_rk/high_rk")


def log_message(output_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def build_model(cfg: DictConfig):
    from baselines.smcn.model import build_model as build_fg_model

    return build_fg_model(cfg)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def collate_one_batch(data_list: List):
    loader = DataLoader(
        data_list,
        batch_size=len(data_list),
        shuffle=False,
        follow_batch=FOLLOW_BATCH_FIELDS,
    )
    return next(iter(loader))


def collate_minibatches(data_list: List, batch_size: int) -> List:
    loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH_FIELDS,
    )
    return list(loader)


def prune_to_baseline_fields(data_list: List) -> None:
    for data in data_list:
        for key in list(data.keys()):
            if key not in BASELINE_KEEP_FIELDS:
                del data[key]


def make_pair_batches(cfg: DictConfig, g: int, rng: np.random.Generator):
    from fundamental_group_dataset_genus2 import build_tight_pair_copies_compatible

    (
        surface_target,
        free_target,
        surface_rel_a,
        surface_rel_b,
        metadata,
        certificate,
    ) = build_tight_pair_copies_compatible(
        g=g,
        rng=rng,
        noise_pairs_per_g=cfg.noise_pairs_per_g,
        min_subdiv=cfg.min_subdiv,
        max_subdiv=cfg.max_subdiv,
        match_backtracks=cfg.match_backtracks,
        max_tries=cfg.max_tries,
        max_expansions=cfg.max_expansions,
        number_of_permuted_copies=cfg.number_of_permuted_copies,
        low_rk=cfg.low_rk,
        high_rk=cfg.high_rk,
        include_subcomplex_features=bool(
            cfg.get("include_subcomplex_features", True)
        ),
        binary_marking=bool(cfg.get("binary_marking", False)),
    )

    if bool(cfg.get("prune_to_baseline_fields", False)):
        for copies in (surface_target, free_target, surface_rel_a, surface_rel_b):
            prune_to_baseline_fields(copies)

    if bool(cfg.get("cache_as_minibatches", False)):
        batch_size = max(1, int(cfg.batch_size))
        result = (
            collate_minibatches(surface_target, batch_size),
            collate_minibatches(free_target, batch_size),
            collate_minibatches(surface_rel_a, batch_size),
            collate_minibatches(surface_rel_b, batch_size),
            metadata,
            certificate,
        )
    else:
        result = (
            collate_one_batch(surface_target),
            collate_one_batch(free_target),
            collate_one_batch(surface_rel_a),
            collate_one_batch(surface_rel_b),
            metadata,
            certificate,
        )
    del surface_target, free_target, surface_rel_a, surface_rel_b
    free_memory()
    return result


def effective_batch_size(cfg: DictConfig, key: str) -> int:
    return max(1, int(cfg.get(key, cfg.number_of_permuted_copies)))


def move_batch_to_device(cfg: DictConfig, batch):
    return batch.to(cfg.device, non_blocking=str(cfg.device).startswith("cuda"))


def iterate_minibatches(complex_batch, batch_size: int):
    if hasattr(complex_batch, "iter_minibatches"):
        yield from complex_batch.iter_minibatches(batch_size)
        return
    if isinstance(complex_batch, list):
        for cached_minibatch in complex_batch:
            yield cached_minibatch.clone()
        return
    if batch_size >= complex_batch.num_graphs:
        yield complex_batch
        return
    loader = DataLoader(
        complex_batch.to_data_list(),
        batch_size=batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH_FIELDS,
    )
    yield from loader


def train_on_pair(cfg, first_batch, second_batch, model, loss_fn, optimizer, label=""):
    model.train()
    train_batch_size = effective_batch_size(cfg, "batch_size")
    if hasattr(first_batch, "iter_minibatches"):
        num_examples = int(first_batch.num_graphs)
    elif isinstance(first_batch, list):
        num_examples = sum(batch.num_graphs for batch in first_batch)
    else:
        num_examples = first_batch.num_graphs

    single_minibatch = (
        not isinstance(first_batch, list)
        and not hasattr(first_batch, "iter_minibatches")
        and train_batch_size >= num_examples
    )
    if single_minibatch:
        first_batch = move_batch_to_device(cfg, first_batch)
        second_batch = move_batch_to_device(cfg, second_batch)

    last_epoch_loss = 0.0
    total_epochs = int(cfg.num_epochs)
    compact_progress = bool(cfg.get("compact_epoch_progress", True))
    for epoch_idx in range(total_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for first_mini_batch, second_mini_batch in zip(
            iterate_minibatches(first_batch, train_batch_size),
            iterate_minibatches(second_batch, train_batch_size),
        ):
            if not single_minibatch:
                first_mini_batch = move_batch_to_device(cfg, first_mini_batch)
                second_mini_batch = move_batch_to_device(cfg, second_mini_batch)
            vec1 = model(first_mini_batch)
            vec2 = model(second_mini_batch)
            y = torch.full((len(vec1),), -1.0, device=cfg.device)
            loss = loss_fn(vec1, vec2, y)
            weight = len(vec1) / num_examples
            (loss * weight).backward()
            epoch_loss += float(loss.item()) * len(vec1)
            del vec1, vec2, y, loss, first_mini_batch, second_mini_batch
        if cfg.get("grad_clip", 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
        optimizer.step()
        # Gradients are no longer needed after the accumulated optimizer step.
        optimizer.zero_grad(set_to_none=True)
        last_epoch_loss = epoch_loss / num_examples
        if label and (not compact_progress or epoch_idx + 1 == total_epochs):
            message = (
                f"TRAIN {label} epochs={epoch_idx + 1}/{total_epochs} "
                f"final_loss={last_epoch_loss:.4f}"
            )
            print(message, flush=True)
    return last_epoch_loss


@torch.no_grad()
def embed_batch(cfg, complex_batch, model) -> torch.Tensor:
    model.eval()
    embeddings = []
    for mini_batch in iterate_minibatches(
        complex_batch, effective_batch_size(cfg, "eval_batch_size")
    ):
        mini_batch = move_batch_to_device(cfg, mini_batch)
        embeddings.append(model(mini_batch).cpu())
        del mini_batch
    return torch.cat(embeddings, dim=0).to(cfg.device)


@torch.no_grad()
def t2_score(cfg, first_batch, second_batch, model) -> float:
    vec1 = embed_batch(cfg, first_batch, model)
    vec2 = embed_batch(cfg, second_batch, model)
    output_dim = vec1.shape[1]
    s_epsilon = torch.diag(
        torch.full((output_dim,), float(cfg.epsilon_matrix), device=cfg.device)
    )
    diff = vec1 - vec2
    diff_mean = diff.mean(dim=0).reshape(-1, 1)
    covariance = torch.cov(diff.t())
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    stat = (diff_mean.t() @ torch.linalg.pinv(covariance + s_epsilon) @ diff_mean)
    return float(stat.item())


def evaluate_pair(cfg, first_batch, second_batch, model, optimizer, loss_fn, label=""):
    loss = train_on_pair(cfg, first_batch, second_batch, model, loss_fn, optimizer, label)
    score = t2_score(cfg, first_batch, second_batch, model)
    above_threshold = score > float(cfg.threshold)
    close_to_threshold = abs(score - float(cfg.threshold)) <= float(cfg.epsilon_cmp)
    free_memory()
    return {
        "loss": float(loss),
        "t2_score": float(score),
        "above_threshold": bool(above_threshold),
        "close_to_threshold": bool(close_to_threshold),
    }


def try_evaluate_pair(
    cfg,
    output_dir: Path,
    pair_index: int,
    label: str,
    first_batch,
    second_batch,
    model,
    optimizer,
    loss_fn,
):
    try:
        return evaluate_pair(
            cfg, first_batch, second_batch, model, optimizer, loss_fn, label=label
        ), False
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "mps backend out of memory" not in message:
            raise
        free_memory()
        log_message(output_dir, f"OOM pair={pair_index} label={label}: {exc}")
        return None, True


def write_outputs(cfg: DictConfig, output_dir: Path, rows: List[Dict], summary: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            OmegaConf.to_container(cfg, resolve=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def seed_value(cfg: DictConfig, name: str, fallback: int = 0) -> int:
    return int(cfg.get(name, fallback))


def cache_dir(cfg: DictConfig) -> Path:
    base_module = ensure_base()
    configured = cfg.get("dataset_cache_dir")
    if configured:
        value = Path(str(configured))
    else:
        marking = "bin" if bool(cfg.get("binary_marking", False)) else "dist"
        compact = "compact" if bool(cfg.get("compact_subcomplex_cache", False)) else "full"
        phase = "phase" if bool(cfg.get("construct_cache_by_phase", False)) else "joint"
        version = str(cfg.get("cache_version", "v1"))
        if bool(cfg.get("include_subcomplex_features", True)):
            cache_role = "smcn_subcomplex"
        elif bool(cfg.get("prune_to_baseline_fields", False)):
            cache_role = "baseline_complex"
        else:
            cache_role = "complex"
        value = Path("fg_pairs") / (
            f"fg_{cache_role}_"
            f"{compact}_{phase}_"
            f"g{int(cfg.min_g)}-{int(cfg.max_g)}_"
            f"s{int(cfg.samples_per_g)}_"
            f"b{int(cfg.batch_size)}_"
            f"c{int(cfg.number_of_permuted_copies)}_"
            f"{marking}_{version}"
        )
    return value if value.is_absolute() else base_module.SCRIPT_DIR / value


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pair_part(path: Path, pin_memory: bool):
    """Load one target/reliability part and optionally pin its two batches."""
    item = torch_load(path)
    if pin_memory and "first_batch" in item:
        for key in ("first_batch", "second_batch"):
            try:
                if isinstance(item[key], list):
                    item[key] = [batch.pin_memory() for batch in item[key]]
                else:
                    item[key] = item[key].pin_memory()
            except RuntimeError:
                break
    return item


class DiskBatchSequence:
    """Materialize one compact cached minibatch at a time."""

    def __init__(
        self,
        cache_dir: Path,
        file_names: List[str],
        num_graphs: int,
        build_subcomplex: bool = False,
        low_rk: int = 0,
        high_rk: int = 1,
        binary_marking: bool = False,
        view: str | None = None,
    ):
        self.cache_dir = cache_dir
        self.file_names = file_names
        self.num_graphs = num_graphs
        self.build_subcomplex = build_subcomplex
        self.low_rk = low_rk
        self.high_rk = high_rk
        self.binary_marking = binary_marking
        self.view = view

    def _view_path(self, file_name: str) -> Path:
        view_dir = self.cache_dir / "baseline_views" / str(self.view)
        return view_dir / file_name

    def _collate_cached(self, cached):
        if not isinstance(cached, list):
            return cached
        if self.build_subcomplex:
            for data in cached:
                data.compute_subcomplex_feature(
                    low_rk=self.low_rk,
                    high_rk=self.high_rk,
                    binary_marking=self.binary_marking,
                )
        loader = DataLoader(
            cached,
            batch_size=len(cached),
            shuffle=False,
            follow_batch=ensure_base().FOLLOW_BATCH_FIELDS,
        )
        batch = next(iter(loader))
        del loader, cached
        return batch

    @staticmethod
    def _graph_view(batch):
        return Data(
            x_0=torch.zeros((batch.x_0.size(0), 1), dtype=torch.float32),
            edge_index_adjacency_0_1=batch.edge_index_adjacency_0_1.cpu(),
            x_0_batch=batch.x_0_batch.cpu(),
            num_graphs=int(batch.num_graphs),
        )

    @staticmethod
    def _hasse_view(batch):
        n0 = batch.x_0.size(0)
        n1 = batch.x_1.size(0)
        n2 = batch.x_2.size(0) if getattr(batch, "x_2", None) is not None else 0
        inc01 = batch.edge_index_incidence_0_1.cpu()
        src = [inc01[0], inc01[1] + n0]
        dst = [inc01[1] + n0, inc01[0]]
        if n2 > 0:
            inc12 = batch.edge_index_incidence_1_2.cpu()
            src += [inc12[0] + n0, inc12[1] + n0 + n1]
            dst += [inc12[1] + n0 + n1, inc12[0] + n0]
        edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
        rank_ids = torch.cat(
            [
                torch.zeros(n0, dtype=torch.long),
                torch.ones(n1, dtype=torch.long),
                torch.full((n2,), 2, dtype=torch.long),
            ]
        )
        batches = [batch.x_0_batch.cpu(), batch.x_1_batch.cpu()]
        if n2 > 0:
            batches.append(batch.x_2_batch.cpu())
        return Data(
            hasse_edge_index=edge_index,
            hasse_rank_ids=rank_ids,
            hasse_batch=torch.cat(batches),
            num_nodes=int(rank_ids.numel()),
            num_graphs=int(batch.num_graphs),
        )

    def _project_view(self, batch):
        if self.view == "graph":
            return self._graph_view(batch)
        if self.view == "hasse":
            return self._hasse_view(batch)
        return batch

    def _load_or_build_view(self, file_name: str):
        view_path = self._view_path(file_name)
        if view_path.is_file():
            return torch_load(view_path)
        cached = torch_load(self.cache_dir / file_name)
        batch = self._collate_cached(cached)
        projected = self._project_view(batch)
        view_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = view_path.parent / f".{view_path.name}.{os.getpid()}.tmp"
        torch.save(projected, temporary_path)
        os.replace(temporary_path, view_path)
        del batch
        free_memory()
        return projected

    def iter_minibatches(self, batch_size: int | None = None):
        base_module = ensure_base()
        for file_name in self.file_names:
            if self.view:
                batch = self._load_or_build_view(file_name)
            else:
                cached = torch_load(self.cache_dir / file_name)
                batch = self._collate_cached(cached)
            yield batch
            del batch
            base_module.free_memory()


def cache_spec(cfg: DictConfig) -> Dict:
    return {
        "format_version": (
            6 if bool(cfg.get("construct_cache_by_phase", False)) else 4
        ),
        "cache_version": str(cfg.get("cache_version", "v1")),
        "data_seed": seed_value(cfg, "data_seed"),
        "permutation_seed": seed_value(cfg, "permutation_seed"),
        "min_g": int(cfg.min_g),
        "max_g": int(cfg.max_g),
        "samples_per_g": int(cfg.samples_per_g),
        "number_of_permuted_copies": int(cfg.number_of_permuted_copies),
        "noise_pairs_per_g": int(cfg.noise_pairs_per_g),
        "min_subdiv": int(cfg.min_subdiv),
        "max_subdiv": int(cfg.max_subdiv),
        "match_backtracks": bool(cfg.match_backtracks),
        "max_tries": int(cfg.max_tries),
        "max_expansions": int(cfg.max_expansions),
        "low_rk": int(cfg.low_rk),
        "high_rk": int(cfg.high_rk),
        "include_subcomplex_features": bool(
            cfg.get("include_subcomplex_features", True)
        ),
        "binary_marking": bool(cfg.get("binary_marking", False)),
        "prune_to_baseline_fields": bool(
            cfg.get("prune_to_baseline_fields", False)
        ),
        "cache_as_minibatches": bool(cfg.get("cache_as_minibatches", False)),
        "construct_cache_by_phase": bool(
            cfg.get("construct_cache_by_phase", False)
        ),
        "compact_subcomplex_cache": bool(
            cfg.get("compact_subcomplex_cache", False)
        ),
        "cache_batch_size": (
            int(cfg.batch_size)
            if bool(cfg.get("cache_as_minibatches", False))
            else None
        ),
    }


def build_cache(cfg: DictConfig) -> Path:
    """Generate and atomically publish one immutable fixed benchmark."""
    base_module = ensure_base()
    out_dir = cache_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    spec = cache_spec(cfg)

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != spec:
            raise RuntimeError(
                f"Cache {out_dir} has a different specification. Use a new "
                "dataset_cache_dir; do not mix benchmark definitions."
            )
        expected = int(cfg.samples_per_g) * (int(cfg.max_g) - int(cfg.min_g) + 1)
        missing = [
            i
            for i in range(expected)
            if not (out_dir / f"pair_{i:04d}_target.pt").is_file()
            or not (out_dir / f"pair_{i:04d}_reliability.pt").is_file()
        ]
        if not missing:
            print(f"Fixed dataset cache already complete: {out_dir.name}", flush=True)
            return out_dir
        raise RuntimeError(f"Cache manifest exists but pair files are missing: {missing}")

    data_rng = np.random.default_rng(seed_value(cfg, "data_seed"))
    torch.manual_seed(seed_value(cfg, "permutation_seed"))

    pair_index = 0
    for g in range(int(cfg.min_g), int(cfg.max_g) + 1):
        for sample_index in range(int(cfg.samples_per_g)):
            print(
                f"CACHE BUILD pair={pair_index} g={g} sample={sample_index}",
                flush=True,
            )
            if bool(cfg.get("construct_cache_by_phase", False)):
                build_and_save_pair_phases(
                    cfg, out_dir, pair_index, g, sample_index, data_rng
                )
            else:
                batches = base_module.make_pair_batches(cfg, g, data_rng)
                target_payload = {
                    "pair_index": pair_index,
                    "g": g,
                    "sample_index": sample_index,
                    "first_batch": batches[0],
                    "second_batch": batches[1],
                    "metadata": batches[4],
                    "certificate": batches[5],
                }
                reliability_payload = {
                    "pair_index": pair_index,
                    "g": g,
                    "sample_index": sample_index,
                    "first_batch": batches[2],
                    "second_batch": batches[3],
                }
                save_part(out_dir, pair_index, "target", target_payload)
                save_part(out_dir, pair_index, "reliability", reliability_payload)
                del batches, target_payload, reliability_payload
            base_module.free_memory()
            pair_index += 1

    temporary_manifest = out_dir / f".manifest.{os.getpid()}.tmp"
    temporary_manifest.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"Fixed dataset cache complete: {out_dir.name}", flush=True)
    return out_dir


def save_part(cache_dir: Path, pair_index: int, part_name: str, payload: Dict) -> None:
    final_path = cache_dir / f"pair_{pair_index:04d}_{part_name}.pt"
    temporary_path = (
        cache_dir / f".pair_{pair_index:04d}_{part_name}.{os.getpid()}.tmp"
    )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, final_path)


def save_word_batch_shards(
    cfg: DictConfig,
    cache_dir: Path,
    pair_index: int,
    phase: str,
    class_name: str,
    word_path,
) -> tuple[List[str], int]:
    """Build one class, save each minibatch separately, then release it."""
    base_module = ensure_base()
    from fundamental_group_dataset import build_fundamental_group_copies_from_word

    compact = bool(cfg.get("compact_subcomplex_cache", False))
    copies = build_fundamental_group_copies_from_word(
        word_path,
        int(cfg.number_of_permuted_copies),
        int(cfg.low_rk),
        int(cfg.high_rk),
        include_subcomplex_features=(
            bool(cfg.get("include_subcomplex_features", True)) and not compact
        ),
        binary_marking=bool(cfg.get("binary_marking", False)),
    )
    if bool(cfg.get("prune_to_baseline_fields", False)):
        prune_to_baseline_fields(copies)
    file_names: List[str] = []
    total_graphs = 0
    batch_size = int(cfg.batch_size)
    for batch_index, start in enumerate(range(0, len(copies), batch_size)):
        data_list = copies[start : start + batch_size]
        if compact:
            payload = data_list
            num_graphs = len(data_list)
        else:
            loader = DataLoader(
                data_list,
                batch_size=len(data_list),
                shuffle=False,
                follow_batch=base_module.FOLLOW_BATCH_FIELDS,
            )
            payload = next(iter(loader))
            num_graphs = int(payload.num_graphs)
            del loader
        file_name = (
            f"pair_{pair_index:04d}_{phase}_{class_name}_"
            f"batch_{batch_index:03d}.pt"
        )
        final_path = cache_dir / file_name
        temporary_path = cache_dir / f".{file_name}.{os.getpid()}.tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, final_path)
        file_names.append(file_name)
        total_graphs += num_graphs
        del payload, data_list
    del copies
    base_module.free_memory()
    return file_names, total_graphs


def build_and_save_pair_phases(
    cfg: DictConfig,
    cache_dir: Path,
    pair_index: int,
    g: int,
    sample_index: int,
    data_rng: np.random.Generator,
) -> None:
    base_module = ensure_base()
    from fundamental_group_dataset_genus2 import generate_tight_pair_descriptor

    derived_seed = int(
        data_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
    )
    noise_candidates = tuple(
        sorted(
            {
                max(1, int(cfg.noise_pairs_per_g)),
                max(1, int(cfg.noise_pairs_per_g) * 2),
                max(1, int(cfg.noise_pairs_per_g) * 4),
            }
        )
    )
    pair, metadata, certificate = generate_tight_pair_descriptor(
        seed=derived_seed,
        noise_candidates=noise_candidates,
        subdivision=int(cfg.min_subdiv),
        max_attempts=max(5_000, int(cfg.max_tries) * int(cfg.max_expansions) * 5),
    )

    surface_files, surface_count = save_word_batch_shards(
        cfg, cache_dir, pair_index, "target", "surface", pair.surface_word_path
    )
    free_files, free_count = save_word_batch_shards(
        cfg, cache_dir, pair_index, "target", "free", pair.free_word_path
    )
    save_part(
        cache_dir,
        pair_index,
        "target",
        {
            "pair_index": pair_index,
            "g": g,
            "sample_index": sample_index,
            "first_batch_files": surface_files,
            "second_batch_files": free_files,
            "first_num_graphs": surface_count,
            "second_num_graphs": free_count,
            "metadata": metadata,
            "certificate": certificate,
        },
    )
    del surface_files, free_files
    base_module.free_memory()

    rel_a_files, rel_a_count = save_word_batch_shards(
        cfg, cache_dir, pair_index, "reliability", "a", pair.surface_word_path
    )
    rel_b_files, rel_b_count = save_word_batch_shards(
        cfg, cache_dir, pair_index, "reliability", "b", pair.surface_word_path
    )
    save_part(
        cache_dir,
        pair_index,
        "reliability",
        {
            "pair_index": pair_index,
            "g": g,
            "sample_index": sample_index,
            "first_batch_files": rel_a_files,
            "second_batch_files": rel_b_files,
            "first_num_graphs": rel_a_count,
            "second_num_graphs": rel_b_count,
        },
    )
    del pair, rel_a_files, rel_b_files
    base_module.free_memory()


def validate_cache(cfg: DictConfig) -> tuple[Path, List[int]]:
    out_dir = cache_dir(cfg)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No fixed dataset cache at {out_dir}. Run once with cache_only=true."
        )
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = cache_spec(cfg)
    if actual != expected:
        raise RuntimeError(
            f"Cache specification mismatch at {out_dir}. Generate a separate cache."
        )
    count = int(cfg.samples_per_g) * (int(cfg.max_g) - int(cfg.min_g) + 1)
    indices = list(range(count))
    np.random.default_rng(seed_value(cfg, "order_seed")).shuffle(indices)
    for index in indices:
        for part_name in ("target", "reliability"):
            path = out_dir / f"pair_{index:04d}_{part_name}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            if bool(cfg.get("construct_cache_by_phase", False)):
                item = torch_load(path)
                for key in ("first_batch_files", "second_batch_files"):
                    for file_name in item.get(key, []):
                        shard_path = out_dir / file_name
                        if not shard_path.is_file():
                            raise FileNotFoundError(shard_path)
                del item
    return out_dir, indices


def prepare_cache(cfg: DictConfig) -> tuple[Path, List[int]]:
    """Use an existing matching cache, or build it once when missing."""
    try:
        return validate_cache(cfg)
    except FileNotFoundError:
        if not bool(cfg.get("auto_build_cache", False)):
            raise
        build_cache(cfg)
        return validate_cache(cfg)


def write_results(
    cfg: DictConfig,
    output_dir: Path,
    rows: List[Dict],
    summary: Dict,
) -> None:
    """Use the established output schema and add the separated seeds."""
    base_module = ensure_base()
    base_module.write_outputs(cfg, output_dir, rows, summary)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["data_seed"] = seed_value(cfg, "data_seed")
    metrics["permutation_seed"] = seed_value(cfg, "permutation_seed")
    metrics["order_seed"] = seed_value(cfg, "order_seed")
    metrics["model_seed"] = seed_value(cfg, "model_seed")
    metrics["dataset_cache_dir"] = str(cache_dir(cfg))
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def pair_barrier(cfg: DictConfig, sequence_index: int) -> None:
    """Prevent four seed workers from retaining different pairs in node RAM."""
    world_size = int(cfg.get("sync_world_size", 1))
    sync_dir_value = cfg.get("sync_dir")
    if world_size <= 1 or not sync_dir_value:
        return

    sync_dir = Path(str(sync_dir_value))
    sync_dir.mkdir(parents=True, exist_ok=True)
    rank = int(cfg.get("sync_rank", 0))
    (sync_dir / f"pair_{sequence_index:04d}_rank_{rank:02d}.done").touch()

    deadline = time.monotonic() + float(cfg.get("sync_timeout_seconds", 1800))
    pattern = f"pair_{sequence_index:04d}_rank_*.done"
    while len(list(sync_dir.glob(pattern))) < world_size:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting at pair barrier {sequence_index}; "
                f"found {len(list(sync_dir.glob(pattern)))}/{world_size} workers"
            )
        time.sleep(0.2)
