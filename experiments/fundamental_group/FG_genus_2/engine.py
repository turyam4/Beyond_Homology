"""Training/evaluation runners for the tight FG genus-2 benchmark."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn import CosineEmbeddingLoss

from utils import (
    DiskBatchSequence,
    build_cache,
    cache_dir,
    ensure_base,
    free_memory,
    iterate_minibatches,
    load_pair_part,
    pair_barrier,
    prepare_cache,
    seed_value,
    write_results,
)

SCALAR_MLP_BASELINES = {"mlp", "dist_mlp", "boundary_mlp", "spec_mlp"}


class TensorFeatureSequence:
    """Small CPU feature minibatches replacing heavyweight ComplexData batches."""

    def __init__(self, features: torch.Tensor) -> None:
        self.features = features.cpu()
        self.num_graphs = int(self.features.size(0))

    def iter_minibatches(self, batch_size: int):
        for start in range(0, self.num_graphs, max(1, int(batch_size))):
            yield self.features[start : start + batch_size].clone()


def _count_features_from_batch(data) -> torch.Tensor:
    num_graphs = int(data.num_graphs)
    n0 = torch.bincount(data.x_0_batch.cpu(), minlength=num_graphs).float()
    n1 = torch.bincount(data.x_1_batch.cpu(), minlength=num_graphs).float()
    if getattr(data, "x_2", None) is None:
        n2 = torch.zeros_like(n0)
    else:
        n2 = torch.bincount(data.x_2_batch.cpu(), minlength=num_graphs).float()
    return torch.stack((n0 - n1 + n2, n0, n1, n2), dim=1)


def _scalar_features_from_batch(cfg: DictConfig, data, name: str) -> torch.Tensor:
    if name == "mlp":
        return _count_features_from_batch(data)
    if name == "dist_mlp":
        from baselines.dist_mlp.model import distance_histogram_features

        model_cfg = cfg.models.dist_mlp
        return torch.stack(
            [
                distance_histogram_features(
                    item,
                    high_rank=int(model_cfg.high_rank),
                    max_dist=int(model_cfg.max_dist),
                )
                for item in data.to_data_list()
            ],
            dim=0,
        ).float()
    if name == "boundary_mlp":
        from baselines.boundary_mlp.model import boundary_loop_features

        return torch.stack(
            [boundary_loop_features(item) for item in data.to_data_list()],
            dim=0,
        ).float()
    if name == "spec_mlp":
        from baselines.spec_mlp.model import DEFAULT_OPERATORS, spectral_features

        model_cfg = cfg.models.spec_mlp
        operators = tuple(
            str(operator)
            for operator in model_cfg.get("operators", list(DEFAULT_OPERATORS))
        )
        return torch.stack(
            [
                spectral_features(
                    item,
                    k=int(model_cfg.k),
                    moment_L=int(model_cfg.moment_L),
                    operators=operators,
                )
                for item in data.to_data_list()
            ],
            dim=0,
        ).float()
    raise ValueError(f"Unknown scalar baseline {name!r}")


def _to_scalar_feature_sequence(cfg: DictConfig, complex_batch, name: str):
    """Materialize tiny scalar features once per pair, not once per epoch."""
    if name == "spec_mlp":
        # Each sequence is made exclusively from permutations of one complex.
        # Spectra are permutation invariant, so computing all 32 copies would
        # repeat the same dense eigendecompositions 32 times.
        from baselines.spec_mlp.model import DEFAULT_OPERATORS, spectral_features

        first_minibatch = next(
            iterate_minibatches(complex_batch, max(1, int(cfg.batch_size)))
        )
        model_cfg = cfg.models.spec_mlp
        operators = tuple(
            str(operator)
            for operator in model_cfg.get("operators", list(DEFAULT_OPERATORS))
        )
        first_item = first_minibatch.to_data_list()[0]
        feature = spectral_features(
            first_item,
            k=int(model_cfg.k),
            moment_L=int(model_cfg.moment_L),
            operators=operators,
        ).unsqueeze(0).cpu()
        if hasattr(complex_batch, "num_graphs"):
            num_graphs = int(complex_batch.num_graphs)
        else:
            num_graphs = sum(int(batch.num_graphs) for batch in complex_batch)
        del first_minibatch
        free_memory()
        return TensorFeatureSequence(feature.repeat(num_graphs, 1))

    chunks: List[torch.Tensor] = []
    for mini_batch in iterate_minibatches(complex_batch, max(1, int(cfg.batch_size))):
        chunks.append(_scalar_features_from_batch(cfg, mini_batch, name).cpu())
        del mini_batch
        free_memory()
    return TensorFeatureSequence(torch.cat(chunks, dim=0))


def _maybe_scalar_features(cfg: DictConfig, batch):
    name = str(cfg.get("baseline_name", ""))
    if name in SCALAR_MLP_BASELINES:
        return _to_scalar_feature_sequence(cfg, batch, name)
    return batch


def _build_subcomplex_on_load(cfg: DictConfig) -> bool:
    """Compact shards need subcomplex features only for the SMCN model."""
    return bool(cfg.get("compact_subcomplex_cache", False)) and not cfg.get(
        "baseline_name"
    )


def _baseline_view(cfg: DictConfig) -> str | None:
    name = str(cfg.get("baseline_name", ""))
    if name in {"gin", "esan"}:
        return "graph"
    if name == "hasse_gin":
        return "hasse"
    return None


def run_pairwise_brec(cfg: DictConfig) -> None:
    """Shared target/reliability pairwise BREC loop for SMCN and baselines."""
    base = ensure_base()
    base.validate_cfg(cfg)
    if bool(cfg.get("cache_only", False)):
        build_cache(cfg)
        return

    fixed_cache_dir, pair_order = prepare_cache(cfg)
    cfg.device = base.resolve_device(str(cfg.device))
    cfg.output_dir = str(base.resolve_output_dir(str(cfg.output_dir)))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.log").write_text("", encoding="utf-8")
    (output_dir / "certificates.jsonl").write_text("", encoding="utf-8")

    model_seed = seed_value(cfg, "model_seed")
    # Reset immediately before model construction: cache creation/loading and
    # permutation generation can never perturb model initialization.
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    base.log_message(output_dir, "START RUN fg_fixed_data_codex")
    base.log_message(
        output_dir,
        f"SEEDS data={seed_value(cfg, 'data_seed')} "
        f"permutation={seed_value(cfg, 'permutation_seed')} "
        f"order={seed_value(cfg, 'order_seed')} model={model_seed}",
    )
    base.log_message(output_dir, f"Using device: {cfg.device}")
    base.log_message(output_dir, f"Using fixed cache: {fixed_cache_dir.name}")
    if not cfg.get("baseline_name"):
        base.log_message(
            output_dir,
            "Subcomplex marking: "
            f"{'binary incidence' if cfg.get('binary_marking', False) else 'distance'}",
        )
    base.log_message(output_dir, f"Pair order: {pair_order}")

    fresh_model_per_pair = bool(cfg.get("fresh_model_per_pair", False))
    model = None
    optimizer = None
    if not fresh_model_per_pair:
        model = base.build_model(cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    loss_fn = CosineEmbeddingLoss(margin=cfg.margin)

    rows: List[Dict] = []
    target_success_ids: List[int] = []
    reliability_pass_ids: List[int] = []
    brec_solved_ids: List[int] = []
    oom_skipped_pair_ids: List[int] = []
    start_time = time.time()
    total_pairs = len(pair_order)

    for sequence_index, cached_index in enumerate(pair_order):
        # Load only the two target copy-sets. Reliability stays on disk until
        # target evaluation and cleanup have completed.
        target_item = load_pair_part(
            fixed_cache_dir / f"pair_{cached_index:04d}_target.pt",
            str(cfg.device).startswith("cuda") and bool(cfg.get("pin_memory", False)),
        )
        pair_index = int(target_item["pair_index"])
        g = int(target_item["g"])
        sample_index = int(target_item["sample_index"])
        metadata = target_item.pop("metadata")
        certificate = target_item.pop("certificate")
        if fresh_model_per_pair:
            if model is not None:
                del model, optimizer
                free_memory()
            pair_model_seed = model_seed + pair_index
            torch.manual_seed(pair_model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(pair_model_seed)
            model = base.build_model(cfg)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        if "first_batch_files" in target_item:
            surface_batch = DiskBatchSequence(
                fixed_cache_dir,
                target_item.pop("first_batch_files"),
                int(target_item.pop("first_num_graphs")),
                build_subcomplex=_build_subcomplex_on_load(cfg),
                low_rk=int(cfg.low_rk),
                high_rk=int(cfg.high_rk),
                binary_marking=bool(cfg.get("binary_marking", False)),
                view=_baseline_view(cfg),
            )
            free_batch = DiskBatchSequence(
                fixed_cache_dir,
                target_item.pop("second_batch_files"),
                int(target_item.pop("second_num_graphs")),
                build_subcomplex=_build_subcomplex_on_load(cfg),
                low_rk=int(cfg.low_rk),
                high_rk=int(cfg.high_rk),
                binary_marking=bool(cfg.get("binary_marking", False)),
                view=_baseline_view(cfg),
            )
        else:
            surface_batch = target_item.pop("first_batch")
            free_batch = target_item.pop("second_batch")
        surface_batch = _maybe_scalar_features(cfg, surface_batch)
        free_batch = _maybe_scalar_features(cfg, free_batch)

        with (output_dir / "certificates.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pair_index": pair_index,
                        "sequence_index": sequence_index,
                        "g": g,
                        "sample_index": sample_index,
                        **certificate,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        base.log_message(
            output_dir,
            f"START target pair={sequence_index + 1}/{total_pairs} cached_pair={pair_index}",
        )
        target_result, target_oom = base.try_evaluate_pair(
            cfg,
            output_dir,
            pair_index,
            "target_surface_vs_free",
            surface_batch,
            free_batch,
            model,
            optimizer,
            loss_fn,
        )
        target_passed = bool(
            target_result
            and target_result["above_threshold"]
            and not target_result["close_to_threshold"]
        )
        if target_passed:
            target_success_ids.append(pair_index)
        rows.append(
            {
                "pair_index": pair_index,
                "g": g,
                "sample_index": sample_index,
                "trial_type": "target_surface_vs_free",
                "passed": target_passed,
                "oom_skipped": target_result is None,
                **metadata,
                **(target_result or {}),
            }
        )
        # t2_score has synchronized the target work. Removing these last
        # references lets reliability reuse the allocator blocks instead of
        # retaining all four copy-sets on the GPU together.
        del target_item, surface_batch, free_batch
        base.free_memory()

        base.log_message(
            output_dir,
            f"START reliability pair={sequence_index + 1}/{total_pairs} cached_pair={pair_index}",
        )
        reliability_item = load_pair_part(
            fixed_cache_dir / f"pair_{cached_index:04d}_reliability.pt",
            str(cfg.device).startswith("cuda") and bool(cfg.get("pin_memory", False)),
        )
        if "first_batch_files" in reliability_item:
            rel_a_batch = DiskBatchSequence(
                fixed_cache_dir,
                reliability_item.pop("first_batch_files"),
                int(reliability_item.pop("first_num_graphs")),
                build_subcomplex=_build_subcomplex_on_load(cfg),
                low_rk=int(cfg.low_rk),
                high_rk=int(cfg.high_rk),
                binary_marking=bool(cfg.get("binary_marking", False)),
                view=_baseline_view(cfg),
            )
            rel_b_batch = DiskBatchSequence(
                fixed_cache_dir,
                reliability_item.pop("second_batch_files"),
                int(reliability_item.pop("second_num_graphs")),
                build_subcomplex=_build_subcomplex_on_load(cfg),
                low_rk=int(cfg.low_rk),
                high_rk=int(cfg.high_rk),
                binary_marking=bool(cfg.get("binary_marking", False)),
                view=_baseline_view(cfg),
            )
        else:
            rel_a_batch = reliability_item.pop("first_batch")
            rel_b_batch = reliability_item.pop("second_batch")
        rel_a_batch = _maybe_scalar_features(cfg, rel_a_batch)
        rel_b_batch = _maybe_scalar_features(cfg, rel_b_batch)
        reliability_result, reliability_oom = base.try_evaluate_pair(
            cfg,
            output_dir,
            pair_index,
            "reliability_surface_vs_surface",
            rel_a_batch,
            rel_b_batch,
            model,
            optimizer,
            loss_fn,
        )
        reliability_passed = bool(
            reliability_result
            and not reliability_result["above_threshold"]
            and not reliability_result["close_to_threshold"]
        )
        if reliability_passed:
            reliability_pass_ids.append(pair_index)
        rows.append(
            {
                "pair_index": pair_index,
                "g": g,
                "sample_index": sample_index,
                "trial_type": "reliability_surface_vs_surface",
                "passed": reliability_passed,
                "oom_skipped": reliability_result is None,
                **metadata,
                **(reliability_result or {}),
            }
        )

        if target_passed and reliability_passed:
            brec_solved_ids.append(pair_index)
        if target_oom or reliability_oom:
            oom_skipped_pair_ids.append(pair_index)

        base.log_message(
            output_dir,
            f"RESULT pair={sequence_index + 1}/{total_pairs} cached_pair={pair_index} "
            f"target_passed={target_passed} "
            f"reliability_passed={reliability_passed}",
        )
        del reliability_item, rel_a_batch, rel_b_batch
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if fresh_model_per_pair:
            del model, optimizer
            model = None
            optimizer = None
        base.free_memory()
        pair_barrier(cfg, sequence_index)

    target_scores = [
        row["t2_score"]
        for row in rows
        if row["trial_type"] == "target_surface_vs_free" and not row["oom_skipped"]
    ]
    reliability_scores = [
        row["t2_score"]
        for row in rows
        if row["trial_type"] == "reliability_surface_vs_surface"
        and not row["oom_skipped"]
    ]
    runtime = time.time() - start_time
    summary = {
        "architecture": "SMCN",
        "model_type": cfg.model_type,
        "target_success_count": len(target_success_ids),
        "target_total": total_pairs,
        "reliability_pass_count": len(reliability_pass_ids),
        "reliability_total": total_pairs,
        "brec_solved_count": len(brec_solved_ids),
        "brec_solved_total": total_pairs,
        "target_success_ids": target_success_ids,
        "reliability_pass_ids": reliability_pass_ids,
        "brec_solved_ids": brec_solved_ids,
        "oom_skipped_pair_ids": oom_skipped_pair_ids,
        "oom_skipped_count": len(oom_skipped_pair_ids),
        "mean_t2_target": float(np.mean(target_scores)) if target_scores else 0.0,
        "mean_t2_reliability": (
            float(np.mean(reliability_scores)) if reliability_scores else 0.0
        ),
        "std_t2_target": float(np.std(target_scores)) if target_scores else 0.0,
        "std_t2_reliability": (
            float(np.std(reliability_scores)) if reliability_scores else 0.0
        ),
        "runtime_seconds_total": runtime,
        "runtime_seconds_per_pair": runtime / total_pairs if total_pairs else 0.0,
    }
    write_results(cfg, output_dir, rows, summary)
    base.log_message(output_dir, "END RUN")
    print(
        f"FINAL model_seed={model_seed}: BREC-solved "
        f"{len(brec_solved_ids)}/{total_pairs}",
        flush=True,
    )


def _build_model(cfg: DictConfig, name: str) -> nn.Module:
    model_cfg = cfg.models[name]
    common = {
        "hidden_dim": int(model_cfg.hidden_dim),
        "num_layers": int(model_cfg.num_layers),
        "num_classes": 2,
        "dropout": float(model_cfg.dropout),
    }
    if name == "mlp":
        from baselines.count_mlp.model import CountMLP

        model = CountMLP(hidden_dim=int(model_cfg.hidden_dim))
    elif name == "dist_mlp":
        from baselines.dist_mlp.model import DistanceHistogramMLP

        model = DistanceHistogramMLP(
            max_dist=int(model_cfg.max_dist),
            hidden_dim=int(model_cfg.hidden_dim),
        )
    elif name == "boundary_mlp":
        from baselines.boundary_mlp.model import BoundaryLoopMLP

        model = BoundaryLoopMLP(hidden_dim=int(model_cfg.hidden_dim))
    elif name == "spec_mlp":
        from baselines.spec_mlp.model import DEFAULT_OPERATORS, SpectralMLP

        operators = tuple(
            str(operator)
            for operator in model_cfg.get("operators", list(DEFAULT_OPERATORS))
        )
        model = SpectralMLP(
            k=int(model_cfg.k),
            moment_L=int(model_cfg.moment_L),
            operators=operators,
            hidden_dim=int(model_cfg.hidden_dim),
            num_layers=int(model_cfg.num_layers),
            dropout=float(model_cfg.dropout),
        )
    elif name == "gin":
        from baselines.gin.model import GINClassifier

        model = GINClassifier(**common)
    elif name == "esan":
        from baselines.esan.model import build_esan_model

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
    elif name == "hasse_gin":
        from baselines.hasse.model import HasseGINClassifier

        model = HasseGINClassifier(**common)
    else:
        raise ValueError(f"Unknown baseline {name!r}")
    return model.to(cfg.device)


def _enabled_models(cfg: DictConfig) -> list[str]:
    names = [str(name) for name in cfg.model_names]
    unknown = [name for name in names if name not in cfg.models]
    if unknown:
        raise ValueError(f"model_names contains unknown entries: {unknown}")
    enabled = [name for name in names if bool(cfg.models[name].enabled)]
    if not enabled:
        raise ValueError("At least one model in model_names must be enabled")
    return enabled


def _run_one(cfg: DictConfig, name: str) -> dict:
    base = ensure_base()
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(run_cfg):
        # Keep fields required by the shared tight-benchmark validator.
        run_cfg.model_type = "homp"
        run_cfg.output_ranks = [0, int(run_cfg.high_rk)]
        run_tag = str(run_cfg.get("run_tag") or "")
        model_directory = str(run_cfg.get("run_model_name") or name)
        seed_directory = f"seed{int(run_cfg.model_seed)}"
        if run_tag:
            seed_directory += f"_{run_tag}"
        run_cfg.output_dir = str(
            Path(str(cfg.output_dir)) / model_directory / seed_directory
        )
        run_cfg.baseline_name = name
        run_cfg.baseline_model = OmegaConf.to_container(cfg.models[name], resolve=True)

    original_builder: Callable = base.build_model
    model_info: dict[str, int] = {}

    def build_baseline(active_cfg: DictConfig) -> nn.Module:
        model = _build_model(active_cfg, name)
        model_info["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        return model

    base.build_model = build_baseline
    try:
        run_pairwise_brec(run_cfg)
    finally:
        base.build_model = original_builder

    metrics_path = base.resolve_output_dir(run_cfg.output_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["architecture"] = name
    metrics["parameter_count"] = model_info["parameter_count"]
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def run_baselines(cfg: DictConfig) -> None:
    base = ensure_base()
    base.validate_cfg(cfg)
    names = _enabled_models(cfg)

    if bool(cfg.get("cache_only", False)):
        build_cache(cfg)
        return
    if not (cache_dir(cfg) / "manifest.json").is_file():
        if not bool(cfg.get("auto_build_cache", True)):
            raise FileNotFoundError(
                "Fixed cache is missing. Run with cache_only=true or set "
                "auto_build_cache=true."
            )
        build_cache(cfg)

    summaries = {name: _run_one(cfg, name) for name in names}
    print(
        "FINAL baseline results: "
        + ", ".join(
            f"{name}={result['brec_solved_count']}/{result['brec_solved_total']}"
            for name, result in summaries.items()
        ),
        flush=True,
    )

SMCN_MODEL_NAMES = {"smcn", "subcomplex"}


def selected_model_names(cfg: DictConfig) -> List[str]:
    names = [str(name) for name in cfg.model_names]
    if not names:
        raise ValueError("model_names cannot be empty")
    return names


def is_smcn_run(cfg: DictConfig) -> bool:
    names = selected_model_names(cfg)
    return len(names) == 1 and names[0] in SMCN_MODEL_NAMES


def run(cfg: DictConfig) -> None:
    """Run the required fundamental-group evaluator from this single module."""
    if is_smcn_run(cfg):
        run_pairwise_brec(cfg)
    else:
        run_baselines(cfg)


# Backward-compatible name for older direct imports/scripts.
run_smcn = run_pairwise_brec
