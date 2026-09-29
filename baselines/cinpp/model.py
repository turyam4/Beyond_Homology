
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _require_tensor(obj: object, name: str) -> Tensor:
    value = getattr(obj, name, None)
    if not isinstance(value, Tensor):
        raise AttributeError(f"expected tensor attribute {name!r}")
    return value


def _scatter_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if index.dim() != 1 or values.size(0) != index.numel():
        raise ValueError("index must be [num_messages] and align with values")
    out = values.new_zeros((dim_size, values.size(-1)))
    return out.index_add_(0, index.to(values.device), values)


def _pool_sum(x: Tensor, batch: Tensor, size: int) -> Tensor:
    if batch.dim() != 1 or batch.numel() != x.size(0):
        raise ValueError("cochain batch vector must align with cochain features")
    return _scatter_sum(x, batch.to(x.device), size)


def _make_update(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
    )


class CINppCochainLayer(nn.Module):
    """One CIN++ cochain update at a fixed rank.

    ``up_index``, ``down_index`` and ``boundary_index`` must have shape
    ``[2, num_messages]`` with rows ordered as ``[target_cell, source_cell]``.
    Empty streams must be represented as explicit ``[2, 0]`` long tensors.
    """

    def __init__(self, input_dim: int, hidden_dim: int, train_eps: bool = False) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        eps = torch.zeros(1)
        if train_eps:
            self.eps_up = nn.Parameter(eps.clone())
            self.eps_down = nn.Parameter(eps.clone())
            self.eps_boundary = nn.Parameter(eps.clone())
        else:
            self.register_buffer("eps_up", eps.clone())
            self.register_buffer("eps_down", eps.clone())
            self.register_buffer("eps_boundary", eps.clone())
        self.update_up = _make_update(input_dim, hidden_dim)
        self.update_down = _make_update(input_dim, hidden_dim)
        self.update_boundary = _make_update(input_dim, hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _message(x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("message edge_index must have shape [2, num_messages]")
        if edge_index.numel() == 0:
            return x.new_zeros(x.shape)
        dst = edge_index[0].to(torch.long).to(x.device)
        src = edge_index[1].to(torch.long).to(x.device)
        return _scatter_sum(torch.index_select(x, 0, src), dst, x.size(0))

    def forward(
        self,
        x: Tensor,
        up_index: Tensor,
        down_index: Tensor,
        boundary_index: Tensor,
    ) -> Tensor:
        up = self.update_up(self._message(x, up_index) + (1.0 + self.eps_up) * x)
        down = self.update_down(self._message(x, down_index) + (1.0 + self.eps_down) * x)
        boundary = self.update_boundary(
            self._message(x, boundary_index) + (1.0 + self.eps_boundary) * x
        )
        return self.combine(torch.cat([up, down, boundary], dim=-1))


class CINppConv(nn.Module):
    """CIN++ layer applied independently across all cochain ranks."""

    def __init__(
        self, input_dim: int, hidden_dim: int, max_dim: int = 2, train_eps: bool = False
    ) -> None:
        super().__init__()
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        self.max_dim = max_dim
        self.levels = nn.ModuleList(
            CINppCochainLayer(input_dim, hidden_dim, train_eps=train_eps)
            for _ in range(max_dim + 1)
        )

    def forward(self, cochains: Sequence[object]) -> list[Tensor]:
        if len(cochains) != self.max_dim + 1:
            raise ValueError(f"expected {self.max_dim + 1} cochains, got {len(cochains)}")
        outputs = []
        for rank, cochain in enumerate(cochains):
            outputs.append(
                self.levels[rank](
                    _require_tensor(cochain, "x"),
                    _require_tensor(cochain, "up_index"),
                    _require_tensor(cochain, "down_index"),
                    _require_tensor(cochain, "boundary_index"),
                )
            )
        return outputs


class CINpp(nn.Module):
    """Graph-level CIN++ classifier/embedding head over a CWN-style complex.

    The forward input must expose ``cochains`` as a sequence of rank objects.
    Each rank object must expose:
        ``x``: cell features ``[num_cells_rank, feature_dim]``;
        ``batch``: graph id per cell ``[num_cells_rank]``;
        ``up_index``: upper-neighbor message index ``[2, num_messages]``;
        ``down_index``: lower-neighbor message index ``[2, num_messages]``;
        ``boundary_index``: boundary-cell message index ``[2, num_messages]``.

    Args:
        num_input_features: Initial cell feature dimension.
        num_classes: Output dimension.
        num_layers: Number of CIN++ message-passing layers.
        hidden_dim: Hidden cochain dimension.
        max_dim: Highest cochain rank included.
        readout_dims: Ranks pooled into the graph representation.
        final_readout: ``"sum"`` or ``"mean"`` over pooled ranks.
        dropout_rate: Dropout applied before the final linear classifier.
        train_eps: Whether residual epsilon values are learned.
    """

    def __init__(
        self,
        num_input_features: int,
        num_classes: int,
        num_layers: int,
        hidden_dim: int,
        max_dim: int = 2,
        readout_dims: tuple[int, ...] | None = None,
        final_readout: str = "sum",
        dropout_rate: float = 0.5,
        train_eps: bool = False,
    ) -> None:
        super().__init__()
        if num_input_features <= 0 or num_classes <= 0 or num_layers <= 0 or hidden_dim <= 0:
            raise ValueError("feature dimensions, num_classes, and num_layers must be positive")
        if final_readout not in {"sum", "mean"}:
            raise ValueError("final_readout must be 'sum' or 'mean'")
        self.max_dim = max_dim
        self.readout_dims = tuple(range(max_dim + 1)) if readout_dims is None else readout_dims
        if any(dim < 0 or dim > max_dim for dim in self.readout_dims):
            raise ValueError("readout_dims must be within [0, max_dim]")
        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.input_projections = nn.ModuleList(
            nn.Linear(num_input_features, hidden_dim) for _ in range(max_dim + 1)
        )
        self.convs = nn.ModuleList(
            CINppConv(hidden_dim, hidden_dim, max_dim=max_dim, train_eps=train_eps)
            for _ in range(num_layers)
        )
        self.rank_heads = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(max_dim + 1)
        )
        self.output = nn.Linear(hidden_dim * 2, num_classes)

    def _batch_size(self, cochains: Sequence[object]) -> int:
        batch0 = _require_tensor(cochains[0], "batch")
        if batch0.numel() == 0:
            raise ValueError("rank-0 cochain batch is empty")
        return int(batch0.max().item()) + 1

    def forward(self, complex_batch: object) -> Tensor:
        cochains = getattr(complex_batch, "cochains", None)
        if not isinstance(cochains, Sequence):
            raise AttributeError("complex_batch must expose a cochains sequence")
        if len(cochains) != self.max_dim + 1:
            raise ValueError(f"expected {self.max_dim + 1} cochains, got {len(cochains)}")

        xs = [
            self.input_projections[rank](_require_tensor(cochain, "x").to(torch.float32))
            for rank, cochain in enumerate(cochains)
        ]
        for conv in self.convs:
            for rank, x in enumerate(xs):
                cochains[rank].x = x
            xs = conv(cochains)

        batch_size = self._batch_size(cochains)
        pooled = []
        for rank in self.readout_dims:
            batch = _require_tensor(cochains[rank], "batch").to(torch.long)
            pooled_rank = _pool_sum(xs[rank], batch, batch_size)
            pooled.append(torch.relu(self.rank_heads[rank](pooled_rank)))
        stacked = torch.stack(pooled, dim=0)
        graph_x = stacked.sum(dim=0) if self.final_readout == "sum" else stacked.mean(dim=0)
        graph_x = F.dropout(graph_x, p=self.dropout_rate, training=self.training)
        return self.output(graph_x)
