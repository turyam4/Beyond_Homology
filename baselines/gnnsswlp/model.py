
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


def _require_tensor(batch: object, name: str) -> Tensor:
    value = getattr(batch, name, None)
    if not isinstance(value, Tensor):
        raise AttributeError(f"batch must provide tensor attribute {name!r}")
    return value


def _scatter_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if index.dim() != 1:
        raise ValueError("scatter index must have shape [num_messages]")
    if index.numel() != values.size(0):
        raise ValueError("scatter index length must match the number of messages")
    out = values.new_zeros((dim_size, values.size(-1)))
    return out.index_add_(0, index.to(values.device), values)


def _global_mean_pool(x: Tensor, batch: Tensor) -> Tensor:
    if batch.dim() != 1 or batch.numel() != x.size(0):
        raise ValueError("batch must have shape [num_nodes] and align with x")
    batch = batch.to(x.device)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    summed = _scatter_sum(x, batch, num_graphs)
    counts = x.new_zeros((num_graphs, 1)).index_add_(
        0, batch, x.new_ones((x.size(0), 1))
    )
    if torch.any(counts == 0):
        raise ValueError("empty graph encountered during mean pooling")
    return summed / counts


class NormReLU(nn.Sequential):
    """BatchNorm1d followed by ReLU for hidden subgraph-node embeddings."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        super().__init__(nn.BatchNorm1d(dim), nn.ReLU())


class MLP(nn.Sequential):
    """Two-layer MLP used by SWL aggregation and prediction blocks."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int | None = None, norm: bool = True
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        hidden_dim = hidden_dim or input_dim
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            NormReLU(hidden_dim) if norm else nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class DistanceAtomEncoder(nn.Module):
    """Encode lifted subgraph nodes with shortest-path distance embeddings.

    Input data must expose ``x`` and ``d``.  With ``encode_atoms=False`` this
    matches the GraphCount/substructure setup in SWL: node identity features are
    ignored and only distance embeddings are used.  With ``encode_atoms=True``,
    OGB's ``AtomEncoder`` is required, matching the ZINC path in SWL.
    """

    def __init__(self, hidden_dim: int, max_distance: int, encode_atoms: bool = False) -> None:
        super().__init__()
        if hidden_dim <= 0 or max_distance < 0:
            raise ValueError("hidden_dim must be positive and max_distance non-negative")
        self.max_distance = max_distance
        self.encode_atoms = encode_atoms
        if encode_atoms:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder
            except ImportError as exc:
                raise ImportError("encode_atoms=True requires ogb.graphproppred") from exc
            self.atom_encoder: nn.Module | None = AtomEncoder(hidden_dim)
        else:
            self.atom_encoder = None
        self.distance_embedding = nn.Embedding(max_distance + 1, hidden_dim)

    def forward(self, batch: object) -> Tensor:
        distances = torch.clamp(_require_tensor(batch, "d").to(torch.long), max=self.max_distance)
        distance_x = self.distance_embedding(distances)
        if self.atom_encoder is None:
            return distance_x
        atom_x = _require_tensor(batch, "x")
        if atom_x.dim() == 1:
            atom_x = atom_x[:, None]
        return self.atom_encoder(atom_x.to(torch.long)) + distance_x


class BondMessageEncoder(nn.Module):
    """Optional edge-attribute encoder for local SWL message streams."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        try:
            from ogb.graphproppred.mol_encoder import BondEncoder
        except ImportError as exc:
            raise ImportError("edge attributes require ogb.graphproppred BondEncoder") from exc
        self.bond_encoder = BondEncoder(hidden_dim)

    def forward(self, message: Tensor, attrs: Tensor | None) -> Tensor:
        if attrs is None:
            return torch.relu(message)
        if attrs.dim() == 1:
            attrs = attrs[:, None]
        return torch.relu(message + self.bond_encoder(attrs.to(torch.long)))


class SSWLPlusAggregation(nn.Module):
    """One GNN-SSWL+ aggregation block over a selected SWL scheme."""

    def __init__(self, input_dim: int, output_dim: int, scheme: Iterable[str], gin: bool = True) -> None:
        super().__init__()
        scheme = tuple(scheme)
        if not scheme:
            raise ValueError("scheme must contain at least one aggregation operator")
        self.scheme = scheme
        self.gin = gin
        self.eps = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in scheme})
        self.encoders = nn.ModuleDict(
            {name: BondMessageEncoder(input_dim) for name in scheme if name.endswith("L")}
        )
        self.mlps = nn.ModuleDict({name: MLP(input_dim, output_dim) for name in scheme})
        self.norm = NormReLU(output_dim)

    def _aggregate(self, x: Tensor, batch: object, name: str) -> Tensor:
        edge_index = _require_tensor(batch, f"index_{name}").to(x.device)
        if edge_index.shape[0] != 2:
            raise ValueError(f"index_{name} must have shape [2, num_messages]")
        dst, src = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
        message = torch.index_select(x, dim=0, index=src)
        if name in self.encoders:
            attrs = getattr(batch, f"attrs_{name}", None)
            if attrs is not None and not isinstance(attrs, Tensor):
                raise AttributeError(f"attrs_{name} must be a tensor when present")
            message = self.encoders[name](message, attrs.to(x.device) if attrs is not None else None)
        return _scatter_sum(message, dst, x.size(0))

    def forward(self, x: Tensor, batch: object) -> Tensor:
        updates = []
        for name in self.scheme:
            residual = x * (1.0 + self.eps[name]) if self.gin else torch.zeros_like(x)
            updates.append(self.mlps[name](residual + self._aggregate(x, batch, name)))
        return self.norm(torch.stack(updates, dim=0).sum(dim=0))


class GNNSSWLPlus(nn.Module):
    """Graph-level GNN-SSWL+ model.

    The forward input must be a PyG-like data object with ``x``, ``d``,
    ``batch``, ``index_uL``, ``index_vL``, ``index_vv`` and ``index_uG``.  The
    first three index tensors define the SSWL+ message-passing streams; the
    final ``uG`` tensor is the graph-level SWL aggregation used before pooling.

    Args:
        hidden_dim: Hidden dimension for lifted subgraph-node embeddings.
        output_dim: Output dimension of the graph prediction head.
        num_layers: Number of repeated SSWL+ aggregation blocks.
        max_distance: Maximum shortest-path distance embedding index.
        encode_atoms: Whether to use OGB AtomEncoder in addition to distances.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        output_dim: int = 1,
        num_layers: int = 6,
        max_distance: int = 5,
        encode_atoms: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim, output_dim, and num_layers must be positive")
        self.encoder = DistanceAtomEncoder(hidden_dim, max_distance, encode_atoms)
        self.layers = nn.ModuleList(
            SSWLPlusAggregation(hidden_dim, hidden_dim, ("uL", "vL", "vv"), gin=True)
            for _ in range(num_layers)
        )
        self.global_layer = SSWLPlusAggregation(hidden_dim, hidden_dim, ("uG",), gin=False)
        self.predict = MLP(hidden_dim, output_dim, hidden_dim=hidden_dim * 2, norm=False)

    def forward(self, batch: object) -> Tensor:
        x = self.encoder(batch)
        for layer in self.layers:
            x = layer(x, batch)
        x = self.global_layer(x, batch)
        return self.predict(_global_mean_pool(x, _require_tensor(batch, "batch").to(torch.long)))
