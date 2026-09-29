
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


def _bfs_reach_chunk(
    edge_index: torch.Tensor,
    center_start: int,
    center_end: int,
    num_nodes: int,
    num_hops: int,
) -> torch.Tensor:
    """Boolean [C, num_nodes] reachability within num_hops of centers
    [center_start, center_end), via layer-by-layer frontier expansion over
    edge_index. No cross-graph reach is possible since a batched
    ComplexData's edge_index_adjacency_0_1 never has cross-graph edges."""
    src, dst = edge_index[0], edge_index[1]
    device = edge_index.device
    num_centers = center_end - center_start

    reach = torch.zeros(num_centers, num_nodes, dtype=torch.bool, device=device)
    rows = torch.arange(num_centers, device=device)
    cols = torch.arange(center_start, center_end, device=device)
    reach[rows, cols] = True
    frontier = reach.clone()

    for _ in range(num_hops):
        if not frontier.any():
            break
        contributions = frontier[:, src].float()
        accumulated = torch.zeros(num_centers, num_nodes, device=device)
        accumulated.index_add_(1, dst, contributions)
        newly_reached = (accumulated > 0) & (~reach)
        reach = reach | newly_reached
        frontier = newly_reached

    return reach


def build_ego_net_bag(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int = 2,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds the ego-nets bag of subgraphs for a (possibly batched) graph.

    Returns:
        copy_node_id: [num_copies] -- original vertex id each copy is a
            copy of.
        copy_subgraph_id: [num_copies] -- which subgraph (= center vertex
            id) each copy belongs to.
        subgraph_edge_index: [2, num_local_edges] -- edges within each
            subgraph, indexed in copy space, ready for a single GINConv
            call over the whole (block-diagonal) bag.
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]

    copy_node_ids, copy_subgraph_ids, local_edges = [], [], []
    running_total = 0

    for center_start in range(0, num_nodes, chunk_size):
        center_end = min(center_start + chunk_size, num_nodes)
        reach = _bfs_reach_chunk(edge_index, center_start, center_end, num_nodes, num_hops)

        local_row, node_id = reach.nonzero(as_tuple=True)
        num_copies = local_row.size(0)
        copy_idx = torch.arange(num_copies, device=device) + running_total

        pos = torch.full((center_end - center_start, num_nodes), -1, dtype=torch.long, device=device)
        pos[local_row, node_id] = copy_idx

        edge_valid = reach[:, src] & reach[:, dst]
        local_row_e, edge_pos = edge_valid.nonzero(as_tuple=True)
        u, w = src[edge_pos], dst[edge_pos]
        if edge_pos.numel():
            local_edges.append(torch.stack([pos[local_row_e, u], pos[local_row_e, w]]))
        else:
            local_edges.append(torch.empty(2, 0, dtype=torch.long, device=device))

        copy_node_ids.append(node_id)
        copy_subgraph_ids.append(center_start + local_row)
        running_total += num_copies

    return (
        torch.cat(copy_node_ids),
        torch.cat(copy_subgraph_ids),
        torch.cat(local_edges, dim=1),
    )


class DSSLayer(nn.Module):
    """One DSS-GNN layer: local (within-subgraph) GIN conv + cross-subgraph
    Deep-Sets conv, summed (Bevilacqua et al., Eq. 4-5)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        local_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINConv(local_mlp, train_eps=True)

        self.global_eps = nn.Parameter(torch.zeros(1))
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        subgraph_edge_index: torch.Tensor,
        copy_node_id: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        local_out = self.local_conv(x, subgraph_edge_index)

        cross_subgraph_sum = torch.zeros(num_nodes, x.size(-1), device=x.device, dtype=x.dtype)
        cross_subgraph_sum.index_add_(0, copy_node_id, x)
        global_out = self.global_mlp(
            (1 + self.global_eps) * x + cross_subgraph_sum[copy_node_id]
        )

        return self.dropout(F.relu(self.bn(local_out + global_out)))


class ESANDSSClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,
        num_hops: int = 2,
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.chunk_size = chunk_size
        self.input_proj = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList(
            [DSSLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_layers + 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def _forward_single_graph(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        copy_node_id, copy_subgraph_id, subgraph_edge_index = build_ego_net_bag(
            edge_index, num_nodes, num_hops=self.num_hops, chunk_size=self.chunk_size
        )

        # uniform constant features -- data.x_0's values are deliberately unused
        x = self.input_proj(
            torch.ones(copy_node_id.size(0), 1, device=device)
        )

        layer_outputs = [x]
        for layer in self.layers:
            x = layer(x, subgraph_edge_index, copy_node_id, num_nodes)
            layer_outputs.append(x)

        out = 0.0
        for h, head in zip(layer_outputs, self.heads):
            subgraph_embed = global_add_pool(h, copy_subgraph_id, size=num_nodes)
            graph_embed = subgraph_embed.sum(dim=0, keepdim=True)
            out = out + self.dropout(head(graph_embed))
        return out

    def forward(self, data) -> torch.Tensor:
        """Forward one PyG batch, constructing ESAN bags graph-by-graph.

        ESAN's ego-net bag is quadratic-ish in the graph size.  If we build
        one bag over a whole PyG batch, the temporary reachability tensor is
        sized by total nodes across all copies in the minibatch, even though
        there are no cross-graph edges.  Splitting here preserves the exact
        graph outputs but bounds CPU/GPU memory by the largest graph in the
        minibatch instead of the entire minibatch.
        """
        edge_index = data.edge_index_adjacency_0_1
        graph_batch = data.x_0_batch
        device = data.x_0.device
        counts = torch.bincount(graph_batch, minlength=int(data.num_graphs))
        starts = torch.cat(
            [counts.new_zeros(1), counts.cumsum(dim=0)[:-1]]
        )

        outputs = []
        src = edge_index[0]
        for start, count in zip(starts.tolist(), counts.tolist()):
            if count <= 0:
                continue
            end = start + count
            # PyG batches node attributes contiguously, and ComplexData's
            # adjacency has no cross-graph edges, so filtering by source range
            # safely extracts the graph-local 1-skeleton.
            edge_mask = (src >= start) & (src < end)
            local_edge_index = edge_index[:, edge_mask] - start
            outputs.append(
                self._forward_single_graph(local_edge_index, int(count), device)
            )
        return torch.cat(outputs, dim=0)


def build_esan_model(
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_classes: int = 2,
    dropout: float = 0.0,
    num_hops: int = 2,
    chunk_size: int = 2048,
    device: str = "cpu",
) -> ESANDSSClassifier:
    return ESANDSSClassifier(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        num_hops=num_hops,
        chunk_size=chunk_size,
    ).to(device)
