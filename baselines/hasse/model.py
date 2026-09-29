import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


def build_hasse_graph(data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (edge_index, rank_ids, batch) for the Hasse diagram of a
    (possibly batched) ComplexData object."""
    if hasattr(data, "hasse_edge_index"):
        return data.hasse_edge_index, data.hasse_rank_ids, data.hasse_batch

    n0, n1 = data.x_0.size(0), data.x_1.size(0)
    n2 = data.x_2.size(0) if getattr(data, "x_2", None) is not None else 0
    device = data.x_0.device

    inc01 = data.edge_index_incidence_0_1  # row 0: vertex idx, row 1: edge idx
    src = [inc01[0], inc01[1] + n0]
    dst = [inc01[1] + n0, inc01[0]]
    if n2 > 0:
        inc12 = data.edge_index_incidence_1_2  # row 0: edge idx, row 1: face idx
        src += [inc12[0] + n0, inc12[1] + n0 + n1]
        dst += [inc12[1] + n0 + n1, inc12[0] + n0]
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])

    rank_ids = torch.cat(
        [
            torch.zeros(n0, dtype=torch.long, device=device),
            torch.ones(n1, dtype=torch.long, device=device),
            torch.full((n2,), 2, dtype=torch.long, device=device),
        ]
    )

    batches = [data.x_0_batch, data.x_1_batch]
    if n2 > 0:
        batches.append(data.x_2_batch)
    batch = torch.cat(batches)

    return edge_index, rank_ids, batch


class HasseGINClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank_embedding = nn.Embedding(3, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_layers + 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        edge_index, rank_ids, batch = build_hasse_graph(data)
        x = self.rank_embedding(rank_ids)

        layer_outputs = [x]
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            layer_outputs.append(x)

        out = 0.0
        for h, head in zip(layer_outputs, self.heads):
            out = out + self.dropout(head(global_add_pool(h, batch)))
        return out
