import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)

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

        # one head per readout: input projection + each GIN block
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_layers + 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        # uniform constant features -- data.x_0's values are deliberately unused
        x = self.input_proj(torch.ones(data.x_0.size(0), 1, device=data.x_0.device))
        edge_index = data.edge_index_adjacency_0_1
        batch = data.x_0_batch

        layer_outputs = [x]
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            layer_outputs.append(x)

        out = 0.0
        for h, head in zip(layer_outputs, self.heads):
            out = out + self.dropout(head(global_add_pool(h, batch)))
        return out
