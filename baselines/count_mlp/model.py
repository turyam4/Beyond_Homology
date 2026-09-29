from __future__ import annotations

import torch
import torch.nn as nn


def count_features(data) -> torch.Tensor:
    n0 = int(data.x_0.size(0))
    n1 = int(data.x_1.size(0))
    n2 = int(data.x_2.size(0)) if getattr(data, "x_2", None) is not None else 0
    chi = n0 - n1 + n2
    return torch.tensor([chi, n0, n1, n2], dtype=torch.float32)


class CountMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
