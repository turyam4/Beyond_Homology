import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from baselines.count_mlp.model import CountMLP


def distance_histogram_features(
    data, high_rank: int = 2, max_dist: int = 10
) -> torch.Tensor:
    """Histogram (length max_dist + 2) of clamped node->cell shortest-path
    distances over all (node, high-rank cell) pairs of one sample."""
    n0 = int(data.x_0.size(0))
    incidence = getattr(data, f"edge_index_incidence_0_{high_rank}")
    n_high = int(getattr(data, f"x_{high_rank}").size(0))
    #n_high = int(incidence[1].max().item()) + 1 if incidence.numel() > 0 else 0

    hist = torch.zeros(max_dist + 2, dtype=torch.float32)
    if n_high == 0:
        return hist

    edge_index = data.edge_index_adjacency_0_1.cpu().numpy()
    adj = csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(n0, n0),
    )
    spd = shortest_path(adj, method="D", unweighted=True)  # [n0, n0], inf if unreachable

    inc_low = incidence[0].cpu().numpy()
    inc_high = incidence[1].cpu().numpy()
    for c in range(n_high):
        cell_vertices = inc_low[inc_high == c]
        d = spd[:, cell_vertices].min(axis=1)  # [n0] node->cell distances
        unreachable = np.isinf(d)
        d = np.clip(d, 0, max_dist)
        d[unreachable] = max_dist + 1  # upstream's >1000 bucket
        hist += torch.bincount(
            torch.from_numpy(d.astype(np.int64)), minlength=max_dist + 2
        ).float()
    return hist


class DistanceHistogramMLP(CountMLP):
    """Same architecture as the count baseline; only the feature differs."""

    def __init__(self, max_dist: int = 10, hidden_dim: int = 32):
        super().__init__(input_dim=max_dist + 2, hidden_dim=hidden_dim)
