import networkx as nx
import torch

from baselines.count_mlp.model import CountMLP


def boundary_loop_features(data) -> torch.Tensor:
    """[num_loops, min_len, max_len, mean_len] of one sample's boundary."""
    num_edges = int(data.x_1.size(0))

    inc_1_2 = data.edge_index_incidence_1_2
    edge_face_counts = torch.bincount(inc_1_2[0], minlength=num_edges)
    is_boundary = edge_face_counts == 1
    boundary_edges = is_boundary.nonzero(as_tuple=False).view(-1).tolist()

    if not boundary_edges:
        return torch.zeros(4)

    coadj = data.edge_index_coadjacency_1_0
    mask = is_boundary[coadj[0]] & is_boundary[coadj[1]]

    g = nx.Graph()
    g.add_nodes_from(boundary_edges)
    g.add_edges_from(zip(coadj[0][mask].tolist(), coadj[1][mask].tolist()))

    lengths = torch.tensor(
        [len(c) for c in nx.connected_components(g)], dtype=torch.float32
    )
    return torch.tensor(
        [
            float(lengths.numel()),
            lengths.min().item(),
            lengths.max().item(),
            lengths.mean().item(),
        ]
    )


class BoundaryLoopMLP(CountMLP):
    """Same architecture as the count baseline; only the feature differs."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__(input_dim=4, hidden_dim=hidden_dim)
