"""
spec_mlp_model.py -- SPEC_MLP: a spectral-only probe for channel C6
====================================================================
A baseline that sees NOTHING except spectral invariants of the complex. It is
the empirical counterpart to the C6 gate: the gate asks "are the spectra
equal?", this asks "if they are not equal, is the difference exploitable by a
model that reads only spectra?"

WHY THIS PROBE EXISTS
---------------------
The C6 gate FAILED on the reported genus-2 pairs: eigenvalue multisets differ
on 16/16 pairs for every operator tested, at 1e-3..1e-2 relative magnitude.
Because the gate failed, the "an exact gate subsumes the probe" argument is
unavailable, and the only remaining statement about C6 is an empirical one
about exploitability. That is what SPEC_MLP measures.

HONEST EXPECTATION (state this before you look at the result)
-------------------------------------------------------------
Spectra are permutation invariant, so every permuted copy of one complex maps
to an identical feature vector: within-class scatter is exactly zero. Since
the two classes have measurably DIFFERENT spectra, SPEC_MLP is EXPECTED TO
SEPARATE, i.e. to score high, not at chance. A high score confirms C6 is not
merely open but exploitable. A near-zero score would mean the differences,
though real, are not linearly/MLP-accessible from these summaries.
Either way it is a measurement, not a defence.

WHAT IT READS
-------------
Only `edge_index_adjacency_0_1` (1-skeleton), `edge_index_incidence_0_1` and
`edge_index_incidence_1_2` (UNSIGNED incidence -- the pipeline stores no
signs, so this is exactly what Phi supplies). No coordinates, no cell
identities, no attaching word.

Operators (configurable):
    A        1-skeleton adjacency
    L        combinatorial Laplacian D - A
    Lnorm    normalised Laplacian I - D^-1/2 A D^-1/2
    hodge1   unsigned Hodge L1 = B1^T B1 + B2 B2^T

Per operator the feature vector holds, all permutation invariant:
    * top-k and bottom-k eigenvalues (zero-padded when n < k)
    * normalised spectral moments  sum_i lambda_i^l / n,  l = 1..L
      (tr(M^l)/n, i.e. closed-walk densities -- the quantity a depth-l
       message-passing receptive field can aggregate)
    * mean / std / min / max of the spectrum
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEFAULT_OPERATORS = ("A", "L", "Lnorm", "hodge1")


def _dense_from_edge_index(edge_index, n: int) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.float64)
    if edge_index is None or edge_index.numel() == 0:
        return M
    ei = edge_index.detach().cpu().numpy()
    M[ei[0], ei[1]] = 1.0
    M[ei[1], ei[0]] = 1.0
    return M


def _incidence_dense(edge_index, n_low: int, n_high: int) -> np.ndarray:
    B = np.zeros((n_low, n_high), dtype=np.float64)
    if edge_index is None or edge_index.numel() == 0:
        return B
    ei = edge_index.detach().cpu().numpy()
    B[ei[0], ei[1]] = 1.0          # UNSIGNED: the pipeline stores no signs
    return B


def _spectrum_block(eigs: np.ndarray, k: int, moment_L: int) -> list:
    eigs = np.sort(np.asarray(eigs, dtype=np.float64))
    n = max(len(eigs), 1)
    top = np.zeros(k)
    bot = np.zeros(k)
    take = min(k, len(eigs))
    if take:
        top[:take] = eigs[::-1][:take]
        bot[:take] = eigs[:take]
    # normalised moments: tr(M^l)/n  == sum lambda^l / n
    moments = []
    for l in range(1, moment_L + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            m = float(np.sum(np.power(eigs, l)) / n)
        moments.append(0.0 if not np.isfinite(m) else m)
    stats = [float(eigs.mean()) if len(eigs) else 0.0,
             float(eigs.std()) if len(eigs) else 0.0,
             float(eigs.min()) if len(eigs) else 0.0,
             float(eigs.max()) if len(eigs) else 0.0]
    return list(top) + list(bot) + moments + stats


def spectral_feature_dim(k: int = 16, moment_L: int = 8,
                         operators=DEFAULT_OPERATORS) -> int:
    return len(operators) * (2 * k + moment_L + 4)


def spectral_features(data, k: int = 16, moment_L: int = 8,
                      operators=DEFAULT_OPERATORS) -> torch.Tensor:
    """Spectral-only feature vector for ONE complex (a single PyG Data)."""
    n0 = int(data.x_0.size(0))
    n1 = int(data.x_1.size(0))
    n2 = int(data.x_2.size(0)) if getattr(data, "x_2", None) is not None else 0

    feats: list = []
    A = None
    for op in operators:
        if op in ("A", "L", "Lnorm"):
            if A is None:
                A = _dense_from_edge_index(
                    getattr(data, "edge_index_adjacency_0_1", None), n0)
            d = A.sum(1)
            if op == "A":
                M = A
            elif op == "L":
                M = np.diag(d) - A
            else:
                dinv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
                M = np.eye(n0) - (dinv[:, None] * A * dinv[None, :])
        elif op == "hodge1":
            B1 = _incidence_dense(
                getattr(data, "edge_index_incidence_0_1", None), n0, n1)
            B2 = _incidence_dense(
                getattr(data, "edge_index_incidence_1_2", None), n1, n2)
            M = B1.T @ B1 + B2 @ B2.T
        else:
            raise ValueError(f"unknown spectral operator {op!r}")

        if M.shape[0] == 0:
            feats.extend([0.0] * (2 * k + moment_L + 4))
            continue
        eigs = np.linalg.eigvalsh((M + M.T) / 2.0)
        feats.extend(_spectrum_block(eigs, k, moment_L))

    out = torch.tensor(feats, dtype=torch.float32)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class SpectralMLP(nn.Module):
    """Same tiny architecture as CountMLP / DistanceHistogramMLP so the probe
    is comparable to the other scalar-feature baselines in the ladder."""

    def __init__(self, k: int = 16, moment_L: int = 8,
                 operators=DEFAULT_OPERATORS, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        input_dim = spectral_feature_dim(k, moment_L, operators)
        layers: list = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        # standardise features online: raw eigenvalues and high moments differ
        # by many orders of magnitude, which otherwise dominates the fit
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # sign-preserving log compression keeps large moments finite
        x = torch.sign(x) * torch.log1p(x.abs())
        return self.net(x)
