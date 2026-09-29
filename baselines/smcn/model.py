from __future__ import annotations

from typing import List

import torch.nn as nn
from omegaconf import DictConfig

from baselines.cin.model import build_homp_model, get_homp_layer
from baselines.smcn.embeddings import TwoCellEmbedding, ZeroEmbedding
from baselines.smcn.layers.homp import AdjacencyConv, IncidenceConv, PointwiseConv
from baselines.smcn.layers.merge_node import MergeNode
from baselines.smcn.layers.subcomplex import (
    SubComplexBinaryMarkingEmbed,
    SubComplexBroadcastLow,
    SubComplexDistanceMarkingEmbed,
    SubComplexIncidenceConv,
    SubComplexLowConv,
    SubComplexPool,
)
from baselines.smcn.tensor_diagram import TensorDiagram


def get_feature_embed_layer(embedding_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "x_0": ZeroEmbedding(num_embeddings=1, embedding_dim=embedding_dim, rank=0),
            "x_1": ZeroEmbedding(num_embeddings=1, embedding_dim=embedding_dim, rank=1),
        }
    )


def get_two_cell_embed_layer(embedding_dim: int, number_of_mlp_layers: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "x_2": TwoCellEmbedding(
                embedding_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                learned=True,
            )
        }
    )


def get_marking_embedding_layer(
    *,
    binary_marking: bool,
    embedding_dim: int,
    high_rank: int,
) -> nn.Module:
    if binary_marking:
        return SubComplexBinaryMarkingEmbed(
            low_rank=0,
            high_rank=high_rank,
            embed_dim=embedding_dim,
        )
    return SubComplexDistanceMarkingEmbed(
        low_rank=0,
        high_rank=high_rank,
        embed_dim=embedding_dim,
    )


def get_subcomplex_embed_layer(
    *,
    embedding_dim: int,
    high_rank: int,
    binary_marking: bool,
    aggregation: str = "concatenate",
    activation: str = "relu",
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexBroadcastLow(low_rank=0, high_rank=high_rank),
                    get_marking_embedding_layer(
                        binary_marking=binary_marking,
                        embedding_dim=embedding_dim,
                        high_rank=high_rank,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embedding_dim,
                activation=activation,
            )
        }
    )


def get_subcomplex_layer(
    *,
    embedding_dim: int,
    high_rank: int,
    number_of_mlp_layers: int,
    binary_marking: bool,
    dropout: float,
    residual: bool,
    activation: str,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexIncidenceConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=embedding_dim,
                        output_channels=embedding_dim,
                        hidden_channels=embedding_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                    ),
                    SubComplexLowConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=embedding_dim,
                        output_channels=embedding_dim,
                        hidden_channels=embedding_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        edge_dim=embedding_dim,
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                        conv_type="gin",
                    ),
                    get_marking_embedding_layer(
                        binary_marking=binary_marking,
                        embedding_dim=embedding_dim,
                        high_rank=high_rank,
                    ),
                ],
                aggregation="concatenate",
                embedding_dim=embedding_dim,
                add_residual=residual,
                activation=activation,
            )
        }
    )


def get_subcomplex_pooling_layer(high_rank: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "x_0": SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=True,
                aggregation="concatenate",
            ),
            f"x_{high_rank}": SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=False,
                aggregation="concatenate",
            ),
        }
    )


def build_subcomplex_model(
    *,
    embedding_dim: int,
    number_cin_layers: int,
    number_subgraph_layers: int,
    device: str,
    output_ranks: List[int],
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    in_dropout: float = 0.0,
    residual: bool = False,
    activation: str = "relu",
    high_rank: int = 1,
    binary_marking: bool = False,
    num_classes: int = 2,
) -> TensorDiagram:
    layers = nn.ModuleList(
        [
            get_feature_embed_layer(embedding_dim),
            get_two_cell_embed_layer(embedding_dim, number_of_mlp_layers),
            get_subcomplex_embed_layer(
                embedding_dim=embedding_dim,
                high_rank=high_rank,
                binary_marking=binary_marking,
                activation=activation,
            ),
        ]
    )
    residuals = [False, False, False]

    for _ in range(number_cin_layers):
        layers.append(
            get_homp_layer(
                embed_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                dropout=dropout,
                include_rank2_updates=False,
                activation=activation,
            )
        )
        residuals.append(False)

    for _ in range(number_subgraph_layers):
        layers.append(
            get_subcomplex_layer(
                embedding_dim=embedding_dim,
                high_rank=high_rank,
                number_of_mlp_layers=number_of_mlp_layers,
                binary_marking=binary_marking,
                dropout=dropout,
                residual=residual,
                activation=activation,
            )
        )
        residuals.append(residual)

    layers.append(get_subcomplex_pooling_layer(high_rank))
    residuals.append(False)

    dropout_list = [in_dropout] * 2 + [dropout] * (len(layers) - 2)
    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=num_classes,
        output_ranks=output_ranks,
        device=device,
        dropout_list=dropout_list,
        final_dropout=dropout,
        aggregation="sum",
        residuals=residuals,
        activation=activation,
        zinc_head=False,
    ).to(device)


def build_model(cfg: DictConfig) -> TensorDiagram:
    if str(cfg.model_type).lower() == "homp":
        return build_homp_model(
            number_of_layers=int(cfg.number_cin_layers_for_homp),
            embedding_dim=int(cfg.embedding_dim),
            device=str(cfg.device),
            num_classes=2,
            output_ranks=list(cfg.output_ranks),
            number_of_mlp_layers=int(cfg.number_of_mlp_layers),
            dropout=float(cfg.dropout),
            in_dropout=float(cfg.in_dropout),
            activation=str(cfg.activation),
        )

    return build_subcomplex_model(
        embedding_dim=int(cfg.embedding_dim),
        number_cin_layers=int(cfg.number_cin_layers_for_subcomplex),
        number_subgraph_layers=int(cfg.number_subgraph_layers),
        device=str(cfg.device),
        output_ranks=list(cfg.output_ranks),
        number_of_mlp_layers=int(cfg.number_of_mlp_layers),
        dropout=float(cfg.dropout),
        in_dropout=float(cfg.in_dropout),
        residual=bool(cfg.get("residual", False)),
        activation=str(cfg.activation),
        high_rank=int(cfg.high_rk),
        binary_marking=bool(cfg.get("binary_marking", False)),
        num_classes=2,
    )
