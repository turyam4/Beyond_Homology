from typing import List

import torch.nn as nn

from baselines.smcn.embeddings import TwoCellEmbedding, ZeroEmbedding
from baselines.smcn.layers.homp import AdjacencyConv, IncidenceConv, PointwiseConv
from baselines.smcn.layers.merge_node import MergeNode
from baselines.smcn.tensor_diagram import TensorDiagram


def get_feature_embed_layer(embedding_dim: int = 64):
    return nn.ModuleDict(
        {
            "x_0": ZeroEmbedding(num_embeddings=1, embedding_dim=embedding_dim, rank=0),
            "x_1": ZeroEmbedding(num_embeddings=1, embedding_dim=embedding_dim, rank=1),
        }
    )


def get_two_cell_embed_layer(embedding_dim: int = 64):
    return nn.ModuleDict(
        {"x_2": TwoCellEmbedding(embedding_dim=embedding_dim, learned=False)}
    )


def get_homp_layer(
    embed_dim: int = 64,
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    activation: str = "relu",
    include_rank2_updates: bool = False,
):
    layer = {
        "x_0": MergeNode(
            [
                AdjacencyConv(
                    input_rank=0,
                    bridge_rank=1,
                    embedding_dim=embed_dim,
                    conv_type="gin",
                    number_of_mlp_layers=number_of_mlp_layers,
                    dropout=dropout,
                    train_eps=False,
                    activation=activation,
                ),
                PointwiseConv(
                    rank=0,
                    embedding_dim=embed_dim,
                    number_of_mlp_layers=number_of_mlp_layers,
                    activation=activation,
                ),
            ],
            aggregation="concatenate",
            embedding_dim=embed_dim,
            activation=activation,
        ),
        "x_1": MergeNode(
            [
                IncidenceConv(
                    input_rank=0,
                    output_rank=1,
                    embedding_dim=embed_dim,
                    conv_type="gin",
                    dropout=dropout,
                    train_eps=False,
                    activation=activation,
                ),
                AdjacencyConv(
                    input_rank=1,
                    bridge_rank=2,
                    embedding_dim=embed_dim,
                    conv_type="gin",
                    number_of_mlp_layers=number_of_mlp_layers,
                    dropout=dropout,
                    train_eps=False,
                    activation=activation,
                ),
            ],
            aggregation="concatenate",
            embedding_dim=embed_dim,
            activation=activation,
        ),
    }
    if include_rank2_updates:
        layer["x_2"] = IncidenceConv(
            input_rank=1,
            output_rank=2,
            embedding_dim=embed_dim,
            conv_type="gin",
            dropout=dropout,
            train_eps=False,
            activation=activation,
        )
    return nn.ModuleDict(layer)


def build_homp_model(
    number_of_layers: int = 4,
    embedding_dim: int = 64,
    device: str = "cpu",
    num_classes: int = 2,
    output_ranks: List[int] = [0, 2],
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    final_dropout: float = 0.0,
    in_dropout: float = 0.0,
    activation: str = "relu",
    include_rank2_updates: bool = False,
) -> TensorDiagram:
    layers = nn.ModuleList(
        [
            get_feature_embed_layer(embedding_dim=embedding_dim),
            get_two_cell_embed_layer(embedding_dim=embedding_dim),
        ]
    )
    for _ in range(number_of_layers):
        layers.append(
            get_homp_layer(
                embed_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                dropout=dropout,
                activation=activation,
                include_rank2_updates=include_rank2_updates,
            )
        )

    dropout_list = [in_dropout] * 2 + (len(layers) - 2) * [dropout]
    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=num_classes,
        output_ranks=output_ranks,
        device=device,
        dropout_list=dropout_list,
        aggregation="mean",
        zinc_head=False,
        final_dropout=final_dropout,
        activation=activation,
    ).to(device)
