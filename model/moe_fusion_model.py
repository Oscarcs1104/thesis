"""Experimental fusion variant: Multi-gate Mixture-of-Experts (MMoE, Ma et al. 2018)
over the pooled graph+language representation, instead of the main model's plain
concat-then-MLP fusion.

    f^(k)(x) = sum_i g_i^(k)(x) * f_i(x),   g^(k)(x) = softmax(W_g^(k) x)
    y_k = h^(k)(f^(k)(x))

A shared pool of experts transforms the fused (graph, language) vector; one gate
per task produces a softmax weighting over the experts, and one tower per task
turns the resulting mixture into that task's output. With num_tasks=1 (the only
case used here -- property prediction), this is a single-gate MoE; the multi-task
plumbing is kept so more towers/gates can be added later without touching the
shared experts. Predictor-only, no decoder -- kept in its own file so it never
touches the main architecture in model/model.py.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from .encoders import GraphEncoder, LanguageEncoder
except Exception:
    from encoders import GraphEncoder, LanguageEncoder


class GraphLangMoEModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        graph_backbone: str = "gin",
        num_layers: int = 3,
        dropout: float = 0.3,
        node_encoding: str = "dense",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        language_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        freeze_language_backbone: bool = True,
        trust_remote_code: bool = False,
        num_experts: int = 4,
        expert_hidden_dim: Optional[int] = None,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        expert_hidden_dim = expert_hidden_dim or hidden_dim

        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            graph_backbone=graph_backbone,
            num_layers=num_layers,
            dropout=dropout,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
        )
        self.language_encoder = LanguageEncoder(
            hidden_dim=hidden_dim,
            language_backbone="huggingface",
            num_layers=num_layers,
            dropout=dropout,
            use_language=True,
            language_model_name=language_model_name,
            freeze_language_backbone=freeze_language_backbone,
            trust_remote_code=trust_remote_code,
        )

        fused_dim = hidden_dim * 2
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, expert_hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(fused_dim, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_hidden_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, output_dim),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, data: torch.nn.Module) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        smiles = getattr(data, "smiles", None)
        batch_size = int(batch.max().item()) + 1

        _, layer_graph_states = self.graph_encoder(x, edge_index, batch)
        graph_state = layer_graph_states[-1]
        lang_state = self.language_encoder(smiles, batch_size=batch_size, device=x.device)
        fused = torch.cat([graph_state, lang_state], dim=-1)

        expert_outputs = torch.stack([expert(fused) for expert in self.experts], dim=1)  # (B, num_experts, expert_hidden_dim)

        task_outputs: List[torch.Tensor] = []
        for gate, tower in zip(self.gates, self.towers):
            gate_weights = torch.softmax(gate(fused), dim=-1)  # (B, num_experts)
            mixture = torch.einsum("be,beh->bh", gate_weights, expert_outputs)
            task_outputs.append(tower(mixture))

        return task_outputs[0] if self.num_tasks == 1 else task_outputs


def build_moe_model_from_args(args) -> GraphLangMoEModel:
    return GraphLangMoEModel(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        graph_backbone=args.graph_backbone,
        num_layers=args.num_layers,
        dropout=args.dropout,
        node_encoding=args.node_encoding,
        node_vocab_sizes=args.node_vocab_sizes,
        language_model_name=args.language_model_name,
        freeze_language_backbone=args.freeze_language_backbone,
        trust_remote_code=args.trust_remote_code,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        num_tasks=1,
    )
