"""HybridMoLA variant: thesis_model's graph branch (unchanged) + a PRETRAINED SMILES
branch (ChemBERTa, frozen by default -- see pretrained_smiles_branch.py) instead of
MoLA's from-scratch char-level branch. Same cross-layer fusion as HybridMoLA (unchanged).

No char vocabulary needed here at all -- the SMILES branch tokenizes raw text itself via
its own pretrained tokenizer, so data prep is just smiles_to_data() (which already keeps
.smiles), no featurize_hybrid.prepare_hybrid_data/.sm tensor required.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thesis_model.model.encoders import GraphEncoder
from crossmodal_model.model.pretrained_smiles_branch import PretrainedSmilesBranch


class HybridMoLAPretrainedSMILES(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        graph_backbone: str = "gin",
        graph_pooling: str = "add",
        graph_dropout: float = 0.3,
        smiles_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        freeze_smiles_backbone: bool = True,
        smiles_trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim, graph_backbone=graph_backbone, num_layers=num_layers,
            dropout=graph_dropout, node_encoding="categorical", pool=graph_pooling,
        )
        self.smiles_branch = PretrainedSmilesBranch(
            hidden_dim=hidden_dim, num_layers=num_layers, model_name=smiles_model_name,
            freeze_backbone=freeze_smiles_backbone, trust_remote_code=smiles_trust_remote_code,
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.out_layer_final = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4 * hidden_dim, output_dim),
        )
        self.layer_weights = nn.Parameter(torch.ones(num_layers * 2, 1, 1))

    def forward(self, data):
        _, layer_graph_states = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        sm_states = self.smiles_branch(data.smiles, device=data.x.device)

        fused_out_list = [torch.stack([layer_graph_states[i], sm_states[i]], dim=0) for i in range(len(layer_graph_states))]
        fused_all = torch.cat(fused_out_list, dim=0)

        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)
        fused_out_final = self.out_layer_final(fused_feat)
        return [fused_feat, self.layer_weights.view(-1, 1, 1), attn_map, fused_out_final]
