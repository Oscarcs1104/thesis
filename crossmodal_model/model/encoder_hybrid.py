"""HybridEncoder: thesis_model's graph branch (GraphEncoder -- categorical atom/bond
embeddings + GINEConv + configurable pooling) combined with MoLA's SMILES branch
(char-level embedding + per-layer TransformerEncoder) and MoLA's per-layer
cross-attention fusion.

use_graph/use_smiles: disable either branch entirely for single-modality ablations
(tokens_per_layer drops from 2 to 1). forward_with_raw() -- the generation entry point --
requires both enabled and positional_smiles=True.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis_model.model.encoders import GraphEncoder


class HybridEncoder(nn.Module):
    def __init__(
        self,
        sm_vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        positional_smiles: bool = False,
        max_sm_len: int = 100,
        graph_backbone: str = "gin",
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        edge_vocab_sizes: Optional[Sequence[int]] = None,
        graph_pooling: str = "add",
        graph_dropout: float = 0.3,
        use_graph: bool = True,
        use_smiles: bool = True,
        gin_hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        if not use_graph and not use_smiles:
            raise ValueError("HybridEncoder needs at least one of use_graph/use_smiles enabled")
        self.num_layers = num_layers
        self.use_graph = use_graph
        self.use_smiles = use_smiles
        self.tokens_per_layer = int(use_graph) + int(use_smiles)
        self.positional_smiles = positional_smiles
        self.max_sm_len = max_sm_len

        if use_graph:
            self.graph_encoder = GraphEncoder(
                hidden_dim=hidden_dim,
                graph_backbone=graph_backbone,
                num_layers=num_layers,
                dropout=graph_dropout,
                node_encoding=node_encoding,
                node_vocab_sizes=node_vocab_sizes,
                edge_vocab_sizes=edge_vocab_sizes,
                pool=graph_pooling,
                gin_hidden_mult=gin_hidden_mult,
            )

        if use_smiles:
            if positional_smiles:
                self.position_embedding = nn.Embedding(max_sm_len, hidden_dim)

            def _sm_transformer():
                return nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=positional_smiles),
                    num_layers=1,
                )

            self.sm_embedding = nn.Embedding(sm_vocab_size, hidden_dim, padding_idx=0)
            self.sm_layers = nn.ModuleList([_sm_transformer() for _ in range(num_layers)])
            self.sm_dropout = nn.Dropout(0.3)

    def _pool_sm(self, sm_x: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
        if keep is None:
            return sm_x.mean(dim=1)
        return (sm_x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1e-6)

    def forward(self, x, edge_index, edge_attr, sm, batch):
        fused_all, _ = self._forward_impl(x, edge_index, edge_attr, sm, batch, need_raw=False)
        return fused_all

    def forward_with_raw(self, x, edge_index, edge_attr, sm, batch):
        """Generation entry point -- see crossmodal_model/model/encoder.py's Encoder for
        the non-hybrid equivalent.

        Single-modality is allowed here on purpose: comparing generation quality with the
        graph branch off against the SMILES branch off IS the thesis's ablation, and
        refusing to build that memory made the experiment impossible to run. The absent
        stream comes back as None in `raw`, and build_memory concatenates whatever is
        present. positional_smiles is still required whenever the SMILES branch is on --
        a permutation-invariant, pad-diluted character encoder cannot support generation.
        """
        if self.use_smiles and not self.positional_smiles:
            raise ValueError("forward_with_raw needs positional_smiles=True when use_smiles=True")
        fused_all, raw = self._forward_impl(x, edge_index, edge_attr, sm, batch, need_raw=True)
        return fused_all, raw

    def _forward_impl(self, x, edge_index, edge_attr, sm, batch, need_raw: bool = False):
        final_node_state = None
        layer_graph_states = None
        if self.use_graph:
            # thesis_model's GraphEncoder handles its own per-layer embed/conv/relu/
            # dropout/pool internally -- one call gets every layer's state.
            final_node_state, layer_graph_states = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)

        pad_mask = keep = h_sm = None
        if self.use_smiles:
            sm_embed = self.sm_embedding(sm)
            if self.positional_smiles:
                seq_len = sm.size(1)
                positions = torch.arange(seq_len, device=sm.device).unsqueeze(0).expand(sm.size(0), -1)
                sm_embed = sm_embed + self.position_embedding(positions)
                pad_mask = sm == 0
                keep = (~pad_mask).unsqueeze(-1).to(sm_embed.dtype)
            h_sm = sm_embed

        fused_out_list = []
        for layer_idx in range(self.num_layers):
            tokens = []
            if self.use_graph:
                tokens.append(layer_graph_states[layer_idx])
            if self.use_smiles:
                h_sm = self.sm_layers[layer_idx](h_sm, src_key_padding_mask=pad_mask)
                h_sm = F.relu(h_sm)
                sm_feat = self._pool_sm(h_sm, keep)
                h_sm = self.sm_dropout(h_sm)
                tokens.append(sm_feat)
            fused_out_list.append(torch.stack(tokens, dim=0))

        fused_all = torch.cat(fused_out_list, dim=0)

        raw = None
        if need_raw:
            raw = {"node_state": final_node_state, "node_batch": batch, "sm_state": h_sm, "sm_pad_mask": pad_mask}
        return fused_all, raw
