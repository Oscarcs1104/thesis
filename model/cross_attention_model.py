"""Experimental fusion variant: cross-attention between the graph and language
branches, instead of the main model's pool-then-concatenate fusion.

Both branches keep their full per-token / per-node sequences (not pooled) and
attend to each other before pooling for the property head. Predictor-only --
no decoder/generation support, this is purely to test whether richer fusion
helps property prediction. Kept in its own file so it never touches the main
architecture in model/model.py.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

try:
    from .encoders import GraphEncoder, _load_text_tokenizer
except Exception:
    from encoders import GraphEncoder, _load_text_tokenizer


class GraphLangCrossAttentionModel(nn.Module):
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
        num_heads: int = 4,
        num_cross_layers: int = 1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.hidden_dim = hidden_dim

        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            graph_backbone=graph_backbone,
            num_layers=num_layers,
            dropout=dropout,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
        )

        self.text_tokenizer = _load_text_tokenizer(language_model_name, trust_remote_code)
        self.text_model = AutoModel.from_pretrained(language_model_name, trust_remote_code=trust_remote_code)
        text_hidden = int(getattr(self.text_model.config, "hidden_size", hidden_dim))
        self.text_proj = nn.Linear(text_hidden, hidden_dim)
        if freeze_language_backbone:
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        self.graph_to_lang_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout) for _ in range(num_cross_layers)]
        )
        self.lang_to_graph_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout) for _ in range(num_cross_layers)]
        )
        self.graph_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_layers)])
        self.lang_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_layers)])
        self.dropout = nn.Dropout(dropout)

        fused_dim = hidden_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, output_dim),
        )

    def _encode_text_tokens(self, smiles_list, device: torch.device):
        encoded = self.text_tokenizer(smiles_list, padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.text_model(**encoded)
        hidden = self.text_proj(outputs.last_hidden_state)  # (B, L, hidden_dim)
        attn_mask = encoded["attention_mask"].bool()  # True = real token
        return hidden, attn_mask

    def forward(self, data: torch.nn.Module) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        smiles = getattr(data, "smiles", None)
        raw_texts = list(smiles) if isinstance(smiles, (list, tuple)) else [smiles]
        texts = ["" if item is None else str(item) for item in raw_texts]

        node_state, _ = self.graph_encoder(x, edge_index, batch)
        graph_seq, graph_mask = to_dense_batch(node_state, batch)  # (B, max_nodes, H), True = real node
        lang_seq, lang_mask = self._encode_text_tokens(texts, x.device)  # (B, L, H), True = real token

        for graph_attn, lang_attn, graph_norm, lang_norm in zip(
            self.graph_to_lang_attn, self.lang_to_graph_attn, self.graph_norm, self.lang_norm
        ):
            graph_update, _ = graph_attn(query=graph_seq, key=lang_seq, value=lang_seq, key_padding_mask=~lang_mask)
            graph_seq = graph_norm(graph_seq + self.dropout(graph_update))
            lang_update, _ = lang_attn(query=lang_seq, key=graph_seq, value=graph_seq, key_padding_mask=~graph_mask)
            lang_seq = lang_norm(lang_seq + self.dropout(lang_update))

        graph_pooled = (graph_seq * graph_mask.unsqueeze(-1)).sum(1) / graph_mask.sum(1, keepdim=True).clamp_min(1)
        lang_pooled = (lang_seq * lang_mask.unsqueeze(-1)).sum(1) / lang_mask.sum(1, keepdim=True).clamp_min(1)

        fused = torch.cat([graph_pooled, lang_pooled], dim=-1)
        return self.head(fused)


def build_cross_attention_model_from_args(args) -> GraphLangCrossAttentionModel:
    return GraphLangCrossAttentionModel(
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
        num_heads=args.num_heads,
        num_cross_layers=args.num_cross_layers,
    )
