from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from .encoders import GraphEncoder, LanguageEncoder, VALID_GRAPH_BACKBONES
    from .smiles_decoder import SmilesDecoder
except Exception:
    from encoders import GraphEncoder, LanguageEncoder, VALID_GRAPH_BACKBONES
    from smiles_decoder import SmilesDecoder


VALID_FUSIONS = {"concat", "mola", "molprop"}


class MultimodalModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        graph_backbone: str = "gatv2",
        language_backbone: str = "molformer",
        fusion: str = "mola",
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        use_language: bool = True,
        language_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        freeze_language_backbone: bool = True,
        use_decoder: bool = False,
        decoder_vocab_size: int | None = None,
        decoder_pad_idx: int = 0,
        decoder_start_idx: int = 1,
        decoder_end_idx: int = 2,
    ) -> None:
        super().__init__()

        graph_backbone = graph_backbone.lower()
        fusion = fusion.lower()
        if graph_backbone not in VALID_GRAPH_BACKBONES:
            raise ValueError(f"graph_backbone must be one of {sorted(VALID_GRAPH_BACKBONES)}")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(VALID_FUSIONS)}")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_backbone = graph_backbone
        self.language_backbone = language_backbone.lower()
        self.fusion = fusion
        self.num_layers = num_layers
        self.use_language = use_language and self.language_backbone != "none"
        self.use_decoder = use_decoder
        self.dropout = nn.Dropout(dropout)

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
            language_backbone=self.language_backbone,
            num_layers=num_layers,
            dropout=dropout,
            use_language=self.use_language,
            language_model_name=language_model_name,
            freeze_language_backbone=freeze_language_backbone,
        )

        self.decoder = None
        if self.use_decoder:
            if decoder_vocab_size is None:
                raise ValueError("decoder_vocab_size is required when use_decoder=True")
            self.decoder = SmilesDecoder(
                hidden_dim=hidden_dim,
                vocab_size=decoder_vocab_size,
                pad_idx=decoder_pad_idx,
                start_idx=decoder_start_idx,
                end_idx=decoder_end_idx,
            )

        if fusion == "mola":
            self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
            self.layer_weights = nn.Parameter(torch.ones(num_layers * (2 if self.use_language else 1), 1, 1))
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, output_dim),
            )
        elif fusion == "concat":
            fused_dim = hidden_dim * (2 if self.use_language else 1)
            self.head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, output_dim),
            )
        else:
            self.lang_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            fused_dim = hidden_dim * (2 if self.use_language else 1)
            self.head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, output_dim),
            )

        fused_dim = hidden_dim * (2 if self.use_language else 1)
        self.decoder_condition_proj = nn.Sequential(
            nn.Linear(fused_dim + max(1, output_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _get_states(self, data: torch.nn.Module) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        # Read graph and language inputs once.
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        lang = getattr(data, "smiles", None) if self.language_backbone == "chemberta" else getattr(data, "lang", None)
        if lang is None and self.language_backbone == "chemberta":
            lang = getattr(data, "lang", None)

        batch_size = int(batch.max().item()) + 1
        _, layer_graph_states = self.graph_encoder(x, edge_index, batch)
        lang_state, layer_lang_states = self.language_encoder(lang, batch_size=batch_size, device=x.device)
        return layer_graph_states, lang_state, layer_lang_states

    def encode(self, data: torch.nn.Module) -> torch.Tensor:
        layer_graph_states, lang_state, layer_lang_states = self._get_states(data)

        if self.fusion == "concat":
            return torch.cat([layer_graph_states[-1], lang_state], dim=-1) if self.use_language else layer_graph_states[-1]

        if self.fusion == "molprop":
            if self.use_language:
                return torch.cat([layer_graph_states[-1] * self.lang_gate(lang_state), lang_state], dim=-1)
            return layer_graph_states[-1]

        tokens: List[torch.Tensor] = []
        for graph_state, lang_state_per_layer in zip(layer_graph_states, layer_lang_states):
            tokens.append(graph_state)
            if self.use_language:
                tokens.append(lang_state_per_layer)

        fused_all = torch.stack(tokens, dim=0)
        attn_out, _ = self.cross_attention(fused_all, fused_all, fused_all)
        return (attn_out * self.layer_weights).sum(dim=0)

    def _build_decoder_latent(self, fused_feat: torch.Tensor, property_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        if property_values is None:
            return fused_feat

        prop = property_values.float()
        if prop.dim() == 0:
            prop = prop.unsqueeze(0)
        if prop.dim() == 1:
            prop = prop.unsqueeze(-1)
        if prop.size(0) != fused_feat.size(0):
            if prop.numel() == 1:
                prop = prop.expand(fused_feat.size(0), -1)
            else:
                raise ValueError(f"Property values batch size {prop.size(0)} does not match fused feature batch size {fused_feat.size(0)}")
        if prop.size(-1) != self.output_dim and self.output_dim == 1:
            prop = prop.view(-1, 1)
        elif prop.size(-1) != self.output_dim:
            raise ValueError(f"Property values have {prop.size(-1)} dims but model expects {self.output_dim}")

        cond_input = torch.cat([fused_feat, prop], dim=-1)
        return self.decoder_condition_proj(cond_input)

    def forward(self, data: torch.nn.Module, decoder_input_ids: Optional[torch.Tensor] = None, return_aux: bool = False, property_values: Optional[torch.Tensor] = None):
        layer_graph_states, lang_state, layer_lang_states = self._get_states(data)

        layer_weights = None
        attn_map = None

        if self.fusion == "concat":
            fused_feat = torch.cat([layer_graph_states[-1], lang_state], dim=-1) if self.use_language else layer_graph_states[-1]
        elif self.fusion == "molprop":
            fused_feat = torch.cat([layer_graph_states[-1] * self.lang_gate(lang_state), lang_state], dim=-1) if self.use_language else layer_graph_states[-1]
        else:
            tokens: List[torch.Tensor] = []
            for graph_state, lang_state_per_layer in zip(layer_graph_states, layer_lang_states):
                tokens.append(graph_state)
                if self.use_language:
                    tokens.append(lang_state_per_layer)
            fused_all = torch.stack(tokens, dim=0)
            attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
            fused_feat = (attn_out * self.layer_weights).sum(dim=0)
            layer_weights = self.layer_weights.view(-1, 1, 1)

        logits = self.head(fused_feat)
        decoder_latent = self._build_decoder_latent(fused_feat, property_values=property_values)
        decoder_logits = self.decoder(decoder_latent, decoder_input_ids) if self.decoder is not None and decoder_input_ids is not None else None

        if not return_aux:
            return (logits, decoder_logits) if decoder_logits is not None else logits

        result = {"fused": fused_feat, "logits": logits}
        if layer_weights is not None:
            result["layer_weights"] = layer_weights
        if attn_map is not None:
            result["attention"] = attn_map
        if decoder_logits is not None:
            result["decoder_logits"] = decoder_logits
        return result

    def generate_smiles(self, data: torch.nn.Module, id_to_token: dict[int, str], max_len: int = 64, property_values: Optional[torch.Tensor] = None) -> str:
        if self.decoder is None:
            raise ValueError("Decoder is disabled")

        self.eval()
        with torch.no_grad():
            fused_feat = self.encode(data)
            decoder_latent = self._build_decoder_latent(fused_feat, property_values=property_values)
            return self.decoder.generate(decoder_latent, id_to_token, max_len=max_len)

    def generate_smiles_candidates(
        self,
        data: torch.nn.Module,
        id_to_token: dict[int, str],
        max_len: int = 64,
        num_samples: int = 10,
        temperature: float = 1.0,
        property_values: Optional[torch.Tensor] = None,
    ) -> List[str]:
        if self.decoder is None:
            raise ValueError("Decoder is disabled")

        self.eval()
        candidates: List[str] = []
        with torch.no_grad():
            fused_feat = self.encode(data)
            decoder_latent = self._build_decoder_latent(fused_feat, property_values=property_values)
            for _ in range(max(1, num_samples)):
                candidates.append(
                    self.decoder.generate(
                        decoder_latent,
                        id_to_token,
                        max_len=max_len,
                        temperature=temperature,
                        sample=True,
                    )
                )
        return candidates


def build_model_from_args(args) -> MultimodalModel:
    # Train code only passes parsed arguments; this keeps model construction in one place.
    return MultimodalModel(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        graph_backbone=args.graph_backbone,
        language_backbone=args.language_backbone,
        fusion=args.fusion,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_encoding=args.node_encoding,
        node_vocab_sizes=args.node_vocab_sizes,
        use_language=args.use_language,
        language_model_name=getattr(args, "language_model_name", "DeepChem/ChemBERTa-77M-MLM"),
        freeze_language_backbone=getattr(args, "freeze_language_backbone", True),
        use_decoder=getattr(args, "use_decoder", False),
        decoder_vocab_size=getattr(args, "decoder_vocab_size", None),
        decoder_pad_idx=getattr(args, "decoder_pad_idx", 0),
        decoder_start_idx=getattr(args, "decoder_start_idx", 1),
        decoder_end_idx=getattr(args, "decoder_end_idx", 2),
    )
