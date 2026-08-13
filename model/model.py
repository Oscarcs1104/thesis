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


class MultimodalModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        graph_backbone: str = "gatv2",
        language_backbone: str = "huggingface",
        num_layers: int = 3,
        dropout: float = 0.3,
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        use_language: bool = True,
        language_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        freeze_language_backbone: bool = True,
        trust_remote_code: bool = False,
        use_decoder: bool = False,
        decoder_vocab_size: int | None = None,
        decoder_pad_idx: int = 0,
        decoder_start_idx: int = 1,
        decoder_end_idx: int = 2,
    ) -> None:
        super().__init__()

        graph_backbone = graph_backbone.lower()
        if graph_backbone not in VALID_GRAPH_BACKBONES:
            raise ValueError(f"graph_backbone must be one of {sorted(VALID_GRAPH_BACKBONES)}")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_backbone = graph_backbone
        self.language_backbone = language_backbone.lower()
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
            trust_remote_code=trust_remote_code,
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

        fused_dim = hidden_dim * (2 if self.use_language else 1)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, output_dim),
        )

        self.decoder_condition_proj = nn.Sequential(
            nn.Linear(fused_dim + max(1, output_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _get_states(self, data: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        # Read graph and language inputs once.
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        smiles = getattr(data, "smiles", None)

        batch_size = int(batch.max().item()) + 1
        _, layer_graph_states = self.graph_encoder(x, edge_index, batch)
        lang_state = self.language_encoder(smiles, batch_size=batch_size, device=x.device)
        return layer_graph_states[-1], lang_state

    def encode(self, data: torch.nn.Module) -> torch.Tensor:
        graph_state, lang_state = self._get_states(data)
        return torch.cat([graph_state, lang_state], dim=-1) if self.use_language else graph_state

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
        fused_feat = self.encode(data)

        logits = self.head(fused_feat)
        decoder_latent = self._build_decoder_latent(fused_feat, property_values=property_values)
        decoder_logits = self.decoder(decoder_latent, decoder_input_ids) if self.decoder is not None and decoder_input_ids is not None else None

        if not return_aux:
            return (logits, decoder_logits) if decoder_logits is not None else logits

        result = {"fused": fused_feat, "logits": logits}
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        node_encoding=args.node_encoding,
        node_vocab_sizes=args.node_vocab_sizes,
        use_language=args.use_language,
        language_model_name=getattr(args, "language_model_name", "DeepChem/ChemBERTa-77M-MLM"),
        freeze_language_backbone=getattr(args, "freeze_language_backbone", True),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        use_decoder=getattr(args, "use_decoder", False),
        decoder_vocab_size=getattr(args, "decoder_vocab_size", None),
        decoder_pad_idx=getattr(args, "decoder_pad_idx", 0),
        decoder_start_idx=getattr(args, "decoder_start_idx", 1),
        decoder_end_idx=getattr(args, "decoder_end_idx", 2),
    )
