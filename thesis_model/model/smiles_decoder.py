from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from common.selfies_vocab import decode_ids

# Vocabulary/tokenization (build_vocab, tokenize_molecule, encode_batch, decode_ids) used to
# live here too; they moved to common/selfies_vocab.py since crossmodal_model's generation
# decoder reuses them verbatim (see crossmodal_model/generation/decoder.py) and a shared
# utility shouldn't live inside either model's own package.


class SmilesDecoder(nn.Module):
    #Transformer decoder with a latent prefix token.

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        pad_idx: int,
        start_idx: int,
        end_idx: int,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(max_len + 1, hidden_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.prefix_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _build_inputs(self, latent: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Prefix the molecule tokens with one learned condition token.
        batch_size, seq_len = input_ids.shape
        condition_token = self.latent_norm(self.latent_proj(latent)).unsqueeze(1)
        token_states = self.token_embedding(input_ids)
        positions = torch.arange(seq_len + 1, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_types = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=input_ids.device),
                torch.ones(batch_size, seq_len, dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )
        hidden = torch.cat([condition_token, token_states], dim=1)
        hidden = hidden + self.position_embedding(positions) + self.type_embedding(token_types)
        return self.prefix_drop(hidden)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _encode(self, latent: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._build_inputs(latent, input_ids)
        mask = self._causal_mask(hidden.size(1), hidden.device)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask)
        return self.final_norm(hidden)

    def forward(self, latent: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Predict the next token at each position, conditioned on the latent prefix.
        hidden = self._encode(latent, input_ids)
        return self.output(hidden[:, 1:])

    def generate(
        self,
        latent: torch.Tensor,
        id_to_token: Dict[int, str],
        max_len: int = 64,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> str:
        # Autoregressive decoding from the conditional prefix.
        self.eval()
        generated: List[int] = []
        temperature = float(temperature)
        if temperature <= 0:
            temperature = 1.0

        with torch.no_grad():
            for _ in range(max_len):
                input_ids = torch.tensor([[self.start_idx] + generated], dtype=torch.long, device=latent.device)
                hidden = self._encode(latent, input_ids)
                logits = self.output(hidden[:, -1]) / temperature
                if sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(logits.argmax(dim=-1).item())
                if next_id == self.end_idx:
                    break
                generated.append(next_id)

        return decode_ids(generated, id_to_token)
