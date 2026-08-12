from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import selfies as _selfies
except Exception:
    _selfies = None


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<OTHER>"


def _to_tokens(text: str) -> List[str]:
    """Split one molecule string into simple tokens."""
    # Prefer SELFIES tokens when possible.
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if _selfies is not None:
        try:
            selfies_text = _selfies.encoder(text, strict=False)
            return list(_selfies.split_selfies(selfies_text))
        except Exception:
            pass

    return list(text)


def build_vocab(texts: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Build a tiny token vocabulary from a list of molecules."""
    # Reserve the special tokens first.
    tokens = {PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN}
    for text in texts:
        tokens.update(_to_tokens(text))

    ordered_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN] + sorted(t for t in tokens if t not in {PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN})
    token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "pad_idx": token_to_id[PAD_TOKEN],
        "start_idx": token_to_id[START_TOKEN],
        "end_idx": token_to_id[END_TOKEN],
        "unk_idx": token_to_id[UNK_TOKEN],
    }


def encode_batch(texts: Sequence[str], vocab: Dict[str, Dict[str, int]], max_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create teacher-forcing inputs and targets for the decoder."""
    # Decoder input is shifted by one token from the target.
    token_to_id = vocab["token_to_id"]
    pad_idx = vocab["pad_idx"]
    start_idx = vocab["start_idx"]
    end_idx = vocab["end_idx"]
    unk_idx = vocab["unk_idx"]

    inputs: List[List[int]] = []
    targets: List[List[int]] = []

    for text in texts:
        token_ids = [start_idx]
        for token in _to_tokens(text):
            token_ids.append(token_to_id.get(token, unk_idx))
        token_ids.append(end_idx)

        token_ids = token_ids[:max_len]
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        if len(input_ids) < max_len - 1:
            pad_amount = (max_len - 1) - len(input_ids)
            input_ids = input_ids + [pad_idx] * pad_amount
            target_ids = target_ids + [pad_idx] * pad_amount

        inputs.append(input_ids)
        targets.append(target_ids)

    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def decode_ids(token_ids: Sequence[int], id_to_token: Dict[int, str]) -> str:
    """Turn predicted token ids back into a molecule string."""
    tokens: List[str] = []
    for token_id in token_ids:
        token = id_to_token.get(int(token_id), UNK_TOKEN)
        if token in {PAD_TOKEN, START_TOKEN}:
            continue
        if token == END_TOKEN:
            break
        tokens.append(token)

    if _selfies is not None:
        try:
            selfies_text = "".join(tokens)
            decoded = _selfies.decoder(selfies_text)
            if decoded:
                return decoded
        except Exception:
            pass

    return "".join(tokens)


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