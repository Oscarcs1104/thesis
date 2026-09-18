"""Conditional molecule generation for MoLA, conditioned Chemformer-style: the decoder
cross-attends over the FULL sequence of the shared encoder's states (graph nodes +
SMILES characters), not a single pooled vector -- see crossmodal_model/model/encoder.py's
Encoder.forward_with_raw / crossmodal_model/model/mola.py's MoLA.encode_for_generation,
and molbart/models/chemformer.py's Chemformer.decode() for the reference this mirrors.

Design (per the design discussion this implements):
  - Shared encoder: reuses the SAME MoLA(positional_smiles=True) graph_conv/sm_transformer
    weights as the regression path. Positional embeddings + padding mask are mandatory
    here (unlike the standalone A2 regression experiment) -- you cannot generate a
    coherent sequence from a permutation-invariant, pad-diluted SMILES encoder.
  - Memory: to_dense_batch(node_state) [B, N_max, H] (mask = real atoms) concatenated with
    the SMILES per-character states [B, L, H] (mask = real characters) along the sequence
    dim, optionally prefixed with one learned "property token" (a projected scalar) --
    the continuous-property analogue of Chemformer's own text property-token prefix
    (MolecularOptimizationDataModule: `input_smiles = prop_tokens + smi`).
  - Decoder: plain nn.TransformerDecoder (self-attention causal mask + cross-attention
    over memory + memory_key_padding_mask), exactly the mechanism BARTModel.decode uses.
  - Vocabulary/tokenization: reused verbatim from common/selfies_vocab.py (shared with
    thesis_model's own SmilesDecoder, SELFIES-based -- guarantees a generated sequence
    decodes to a valid molecule by construction, since selfies.decoder() cannot produce
    an invalid SMILES from a well-formed SELFIES string).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from common.selfies_vocab import (  # noqa: E402  -- reused verbatim, see module docstring
    decode_ids,
    encode_batch,
)
from common.selfies_vocab import build_vocab as build_selfies_vocab  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402


def build_memory(
    raw: Dict[str, torch.Tensor],
    hidden_dim: int,
    property_values: Optional[torch.Tensor] = None,
    property_proj: Optional[nn.Module] = None,
    condition: Optional[torch.Tensor] = None,
):
    """Turn Encoder.forward_with_raw's output into one combined [B, S, H] memory sequence
    + a matching [B, S] bool padding mask (True = ignore), the convention
    nn.TransformerDecoder's memory_key_padding_mask expects.

    Either stream may be absent (raw[...] is None) -- that is the single-modality
    ablation, see HybridEncoder.forward_with_raw. Whatever is present gets concatenated.

    Two conditioning paths, deliberately kept separate:
      property_values + property_proj : one projected scalar, the original Chemformer-style
          token. Retained so ablate_prop_token.py can still diagnose the old models.
      condition : [B, P, H] precomputed condition tokens (see pair_decoder.py's
          ConditionEmbedding), prepended in order. This is the path the pair-conditioned
          generator uses.
    """
    node_state, node_batch, sm_state, sm_pad_mask = (
        raw["node_state"], raw["node_batch"], raw["sm_state"], raw["sm_pad_mask"]
    )

    parts, pads = [], []
    # The fusion tokens go first, ahead of the per-atom and per-character detail. They
    # are [B, 2L, H] and never padded: every molecule has exactly one token per layer per
    # active modality, which is what makes them a summary rather than a sequence.
    fused_state = raw.get("fused_state")
    if fused_state is not None:
        parts.append(fused_state)
        pads.append(torch.zeros(fused_state.shape[:2], dtype=torch.bool,
                                device=fused_state.device))
    if node_state is not None:
        node_dense, node_real_mask = to_dense_batch(node_state, node_batch)  # [B,N,H], [B,N] True=real
        parts.append(node_dense)
        pads.append(~node_real_mask)
    if sm_state is not None:
        parts.append(sm_state)
        pads.append(sm_pad_mask)
    if not parts:
        raise ValueError("build_memory got neither graph nor SMILES states -- the encoder "
                         "produced nothing to condition on")

    memory = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    memory_pad_mask = torch.cat(pads, dim=1) if len(pads) > 1 else pads[0]

    if property_values is not None and condition is not None:
        raise ValueError("pass either property_values or condition, not both")

    if property_values is not None:
        if property_proj is None:
            raise ValueError("property_values given but no property_proj module")
        prop = property_values.to(memory.dtype).view(-1, 1)              # [B,1]
        condition = property_proj(prop).unsqueeze(1)                     # [B,1,H]

    if condition is not None:
        condition = condition.to(memory.dtype)
        memory = torch.cat([condition, memory], dim=1)
        cond_pad = torch.zeros(memory.size(0), condition.size(1), dtype=torch.bool, device=memory.device)
        memory_pad_mask = torch.cat([cond_pad, memory_pad_mask], dim=1)

    return memory, memory_pad_mask


class MoLAGenerativeDecoder(nn.Module):
    """Causal self-attention + cross-attention over `build_memory`'s output -- the same
    mechanism as molbart.models.transformer_models.BARTModel.decode, adapted to MoLA's
    dual-stream (graph+SMILES) memory instead of a single BART encoder memory."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        pad_idx: int,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, memory: torch.Tensor, memory_pad_mask: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = decoder_input_ids.size(1)
        positions = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0).expand(decoder_input_ids.size(0), -1)
        tgt = self.dropout(self.token_embedding(decoder_input_ids) + self.position_embedding(positions))
        tgt_mask = self._causal_mask(seq_len, decoder_input_ids.device)
        tgt_pad_mask = decoder_input_ids == self.pad_idx
        hidden = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.output_proj(hidden)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, memory_pad_mask: torch.Tensor, vocab: dict, max_len: int = 96, temperature: float = 1.0, sample: bool = False) -> str:
        self.eval()
        start_idx, end_idx = vocab["start_idx"], vocab["end_idx"]
        generated: List[int] = [start_idx]
        for _ in range(max_len):
            ids = torch.tensor([generated], dtype=torch.long, device=memory.device)
            logits = self.forward(memory, memory_pad_mask, ids)[:, -1] / max(temperature, 1e-6)
            if sample:
                next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            else:
                next_id = int(logits.argmax(dim=-1).item())
            if next_id == end_idx:
                break
            generated.append(next_id)
        return decode_ids(generated[1:], vocab["id_to_token"])


class MoLAConditionalGenerator(nn.Module):
    """MoLA encoder (shared, positional_smiles=True) + MoLAGenerativeDecoder, wired
    through build_memory. Property-conditioned by default (property_proj present);
    pass use_property=False for an unconditional autoencoder-style variant."""

    def __init__(self, mola: MoLA, vocab_size: int, hidden_dim: int, pad_idx: int, use_property: bool = True, decoder_layers: int = 4, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if not mola.encoder.positional_smiles:
            raise ValueError("MoLAConditionalGenerator needs mola built with positional_smiles=True")
        self.mola = mola
        self.use_property = use_property
        self.property_proj = (
            nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)) if use_property else None
        )
        self.decoder = MoLAGenerativeDecoder(vocab_size, hidden_dim, pad_idx, num_layers=decoder_layers, max_len=max_len, dropout=dropout)
        self.hidden_dim = hidden_dim

    def _memory(self, data, property_values: Optional[torch.Tensor]):
        raw = self.mola.encode_for_generation(data)
        prop = property_values if self.use_property else None
        return build_memory(raw, self.hidden_dim, property_values=prop, property_proj=self.property_proj)

    def forward(self, data, decoder_input_ids: torch.Tensor, property_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        memory, memory_pad_mask = self._memory(data, property_values)
        return self.decoder(memory, memory_pad_mask, decoder_input_ids)

    @torch.no_grad()
    def generate(self, data, vocab: dict, property_values: Optional[torch.Tensor] = None, max_len: int = 96, temperature: float = 1.0, sample: bool = False) -> str:
        self.eval()
        memory, memory_pad_mask = self._memory(data, property_values)
        return self.decoder.generate(memory, memory_pad_mask, vocab, max_len=max_len, temperature=temperature, sample=sample)


__all__ = ["MoLAConditionalGenerator", "MoLAGenerativeDecoder", "build_memory", "build_selfies_vocab", "encode_batch", "decode_ids"]
