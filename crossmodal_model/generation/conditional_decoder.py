"""Conditional molecule generation: encoder sees a generic framework, decoder writes a molecule.

    encoder input  : generic framework   the Murcko scaffold's topology, with atom types
                                         and bond orders erased
    condition      : logP = 2.5          an ABSOLUTE target, binned
    decoder target : the complete molecule

The previous design trained on (M -> M, y = f(M)): the target was inside the decoder's
own memory, so reproducing it never required reading the conditioning token and the
gradient had no reason to teach the decoder to use it. Here the framework fixes only the
core topology; the decoder must decide which positions are N, O or S, where the double
bonds go, and what hangs off the core -- and heteroatoms are exactly what determines
logP and TPSA.

Measured on the full MOSES corpus (data_pipeline/report_corpus.py):
  - 70,885 distinct frameworks for 1.94M molecules, 27.3 per bucket, and 92.7% of
    molecules sit in a bucket of 10 or more.
  - 82% of the total logP variance survives INSIDE a framework bucket (against 55.9%
    for the Murcko scaffold, which already fixes 77% of a molecule's atoms).
So the framework leaves the property undecided and the conditioning token is what
decides it. That is the property the original design lacked.

Conditioning mechanism: one prefix token per property, each a learned embedding of a
quantile bin (common/property_bins.py). Bins rather than a raw scalar through
Linear(1 -> H), which is a weak low-frequency signal and was previously fed
unstandardized. A prepended token rather than FiLM because the old token was ignored for
being *redundant*, not for being a token, and that redundancy is now gone; if it is still
ignored, the diagnostics below say so and FiLM is the next move.

Diagnostics built in:
  - condition dropout to each property's null bin, independently, so sampling can ask for
    "logP = 2.5, don't care about the rest" and classifier-free guidance works.
  - guided_logits(): if raising the guidance weight changes nothing, the condition is not
    being used. Costs one extra forward pass and no retraining.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from crossmodal_model.generation.decoder import MoLAGenerativeDecoder, build_memory


class ConditionEmbedding(nn.Module):
    """One learned embedding table per conditioned property: [B, P] bins -> [B, P, H].

    Each property owns its null bin (the last row of its table) and is dropped
    independently during training. Dropping the whole prefix jointly would only ever
    permit all-or-nothing conditioning; independent dropout is what lets a request name
    one property and leave the others unspecified.
    """

    def __init__(self, vocab_sizes: Sequence[int], null_bins: Sequence[int],
                 hidden_dim: int, dropout_prob: float = 0.15) -> None:
        super().__init__()
        if len(vocab_sizes) != len(null_bins):
            raise ValueError("vocab_sizes and null_bins must have the same length")
        self.n_properties = len(vocab_sizes)
        self.dropout_prob = float(dropout_prob)
        self.embeddings = nn.ModuleList([nn.Embedding(int(v), hidden_dim) for v in vocab_sizes])
        # Buffer, not a plain list: it has to follow the module to the GPU and be saved
        # with the checkpoint, or sampling would silently condition on the wrong rows.
        self.register_buffer("null_bins", torch.tensor(list(null_bins), dtype=torch.long))

    def forward(self, bins: torch.Tensor, force_null: bool = False) -> torch.Tensor:
        """bins: [B, P] int64. force_null replaces every property with its null bin,
        which is the unconditional pass classifier-free guidance needs."""
        if bins.dim() != 2 or bins.size(1) != self.n_properties:
            raise ValueError(f"expected bins of shape [B, {self.n_properties}], got {tuple(bins.shape)}")
        bins = bins.long()
        nulls = self.null_bins.to(bins.device).unsqueeze(0).expand_as(bins)

        if force_null:
            bins = nulls
        elif self.training and self.dropout_prob > 0:
            drop = torch.rand(bins.shape, device=bins.device) < self.dropout_prob
            bins = torch.where(drop, nulls, bins)

        return torch.stack([emb(bins[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


class ConditionalMoleculeGenerator(nn.Module):
    """HybridMoLA encoder over the framework + property-bin prefix tokens + SELFIES decoder.

    `mola` is a HybridMoLA; use_graph / use_smiles on its encoder select the ablation arm.
    """

    def __init__(
        self,
        mola: nn.Module,
        vocab_size: int,
        hidden_dim: int,
        pad_idx: int,
        cond_vocab_sizes: Sequence[int],
        cond_null_bins: Sequence[int],
        cond_dropout: float = 0.15,
        decoder_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder = mola.encoder
        if encoder.use_smiles and not encoder.positional_smiles:
            raise ValueError("ConditionalMoleculeGenerator needs positional_smiles=True whenever "
                             "the SMILES branch is enabled")
        self.mola = mola
        self.hidden_dim = hidden_dim
        self.condition = ConditionEmbedding(cond_vocab_sizes, cond_null_bins, hidden_dim, cond_dropout)
        self.decoder = MoLAGenerativeDecoder(
            vocab_size, hidden_dim, pad_idx,
            num_layers=decoder_layers, num_heads=num_heads, max_len=max_len, dropout=dropout,
        )

    def _memory(self, data, cond_bins: torch.Tensor, force_null: bool = False):
        raw = self.mola.encode_for_generation(data)
        cond = self.condition(cond_bins, force_null=force_null)
        return build_memory(raw, self.hidden_dim, condition=cond)

    def forward(self, data, decoder_input_ids: torch.Tensor, cond_bins: torch.Tensor) -> torch.Tensor:
        memory, memory_pad_mask = self._memory(data, cond_bins)
        return self.decoder(memory, memory_pad_mask, decoder_input_ids)

    @torch.no_grad()
    def guided_logits(self, data, decoder_input_ids: torch.Tensor, cond_bins: torch.Tensor,
                      guidance: float = 1.0) -> torch.Tensor:
        """Classifier-free guidance: extrapolate away from the unconditional prediction.

            logits = uncond + w * (cond - uncond)

        w = 1 is ordinary conditional sampling (one wasted forward pass); w > 1 pushes
        harder toward the requested delta. If sweeping w changes nothing, the decoder is
        ignoring the condition -- the free diagnostic this mechanism buys.
        """
        mem_c, pad_c = self._memory(data, cond_bins, force_null=False)
        cond_logits = self.decoder(mem_c, pad_c, decoder_input_ids)
        if guidance == 1.0:
            return cond_logits
        mem_u, pad_u = self._memory(data, cond_bins, force_null=True)
        uncond_logits = self.decoder(mem_u, pad_u, decoder_input_ids)
        return uncond_logits + guidance * (cond_logits - uncond_logits)

    @torch.no_grad()
    def generate_batch(self, data, cond_bins: torch.Tensor, vocab: Dict,
                       max_len: int = 96, temperature: float = 1.0, sample: bool = True,
                       guidance: float = 1.0) -> List[str]:
        """Batched autoregressive sampling. Decoding one molecule at a time is what made
        the previous evaluation slow enough to discourage sampling thousands."""
        from common.selfies_vocab import decode_ids

        self.eval()
        device = cond_bins.device
        batch_size = cond_bins.size(0)
        start_idx, end_idx = vocab["start_idx"], vocab["end_idx"]
        pad_idx = vocab["pad_idx"]
        temperature = float(temperature) if float(temperature) > 0 else 1.0

        tokens = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self.guided_logits(data, tokens, cond_bins, guidance=guidance)[:, -1] / temperature
            if sample:
                next_ids = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, pad_idx), next_ids)
            finished = finished | (next_ids == end_idx)
            tokens = torch.cat([tokens, next_ids.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break
        return [decode_ids(row[1:], vocab["id_to_token"]) for row in tokens.tolist()]


__all__ = ["ConditionalMoleculeGenerator", "ConditionEmbedding"]
