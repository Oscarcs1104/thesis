"""Pretrained SMILES language models (ChemBERTa, MoLFormer) exposing per-layer states.

MoLA fuses one pooled token per layer per modality, so a backbone used here has to give
up its intermediate hidden states, not just the last one. Transformers does that with
output_hidden_states=True -- but MoLFormer ships custom modeling code via
trust_remote_code, so support is checked at construction rather than assumed. If only the
final state is available the model still runs, with every MoLA slot fed by that one layer,
and says so loudly: silently collapsing the layer axis would make a "cross-layer fusion"
ablation that fuses nothing.

Pooling is masked mean over real tokens, matching how MoLA used MoLFormer.

    ChemBERTa   DeepChem/ChemBERTa-77M-MLM        ~77M params, 6 layers, hidden 384
    MoLFormer   ibm/MoLFormer-XL-both-10pct       linear-attention + rotary, needs
                                                  trust_remote_code=True
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

KNOWN_MODELS = {
    "chemberta": "DeepChem/ChemBERTa-77M-MLM",
    "molformer": "ibm/MoLFormer-XL-both-10pct",
}


def _load_tokenizer(model_name: str, trust_remote_code: bool):
    """Some trust_remote_code repos declare a custom tokenizer class through auto_map whose
    source file does not exist upstream. Fall back to the repo's own tokenizer.json."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except OSError:
        import json

        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = hf_hub_download(model_name, "tokenizer.json")
        special = {}
        try:
            with open(hf_hub_download(model_name, "special_tokens_map.json"), encoding="utf-8") as fh:
                special = {k: (v["content"] if isinstance(v, dict) else v) for k, v in json.load(fh).items()}
        except Exception:
            pass
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **special)


class PretrainedLanguageEncoder(nn.Module):
    """A frozen (or fine-tuned) SMILES transformer, pooled once per selected layer."""

    def __init__(
        self,
        hidden_dim: int,
        model_key: str = "chemberta",
        num_layers: int = 3,
        freeze: bool = True,
        max_length: int = 128,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name or KNOWN_MODELS.get(model_key, model_key)
        self.num_layers = num_layers
        self.max_length = max_length
        self.freeze = freeze

        kwargs: dict = {"trust_remote_code": True}
        if "molformer" in self.model_name.lower():
            # MoLFormer's own card: without this its pooled output is not reproducible.
            kwargs["deterministic_eval"] = True
        self.backbone = AutoModel.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = _load_tokenizer(self.model_name, trust_remote_code=True)

        lm_hidden = int(getattr(self.backbone.config, "hidden_size", hidden_dim))
        self.n_backbone_layers = int(getattr(self.backbone.config, "num_hidden_layers", 0))
        self.proj = nn.Linear(lm_hidden, hidden_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.supports_hidden_states = self._probe_hidden_states()
        if not self.supports_hidden_states:
            print(f"  [warn] {self.model_name} does not return per-layer hidden states; "
                  f"all {num_layers} MoLA slots will be fed by the final layer")

    def _probe_hidden_states(self) -> bool:
        """Ask the backbone once, at construction, instead of discovering mid-training."""
        try:
            enc = self.tokenizer(["CCO"], return_tensors="pt", padding=True, truncation=True,
                                 max_length=8)
            with torch.no_grad():
                out = self.backbone(**enc, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            return hs is not None and len(hs) > 1
        except Exception:
            return False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # keep dropout and norm statistics frozen
        return self

    def _layer_indices(self, n_available: int) -> List[int]:
        """Evenly spaced layers, always including the last. With 6 backbone layers and 3
        slots this takes 2, 4, 6 -- spreading the fusion across depth rather than reading
        three nearly identical top layers."""
        if n_available <= 1:
            return [0] * self.num_layers
        step = max(n_available // self.num_layers, 1)
        idx = [min(n_available - 1, (i + 1) * step - 1) for i in range(self.num_layers)]
        idx[-1] = n_available - 1
        return idx

    @staticmethod
    def _masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(states.dtype)
        return (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-9)

    def forward(self, smiles: Sequence[str], device: torch.device) -> List[torch.Tensor]:
        """Returns num_layers pooled tensors of shape [B, hidden_dim]."""
        texts = ["" if s is None else str(s) for s in smiles]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        mask = enc.get("attention_mask")

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.backbone(**enc, output_hidden_states=self.supports_hidden_states)

        if self.supports_hidden_states:
            hidden = out.hidden_states                       # (n_layers + 1) x [B, L, H]
            picks = self._layer_indices(len(hidden))
            pooled = [self._masked_mean(hidden[i], mask) for i in picks]
        else:
            pooled = [self._masked_mean(out.last_hidden_state, mask)] * self.num_layers

        # Detached above when frozen, so the projection is what carries the gradient.
        return [self.proj(p.detach() if self.freeze else p) for p in pooled]


__all__ = ["PretrainedLanguageEncoder", "KNOWN_MODELS"]
