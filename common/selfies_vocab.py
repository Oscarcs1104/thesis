from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

try:
    import selfies as _selfies
except Exception:
    _selfies = None


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<OTHER>"


def tokenize_molecule(text: str) -> List[str]:
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
        tokens.update(tokenize_molecule(text))

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
        for token in tokenize_molecule(text):
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
