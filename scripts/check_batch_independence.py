"""Does this model's output for one molecule depend on the others in its batch?

    python scripts/check_batch_independence.py

Self-contained and runs on either branch: it builds MoLA with exactly the arguments
crossmodal_model/benchmark/native_split.py passes, feeds it synthetic molecules, and
compares one molecule's prediction scored alone against the same molecule scored inside a
batch. No data, no checkpoint, no training -- only the architecture.

Reading it costs nothing either way. If the two predictions are identical, the model is
independent per molecule and any claim to the contrary is wrong. If they differ, the
reported RMSE is a property of how the test set was batched as well as of the model, and
a single molecule cannot be scored on its own.

The mechanism, for whoever wants to check it by eye rather than by running this: MoLA
takes positional_smiles=False by default and native_split.py does not override it. Under
that setting sm_embed leaves the embedding as [B, L, H] and enters a
TransformerEncoderLayer built with batch_first=False, which reads its input as
[seq, batch, feature]. Attention therefore runs along B -- the molecule axis -- so each
molecule attends to the others at a fixed character position instead of across its own
characters. Passing positional_smiles=True makes the layer batch_first and the attention
runs where the name says it does.

No labels cross between molecules under either setting. This is not leakage of y.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_geometric.data import Batch, Data  # noqa: E402

from crossmodal_model.model.mola import MoLA  # noqa: E402

GRAPH_DIM, VOCAB, HIDDEN, LAYERS, MAX_SM_LEN = 30, 40, 256, 3, 100


def fake_molecule(gen: torch.Generator, n_atoms: int) -> Data:
    edges = torch.stack([
        torch.arange(n_atoms - 1),
        torch.arange(1, n_atoms),
    ])
    edges = torch.cat([edges, edges.flip(0)], dim=1)          # undirected
    sm = torch.randint(1, VOCAB, (MAX_SM_LEN,), generator=gen)
    return Data(
        x=torch.rand(n_atoms, GRAPH_DIM, generator=gen),
        edge_index=edges,
        sm=sm.unsqueeze(0),
        y=torch.zeros(1),
    )


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    mols = [fake_molecule(gen, 8 + i) for i in range(8)]

    for positional in (False, True):
        torch.manual_seed(1)
        # Exactly native_split.py's call, plus the one argument it does not pass.
        model = MoLA(
            graph_dim=GRAPH_DIM, sm_vocab_size=VOCAB, hidden_dim=HIDDEN,
            output_dim=1, num_layers=LAYERS,
            positional_smiles=positional, max_sm_len=MAX_SM_LEN,
        ).eval()

        with torch.no_grad():
            alone = model(Batch.from_data_list(mols[:1]))[3].view(-1)[0].item()
            in_batch = model(Batch.from_data_list(mols))[3].view(-1)[0].item()
            swapped = list(mols)
            swapped[3] = fake_molecule(torch.Generator().manual_seed(99), 11)
            neighbour = model(Batch.from_data_list(swapped))[3].view(-1)[0].item()

        label = "positional_smiles=True (batch_first)" if positional else \
                "positional_smiles=False  <- native_split.py's default"
        print(f"\n  {label}")
        print(f"    molecule 0 scored alone          : {alone:+.6f}")
        print(f"    molecule 0 inside a batch of 8   : {in_batch:+.6f}")
        print(f"    same, after replacing molecule 3 : {neighbour:+.6f}")
        d1, d2 = abs(alone - in_batch), abs(in_batch - neighbour)
        print(f"    |alone - batched|                : {d1:.6f}")
        print(f"    |batched - neighbour swapped|    : {d2:.6f}")
        verdict = ("independent per molecule" if max(d1, d2) < 1e-6
                   else "DEPENDS on the rest of the batch")
        print(f"    -> {verdict}")

    print("\n  Both runs use the same weights and the same molecule. The only thing that")
    print("  changes is what else is in the batch.")


if __name__ == "__main__":
    main()
